"""Ground-truth epipolar precision and angular relative pose evaluation."""
import cv2
import numpy as np
from baseline import angle_errors


def sampson_distance(p0,p1,K0,K1,T):
    if not len(p0):
        return np.empty(0)
    x,y,z=T[:3,3]
    E=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])@T[:3,:3]
    F=np.linalg.inv(K1).T@E@np.linalg.inv(K0)
    a=np.column_stack((p0,np.ones(len(p0))))
    b=np.column_stack((p1,np.ones(len(p1))))
    l1=a@F.T;l0=b@F
    denominator=(l1[:,:2]**2).sum(1)+(l0[:,:2]**2).sum(1)
    result=np.full(len(p0),np.inf)
    valid=denominator>1e-20
    result[valid]=np.abs((b*l1).sum(1)[valid])/np.sqrt(denominator[valid])
    return result


def evaluate(p0,p1,K0,K1,T,ransac_px=1.,epipolar_px=3.):
    distances=sampson_distance(p0,p1,K0,K1,T)
    good=distances<=epipolar_px
    row=dict(matches=len(p0),epipolar_good=int(good.sum()),epipolar_precision=float(good.mean()) if len(good) else 0.,
             median_sampson_px=float(np.median(distances)) if len(distances) and np.isfinite(np.median(distances)) else None,
             pose_estimated=False,rotation_error_deg=None,translation_error_deg=None,pose_error_deg=None,
             ransac_inliers=0,ransac_inlier_ratio=0.,cheirality_inliers=0,
             success_5deg=False,success_10deg=False,success_20deg=False)
    mask=np.zeros(len(p0),bool)
    if len(p0)<5:
        return row,distances,mask
    q0=cv2.undistortPoints(p0[:,None,:].astype(float),K0,None).reshape(-1,2)
    q1=cv2.undistortPoints(p1[:,None,:].astype(float),K1,None).reshape(-1,2)
    focal=np.mean([K0[0,0],K0[1,1],K1[0,0],K1[1,1]])
    try:
        E,rin=cv2.findEssentialMat(q0,q1,np.eye(3),method=cv2.USAC_MAGSAC,prob=.999,
                                   threshold=ransac_px/focal,maxIters=10000)
        if E is None or rin is None:
            return row,distances,mask
        row.update(ransac_inliers=int((rin!=0).sum()),ransac_inlier_ratio=float((rin!=0).mean()))
        best=None
        for e in np.asarray(E).reshape(-1,3,3):
            n,R,t,cin=cv2.recoverPose(e,q0,q1,np.eye(3),mask=rin.copy())
            if best is None or n>best[0]:
                best=(n,R,t.ravel(),cin.ravel()!=0)
        if best is None or best[0]<5:
            return row,distances,mask
        n,R,t,mask=best
        rerr,terr=angle_errors(R,t,T)
        error=max(rerr,terr) if terr is not None else None
        row.update(pose_estimated=True,rotation_error_deg=rerr,translation_error_deg=terr,pose_error_deg=error,
                   cheirality_inliers=int(n),R_est=R.tolist(),t_direction=t.tolist())
        for threshold in (5,10,20):
            row[f'success_{threshold}deg']=error is not None and error<threshold
    except cv2.error as error:
        row['geometry_error']=str(error)
    return row,distances,mask


def auc(errors,threshold):
    """Standard trapezoidal interpolation of sorted pose error / recall points."""
    errors=np.sort(np.asarray(errors,float))
    if not len(errors):
        return 0.
    recall=np.arange(1,len(errors)+1)/len(errors)
    n=np.searchsorted(errors,threshold)
    xs=np.r_[0.,errors[:n],threshold]
    ys=np.r_[0.,recall[:n],recall[n-1] if n else 0.]
    integrate=getattr(np,'trapezoid',np.trapz)
    return float(integrate(ys,xs)/threshold)


def summarize(rows):
    errors=[r['pose_error_deg'] if r.get('pose_error_deg') is not None else np.inf for r in rows]
    timed=[r['matching_ms'] for r in rows if r.get('matching_ms') is not None]
    geometry_times=[r['geometry_ms'] for r in rows if r.get('geometry_ms') is not None]
    total_matches=sum(r.get('matches',0) for r in rows)
    out=dict(pairs=len(rows),pose_estimated_count=sum(r.get('pose_estimated',False) for r in rows),
             median_matches=float(np.median([r.get('matches',0) for r in rows])) if rows else 0.,
             macro_epipolar_precision_pct=100*float(np.mean([r.get('epipolar_precision',0.) for r in rows])) if rows else 0.,
             micro_epipolar_precision_pct=100*sum(r.get('epipolar_good',0) for r in rows)/max(1,total_matches),
             mean_ransac_inlier_pct=100*float(np.mean([r.get('ransac_inlier_ratio',0.) for r in rows])) if rows else 0.,
             median_matching_ms=float(np.median(timed)) if timed else None,
             mean_matching_ms=float(np.mean(timed)) if timed else None,
             median_geometry_ms=float(np.median(geometry_times)) if geometry_times else None,
             timed_pairs=len(timed),failed_input_pairs=sum(r.get('input_error') is not None for r in rows))
    for threshold in (5,10,20):
        out[f'success_{threshold}deg_pct']=100*sum(e<threshold for e in errors)/len(rows) if rows else 0.
        out[f'auc_{threshold}deg_pct']=100*auc(errors,threshold)
    return out
