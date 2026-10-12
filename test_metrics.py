import unittest
import numpy as np
import cv2
from metrics import evaluate,sampson_distance,auc,summarize
class MetricTests(unittest.TestCase):
    def test_gt_and_pose(self):
        rng=np.random.default_rng(42)
        X=rng.uniform([-2,-1,4],[2,1,10],(200,3))
        K0=np.array([[600.,0,320],[0,620,240],[0,0,1]])
        K1=np.array([[700.,0,310],[0,710,230],[0,0,1]])
        T=np.eye(4);T[:3,:3]=cv2.Rodrigues(np.array([.02,.05,.01]))[0];T[:3,3]=[.5,.02,.03]
        def project(X,K):
            p=X@K.T;return p[:,:2]/p[:,2,None]
        a=project(X,K0);b=project(X@T[:3,:3].T+T[:3,3],K1)
        result,errors,mask=evaluate(a,b,K0,K1,T)
        self.assertTrue(result['pose_estimated'])
        self.assertLess(result['pose_error_deg'],.1)
        self.assertLess(errors.max(),1e-8)
        shifted=b.copy();shifted[:,1]+=50
        self.assertGreater(np.median(sampson_distance(a,shifted,K0,K1,T)),3)
        zero=T.copy();zero[:3,3]=0
        self.assertTrue(np.isinf(sampson_distance(a,b,K0,K1,zero)).all())
    def test_failure_auc_and_precision(self):
        good=dict(pose_error_deg=1,pose_estimated=True,matches=10,epipolar_good=8,epipolar_precision=.8)
        bad=dict(pose_error_deg=None,matches=0,pose_estimated=False)
        s=summarize([good,bad])
        self.assertEqual(s['success_5deg_pct'],50.)
        self.assertEqual(s['macro_epipolar_precision_pct'],40.)
        self.assertEqual(s['micro_epipolar_precision_pct'],80.)
        self.assertAlmostEqual(auc([1,np.inf],5),.45)
        self.assertEqual(auc([np.inf],5),0.)
if __name__=='__main__':unittest.main()
