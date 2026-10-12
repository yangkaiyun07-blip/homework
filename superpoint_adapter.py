"""Adapters using the SAME official SuperPoint for NN and SuperGlue."""
import importlib.util
import cv2
import numpy as np


def load_class(path, name):
    spec=importlib.util.spec_from_file_location('_matching_'+path.stem,str(path))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module,name)


def resize_image(im, max_side):
    h,w=im.shape
    scale=min(1.,max_side/max(h,w)) if max_side else 1.
    nw=max(8,int(w*scale)//8*8);nh=max(8,int(h*scale)//8*8)
    return cv2.resize(im,(nw,nh)),np.array([w/nw,h/nh])


def nn_matches(d0,d1,ratio,mutual):
    if len(d0)==0 or len(d1)==0 or (ratio and len(d1)<2):
        return []
    bf=cv2.BFMatcher(cv2.NORM_L2)
    if ratio:
        good=[p[0] for p in bf.knnMatch(d0,d1,k=2) if len(p)==2 and p[0].distance<ratio*p[1].distance]
    else:
        good=list(bf.match(d0,d1))
    if mutual:
        reverse={m.queryIdx:m.trainIdx for m in bf.match(d1,d0)}
        good=[m for m in good if reverse.get(m.trainIdx)==m.queryIdx]
    return sorted(good,key=lambda m:m.distance)


class SuperPointAdapter:
    def __init__(self,source,method,device,max_side,max_keypoints,ratio,mutual,threshold,sg_threshold):
        names=['superpoint_v1.pth']
        if method=='superpoint_superglue':
            names.append('superglue_outdoor.pth')
        for name in names:
            path=source/'models'/'weights'/name
            if not path.is_file() or path.stat().st_size<1024:
                raise FileNotFoundError(f'Missing/invalid pretrained weights: {path}. Clone official SuperGluePretrainedNetwork including models/weights.')
        import torch
        self.torch=torch
        self.device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
        SP=load_class(source/'models'/'superpoint.py','SuperPoint')
        self.sp=SP(dict(nms_radius=4,keypoint_threshold=threshold,max_keypoints=max_keypoints or -1)).eval().to(self.device)
        self.sg=None
        if method=='superpoint_superglue':
            SG=load_class(source/'models'/'superglue.py','SuperGlue')
            self.sg=SG(dict(weights='outdoor',sinkhorn_iterations=20,match_threshold=sg_threshold)).eval().to(self.device)
        self.max_side=max_side;self.ratio=ratio;self.mutual=mutual

    def sync(self):
        if self.device.type=='cuda':
            self.torch.cuda.synchronize(self.device)

    def __call__(self,im0,im1):
        arrays=[];scales=[]
        for im in (im0,im1):
            resized,scale=resize_image(im,self.max_side)
            arrays.append(self.torch.from_numpy(resized).float()[None,None].to(self.device)/255.)
            scales.append(scale)
        with self.torch.inference_mode():
            feats=[self.sp({'image':im}) for im in arrays]
            coords=[f['keypoints'][0].cpu().numpy()*scale for f,scale in zip(feats,scales)]
            if self.sg is None:
                desc=[np.ascontiguousarray(f['descriptors'][0].T.cpu().numpy(),dtype=np.float32) for f in feats]
                matches=nn_matches(*desc,self.ratio,self.mutual)
            elif any(len(x)==0 for x in coords):
                matches=[]
            else:
                data={f'image{i}':im for i,im in enumerate(arrays)}
                for i,f in enumerate(feats):
                    data.update({k+str(i):v[0][None] for k,v in f.items()})
                result=self.sg(data)
                indices=result['matches0'][0].cpu().numpy()
                scores=result['matching_scores0'][0].cpu().numpy()
                matches=[cv2.DMatch(int(i),int(j),float(1-scores[i])) for i,j in enumerate(indices) if j>=0]
                matches.sort(key=lambda m:m.distance)
        keypoints=[[cv2.KeyPoint(float(x),float(y),1.) for x,y in p] for p in coords]
        return *keypoints,matches,dict(keypoints0=len(coords[0]),keypoints1=len(coords[1]))
