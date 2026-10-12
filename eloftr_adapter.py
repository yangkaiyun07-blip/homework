"""Adapter for the EfficientLoFTR source bundled with RUBIK (lazy dependencies)."""
import sys
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np


def resize_for_eloftr(im, max_side):
    h,w=im.shape
    scale=min(1.,max_side/max(h,w)) if max_side else 1.
    nw=max(32,int(w*scale)//32*32)
    nh=max(32,int(h*scale)//32*32)
    return cv2.resize(im,(nw,nh)),np.array([w/nw,h/nh])


class ELoFTRAdapter:
    def __init__(self, repo_root, checkpoint, device='auto', max_side=1024):
        if not checkpoint.is_file():
            raise FileNotFoundError(f'ELoFTR checkpoint missing: {checkpoint}. Obtain eloftr_outdoor.ckpt using EfficientLoFTR/README.md.')
        source=repo_root/'detector_free'/'EfficientLoFTR'
        if not (source/'src/loftr/loftr.py').is_file():
            raise FileNotFoundError(f'EfficientLoFTR source missing: {source}')
        import torch
        sys.path.insert(0,str(source.resolve()))
        # A separate CLI process per method prevents conflicts with other projects named src.
        from src.loftr import LoFTR, full_default_cfg, reparameter
        self.torch=torch
        self.device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
        config=deepcopy(full_default_cfg)
        config['coarse']['npe']=[832,832,max_side or 1600,max_side or 1600]
        self.model=LoFTR(config=config)
        state=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
        self.model.load_state_dict(state.get('state_dict',state))
        self.model=reparameter(self.model).eval().to(self.device)
        self.max_side=max_side

    def sync(self):
        if self.device.type=='cuda':
            self.torch.cuda.synchronize(self.device)

    def __call__(self,im0,im1):
        a,scale0=resize_for_eloftr(im0,self.max_side)
        b,scale1=resize_for_eloftr(im1,self.max_side)
        batch={'image0':self.torch.from_numpy(a).float()[None,None].to(self.device)/255.,
               'image1':self.torch.from_numpy(b).float()[None,None].to(self.device)/255.}
        with self.torch.inference_mode():
            self.model(batch)
        p0=batch['mkpts0_f'].cpu().numpy()*scale0
        p1=batch['mkpts1_f'].cpu().numpy()*scale1
        # KeyPoint wrappers are solely for the common visualization interface;
        # ELoFTR itself does not detect SIFT keypoints.
        k0=[cv2.KeyPoint(float(x),float(y),1.) for x,y in p0]
        k1=[cv2.KeyPoint(float(x),float(y),1.) for x,y in p1]
        conf=batch['mconf'].cpu().numpy() if 'mconf' in batch else np.ones(len(p0))
        matches=[cv2.DMatch(i,i,float(1-c)) for i,c in enumerate(conf)]
        return k0,k1,sorted(matches,key=lambda m:m.distance),{}
