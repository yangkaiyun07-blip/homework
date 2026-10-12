#!/usr/bin/env python3
"""prepare -> fetch -> run -> analyze: matching on official RUBIK pairs."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import cv2
import numpy as np
from baseline import match_images,read_pairs
from dataset import image_index
from metrics import evaluate,summarize

ROOT=Path(__file__).resolve().parent
METHODS=['sift_nn','superpoint_nn','superpoint_superglue','eloftr']


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    temp.replace(path)


def save_csv(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for r in rows for k in r if k not in ('R_est','t_direction')))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def fetch(a):
    # Official pretrained models; ELoFTR uses the existing RUBIK source and a user-supplied checkpoint.
    base='https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/'
    names=['models/__init__.py','models/superpoint.py','models/superglue.py',
           'models/weights/superpoint_v1.pth','models/weights/superglue_outdoor.pth','LICENSE']
    records=[]
    for name in names:
        target=a.model_dir/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            print('Downloading',name,flush=True)
            temp=target.with_suffix(target.suffix+'.part')
            with urllib.request.urlopen(base+name,timeout=90) as src,temp.open('wb') as out:
                while True:
                    block=src.read(1024*1024)
                    if not block:break
                    out.write(block)
            temp.replace(target)
        data=target.read_bytes()
        if name.endswith('.pth') and len(data)<1000000:
            raise ValueError(f'Weight file too small: {target}. Remove invalid file and retry.')
        records.append(dict(file=name,url=base+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    save_json(a.model_dir/'sources.json',records)
    print('Ready:',a.model_dir)


def prepare(a):
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Preparation directory is not empty; use a new --out')
    if a.per_group<0:raise ValueError('--per-group must be nonnegative')
    pairs=read_pairs(a.rubik,a.per_group)
    if not pairs:raise ValueError('Empty RUBIK manifest')
    index=image_index(a.data_root,pairs,a.metadata_root)
    missing=[r['filename'] for r in index.values() if not (a.data_root/r['filename']).is_file()]
    a.out.mkdir(parents=True,exist_ok=True)
    audit=dict(source='official RUBIK',rubik_sha256=hashlib.sha256(a.rubik.read_bytes()).hexdigest(),
               selected_pairs=len(pairs),groups=len({p['group'] for p in pairs}),required_unique_images=len(index),
               missing_images=len(missing),per_group=a.per_group,
               calibration='K1/K2 and rel_pose from RUBIK; metadata used only for file paths/dimensions')
    save_json(a.out/'data_check.json',audit)
    (a.out/'missing_images.txt').write_text('\n'.join(missing))
    print(json.dumps(audit,indent=2),flush=True)
    if missing:raise ValueError('Selected images missing; inspect missing_images.txt. No pairs silently dropped.')
    selected={}
    for pair in pairs:
        selected.setdefault(pair['group'],{}).setdefault(pair['scene'],{})[str((pair['image0'],pair['image1']))]=pair['info']
    save_json(a.out/'rubik_pairs.json',selected)


def load_images(pair,index,root,width,height):
    images=[];intrinsics=[]
    for j in (0,1):
        row=index[pair[f'image{j}']]
        path=root/row['filename']
        if not path.is_file():raise FileNotFoundError(str(path))
        im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if im is None:raise ValueError(f'Cannot decode {path}')
        h,w=im.shape
        if (h,w)!=(row['height'],row['width']):raise ValueError(f'Wrong image dimensions: {path}')
        intrinsics.append(np.diag([width/w,height/h,1.])@np.array(pair['info'][f'K{j+1}'],float))
        images.append(cv2.resize(im,(width,height),interpolation=cv2.INTER_AREA))
    return images,intrinsics


def matcher_for(method,a):
    if method=='sift_nn':
        return None,'cpu'
    if method=='eloftr':
        from eloftr_adapter import ELoFTRAdapter
        if a.rubik_root is None or a.checkpoint is None:
            raise ValueError('ELoFTR requires --rubik-root and --checkpoint')
        adapter=ELoFTRAdapter(a.rubik_root,a.checkpoint,a.device,max(a.width,a.height))
    else:
        from superpoint_adapter import SuperPointAdapter
        adapter=SuperPointAdapter(a.model_dir,method,a.device,0,a.nfeatures,a.ratio,a.mutual,a.sp_threshold,a.sg_threshold)
    return adapter,str(adapter.device)


def run(a):
    if a.width<=0 or a.height<=0 or a.width%32 or a.height%32:
        raise ValueError('Use positive width/height divisible by 32 (e.g. 800x448) for all methods')
    if a.nfeatures<0 or not 0<=a.ratio<1 or a.ransac_px<=0 or a.epipolar_px<=0 or a.warmup<0 or a.visualize<0:
        raise ValueError('Invalid matching/evaluation settings')
    if len(set(a.methods))!=len(a.methods):raise ValueError('Duplicate methods')
    pairs=read_pairs(a.pairs,0)
    if not pairs:raise ValueError('Empty pair manifest')
    index=image_index(a.data_root,pairs,a.metadata_root)
    common=dict(pairs_sha256=hashlib.sha256(a.pairs.read_bytes()).hexdigest(),width=a.width,height=a.height,
                ransac_px=a.ransac_px,epipolar_px=a.epipolar_px,geometry='USAC_MAGSAC',seed=a.seed)
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=common:
        raise ValueError('Output uses a different pair list/evaluation protocol. Choose another --out.')
    for method in a.methods:
        d=a.out/method
        if d.exists() and any(d.iterdir()):raise ValueError(f'{d} already has output. Choose a new --out.')
    save_json(path,common)
    save_json(a.out/'selected_pairs.json',pairs)
    cv2.setNumThreads(1)
    # Find one readable warmup pair without dropping any failed input from evaluation.
    warm_images=None
    if a.warmup:
        for pair in pairs:
            try:warm_images,_=load_images(pair,index,a.data_root,a.width,a.height);break
            except (KeyError,ValueError,OSError):continue
    for method in a.methods:
        adapter,device=matcher_for(method,a)
        d=a.out/method;d.mkdir(parents=True,exist_ok=True)
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='func'}
        config.update(method=method,device_actual=device,python=sys.version,numpy=np.__version__,opencv=cv2.__version__)
        if adapter is not None:
            config['torch']=adapter.torch.__version__
            config['hardware']=adapter.torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU'
        else:config['hardware']='CPU'
        if adapter is not None:
            weight_paths = [a.checkpoint] if method=='eloftr' else [a.model_dir/'models/weights/superpoint_v1.pth']
            if method=='superpoint_superglue':weight_paths.append(a.model_dir/'models/weights/superglue_outdoor.pth')
            config['weight_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in weight_paths}
        save_json(d/'config.json',config)
        def match(images):
            return adapter(*images) if adapter is not None else match_images(*images,a.nfeatures,a.ratio,a.mutual)
        if warm_images is not None:
            for _ in range(a.warmup):match(warm_images)
            if adapter is not None:adapter.sync()
        rows=[];visualized=0
        for i,pair in enumerate(pairs):
            cv2.setRNGSeed(a.seed+i)
            row=dict(pair_id=i,method=method,scene=pair['scene'],group=pair['group'],
                     image0=pair['image0'],image1=pair['image1'],execution_device=device,matching_ms=None,geometry_ms=None)
            try:images,Ks=load_images(pair,index,a.data_root,a.width,a.height)
            except (KeyError,ValueError,OSError) as error:
                row.update(input_error=str(error),matches=0,pose_estimated=False,pose_error_deg=None)
                rows.append(row)
                continue
            if adapter is not None:adapter.sync()
            start=time.perf_counter()
            k0,k1,matches,counts=match(images)
            if adapter is not None:adapter.sync()
            row.update(counts,matching_ms=1000*(time.perf_counter()-start))
            p0=np.array([k0[m.queryIdx].pt for m in matches],float).reshape(-1,2)
            p1=np.array([k1[m.trainIdx].pt for m in matches],float).reshape(-1,2)
            start=time.perf_counter()
            stats,dist,valid=evaluate(p0,p1,*Ks,np.array(pair['info']['rel_pose'],float),a.ransac_px,a.epipolar_px)
            row.update(stats,geometry_ms=1000*(time.perf_counter()-start))
            np.savez_compressed(d/f'pair_{i:05d}.npz',points0=p0,points1=p1,sampson_px=dist,
                                cheirality_inliers=valid,image_size=[a.width,a.height])
            if visualized<a.visualize:
                canvas=cv2.cvtColor(np.hstack(images),cv2.COLOR_GRAY2BGR)
                rng=np.random.default_rng(a.seed+i)
                for j in rng.choice(len(matches),min(100,len(matches)),replace=False):
                    startpt=tuple(np.rint(p0[j]).astype(int));endpt=tuple(np.rint(p1[j]+[a.width,0]).astype(int))
                    cv2.line(canvas,startpt,endpt,(0,200,0) if dist[j]<=a.epipolar_px else (0,0,255),1)
                cv2.imwrite(str(d/f'pair_{i:05d}.jpg'),canvas);visualized+=1
            rows.append(row)
            if (i+1)%25==0:
                save_json(d/'partial_results.json',rows)
                print(method,i+1,'/',len(pairs),flush=True)
        save_json(d/'results.json',rows);save_csv(d/'results.csv',rows)
        summary=summarize(rows);summary['method']=method
        save_json(d/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        del adapter
    print('Run complete. Use analyze to create comparison plots.')


def analyze(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=a.methods or [m for m in METHODS if (a.out/m/'results.json').exists()]
    if not methods:raise ValueError('No completed results found')
    pairs=json.loads((a.out/'selected_pairs.json').read_text())
    rows={m:json.loads((a.out/m/'results.json').read_text()) for m in methods}
    expected=set(range(len(pairs)))
    for m,rs in rows.items():
        if len(rs)!=len(pairs) or {r['pair_id'] for r in rs}!=expected:raise ValueError(f'Incomplete/duplicate pair results: {m}')
    summary=[dict(method=m,**summarize(rs)) for m,rs in rows.items()]
    save_json(a.out/'summary.json',summary);save_csv(a.out/'summary.csv',summary)
    save_csv(a.out/'all_pairs.csv',[r for rs in rows.values() for r in rs])
    grouped=[]
    for group in sorted({p['group'] for p in pairs}):
        for m,rs in rows.items():grouped.append(dict(group=group,method=m,**summarize([r for r in rs if r['group']==group])))
    save_csv(a.out/'grouped_results.csv',grouped)
    comparisons=[]
    for i,m0 in enumerate(methods):
        for m1 in methods[i+1:]:
            r0={r['pair_id']:r.get('success_10deg',False) for r in rows[m0]}
            r1={r['pair_id']:r.get('success_10deg',False) for r in rows[m1]}
            comparisons.append(dict(baseline=m0,comparison=m1,both_success=sum(r0[k] and r1[k] for k in expected),
                                    baseline_only=sum(r0[k] and not r1[k] for k in expected),
                                    comparison_only=sum(r1[k] and not r0[k] for k in expected),
                                    both_fail=sum(not r0[k] and not r1[k] for k in expected)))
    save_csv(a.out/'paired_comparisons.csv',comparisons)
    figures=a.out/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key,title in zip(axes,['success_10deg_pct','macro_epipolar_precision_pct','auc_10deg_pct'],
                           ['Pose success @10 deg (%)','Macro epipolar precision (%)','Pose AUC @10 deg (%)']):
        ax.bar(methods,[s[key] for s in summary]);ax.set_title(title);ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(figures/'overview.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for m,rs in rows.items():
        errors=np.sort([r['pose_error_deg'] if r.get('pose_error_deg') is not None else np.inf for r in rs])
        valid=errors[errors<=30]
        ax.step(np.r_[0,valid,30],np.r_[0,np.arange(1,len(valid)+1)/len(rs)*100,len(valid)/len(rs)*100],where='post',label=m)
    ax.set(xlabel='Pose error (degrees)',ylabel='Recall (%)',xlim=(0,30),ylim=(0,100));ax.legend();fig.tight_layout()
    fig.savefig(figures/'pose_recall.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    groups=sorted({p['group'] for p in pairs})
    for m in methods:
        values=[next(g['success_10deg_pct'] for g in grouped if g['group']==group and g['method']==m) for group in groups]
        ax.plot(range(1,len(groups)+1),values,'o-',label=m)
    ax.set(xlabel='RUBIK group index (see grouped_results.csv order)',ylabel='Pose success @10 deg (%)',ylim=(0,100));ax.legend()
    fig.tight_layout();fig.savefig(figures/'by_group.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    devices=[json.loads((a.out/m/'config.json').read_text())['device_actual'] for m in methods]
    ax.bar([m+'\n'+dev for m,dev in zip(methods,devices)],[s['median_matching_ms'] or 0 for s in summary])
    ax.set(ylabel='Median matching time (ms)',title='Device labels shown; CPU/GPU times are not hardware-normalized')
    ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(figures/'timing.png',dpi=160);plt.close(fig)
    # Aggregate official three-dimensional bins along each difficulty axis.
    ranges={i:ast.literal_eval(pair['group']) for i,pair in enumerate(pairs)}
    difficulty=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for axis,(ax,name) in enumerate(zip(axes,['overlap','scale_ratio','viewpoint_angle'])):
        bins=sorted({tuple(v[axis]) for v in ranges.values()})
        for m,rs in rows.items():
            values=[]
            for lo,hi in bins:
                subset=[r for r in rs if tuple(ranges[r['pair_id']][axis])==(lo,hi)]
                summary_group=summarize(subset)
                difficulty.append(dict(axis=name,lower=lo,upper=hi,method=m,**summary_group))
                values.append(summary_group['success_10deg_pct'])
            ax.plot(range(len(bins)),values,'o-',label=m)
        ax.set_xticks(range(len(bins)),[f'{lo:g}-{hi:g}' for lo,hi in bins],rotation=30)
        ax.set(title=name,ylabel='Pose success @10 deg (%)',ylim=(0,100))
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figures/'difficulty.png',dpi=160);plt.close(fig)
    save_csv(a.out/'difficulty_results.csv',difficulty)
    print('Saved comparison tables and plots to',a.out)


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.set_defaults(func=prepare)
    prep.add_argument('--data-root',type=Path,required=True);prep.add_argument('--metadata-root',type=Path)
    prep.add_argument('--out',type=Path,default=ROOT/'pairs')
    prep.add_argument('--rubik',type=Path,required=True)
    prep.add_argument('--per-group',type=int,default=0,help='0=all official pairs; otherwise fixed evenly-spaced subset per group')
    f=sub.add_parser('fetch');f.set_defaults(func=fetch);f.add_argument('--model-dir',type=Path,default=ROOT/'models/SuperGluePretrainedNetwork')
    r=sub.add_parser('run');r.set_defaults(func=run)
    r.add_argument('--data-root',type=Path,required=True);r.add_argument('--metadata-root',type=Path)
    r.add_argument('--pairs',type=Path,default=ROOT/'pairs/rubik_pairs.json');r.add_argument('--out',type=Path,default=ROOT/'results')
    r.add_argument('--methods',nargs='+',choices=METHODS,default=['sift_nn'])
    r.add_argument('--model-dir',type=Path,default=ROOT/'models/SuperGluePretrainedNetwork')
    r.add_argument('--rubik-root',type=Path);r.add_argument('--checkpoint',type=Path)
    r.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    r.add_argument('--width',type=int,default=800);r.add_argument('--height',type=int,default=448)
    r.add_argument('--nfeatures',type=int,default=4096);r.add_argument('--ratio',type=float,default=.75)
    r.add_argument('--mutual',action='store_true');r.add_argument('--sp-threshold',type=float,default=.005)
    r.add_argument('--sg-threshold',type=float,default=.2)
    r.add_argument('--ransac-px',type=float,default=1.);r.add_argument('--epipolar-px',type=float,default=3.)
    r.add_argument('--visualize',type=int,default=20);r.add_argument('--warmup',type=int,default=3)
    r.add_argument('--seed',type=int,default=20261012)
    z=sub.add_parser('analyze');z.set_defaults(func=analyze);z.add_argument('--out',type=Path,default=ROOT/'results')
    z.add_argument('--methods',nargs='+',choices=METHODS)
    return p

if __name__=='__main__':
    a=parser().parse_args()
    a.func(a)
