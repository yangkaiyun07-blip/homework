import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import cv2
import numpy as np
from dataset import image_index
from baseline import read_pairs

class OfficialPipelineTests(unittest.TestCase):
    def test_official_prepare_run_analyze(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';data.mkdir()
            names=['sample__CAM_FRONT__1.jpg','sample__CAM_FRONT__2.jpg']
            rng=np.random.default_rng(42)
            im=cv2.GaussianBlur(rng.integers(0,256,(900,1600),dtype=np.uint8),(3,3),0)
            shifted=cv2.warpAffine(im,np.float32([[1,0,10],[0,1,5]]),(1600,900))
            for folder,name,image in zip(['samples','sweeps'],names,[im,shifted]):
                file=data/folder/'CAM_FRONT'/name;file.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(file),image)
            K=[[1200,0,800],[0,1200,450],[0,0,1]]
            T=np.eye(4);T[0,3]=1.
            group='[[0.2, 0.4], [1, 1.5], [0, 30]]'
            manifest={group:{'scene-test':{str(tuple(names)):dict(K1=K,K2=K,rel_pose=T.tolist(),overlap=.3,distance=1.2,los_angle=10)}}}
            source=root/'rubik.json';source.write_text(json.dumps(manifest))
            script=Path(__file__).with_name('experiment.py')
            def call(*args):
                result=subprocess.run([sys.executable,str(script),*map(str,args)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            call('prepare','--data-root',data,'--rubik',source,'--out',root/'pairs')
            prepared=root/'pairs/rubik_pairs.json'
            self.assertEqual(json.loads(prepared.read_text()),manifest)
            audit=json.loads((root/'pairs/data_check.json').read_text());self.assertEqual(audit['missing_images'],0)
            index=image_index(data,read_pairs(prepared,0))
            self.assertTrue(index[names[0]]['filename'].startswith('samples/'))
            self.assertTrue(index[names[1]]['filename'].startswith('sweeps/'))
            call('run','--data-root',data,'--pairs',prepared,'--out',root/'results','--nfeatures','500','--width','800','--height','448','--warmup','1')
            rows=json.loads((root/'results/sift_nn/results.json').read_text())
            self.assertEqual(len(rows),1);self.assertGreater(rows[0]['matches'],20)
            call('analyze','--out',root/'results')
            for name in ['summary.csv','difficulty_results.csv','figures/difficulty.png','figures/overview.png']:
                self.assertTrue((root/'results'/name).is_file())
            (data/index[names[1]]['filename']).unlink()
            failed=subprocess.run([sys.executable,str(script),'prepare','--data-root',str(data),'--rubik',str(source),'--out',str(root/'missing')],capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertFalse((root/'missing/rubik_pairs.json').exists())
            self.assertEqual(json.loads((root/'missing/data_check.json').read_text())['missing_images'],1)

    def test_selection_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'rubik.json'
            entries={str((f'x__CAM_FRONT__{i}.jpg',f'x__CAM_FRONT__{i+1}.jpg')):dict(K1=[],K2=[],rel_pose=[]) for i in range(10)}
            path.write_text(json.dumps({'[[0,1],[1,2],[0,30]]':{'scene':entries}}))
            self.assertEqual(read_pairs(path,3),read_pairs(path,3))
            self.assertEqual(len(read_pairs(path,3)),3)
            self.assertEqual(len(read_pairs(path,0)),10)
if __name__=='__main__':unittest.main()
