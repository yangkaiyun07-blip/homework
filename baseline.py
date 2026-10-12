"""Original SIFT+NN implementation and shared helpers."""
from pathlib import Path
import ast
import json
import cv2
import numpy as np

def read_pairs(path, per_group):
    data = json.loads(path.read_text())
    out = []
    for group, scenes in data.items():
        entries = []
        for scene, pairs in scenes.items():
            for key, info in pairs.items():
                names = ast.literal_eval(key)  # Never eval untrusted file contents.
                if not isinstance(names, (tuple, list)) or len(names) != 2:
                    raise ValueError('Invalid pair: ' + key)
                if any(not isinstance(n, str) or Path(n).name != n for n in names):
                    raise ValueError('Expected image basenames')
                entries.append(dict(group=group, scene=scene, image0=names[0], image1=names[1], info=info))
        # Deterministic coverage across scenes; identical selection for every method.
        entries.sort(key=lambda p: (p['scene'], p['image0'], p['image1']))
        if per_group and len(entries) > per_group:
            entries = [entries[i] for i in np.linspace(0, len(entries)-1, per_group, dtype=int)]
        out.extend(entries)
    return out

def match_images(im0, im1, nfeatures=4096, ratio=0.75, mutual=False):
    """Return keypoints, matches, and descriptor-stage counts. ratio=0 disables ratio test."""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    k0, d0 = sift.detectAndCompute(im0, None)
    k1, d1 = sift.detectAndCompute(im1, None)
    counts = dict(keypoints0=len(k0), keypoints1=len(k1), nn_candidates=0, ratio_matches=0)
    if d0 is None or d1 is None or (ratio and len(d1) < 2):
        return k0, k1, [], counts
    bf = cv2.BFMatcher(cv2.NORM_L2)
    if ratio:
        candidates = bf.knnMatch(d0, d1, k=2)
        good = [m[0] for m in candidates if len(m) == 2 and m[0].distance < ratio*m[1].distance]
        counts['nn_candidates'] = len(candidates)
    else:
        good = list(bf.match(d0, d1))
        counts['nn_candidates'] = len(good)
    counts['ratio_matches'] = len(good)
    if mutual:
        reverse = {m.queryIdx: m.trainIdx for m in bf.match(d1, d0)}
        good = [m for m in good if reverse.get(m.trainIdx) == m.queryIdx]
    return k0, k1, sorted(good, key=lambda m: m.distance), counts

def angle_errors(R, t, gt):
    er = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ gt[:3,:3])-1)/2, -1, 1))))
    tg = gt[:3,3]
    if np.linalg.norm(tg) < 1e-8:
        return er, None  # Translation direction undefined for zero baseline.
    cosine = abs(np.dot(t, tg)/(np.linalg.norm(t)*np.linalg.norm(tg)))
    return er, float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
