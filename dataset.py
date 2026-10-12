"""Locate original nuScenes camera images named by the official RUBIK manifest."""
import json
from pathlib import Path


def image_index(root,pairs,metadata_root=None):
    names={p[k] for p in pairs for k in ('image0','image1')}
    records={}
    base=metadata_root or root
    metadata=base/'sample_data.json' if (base/'sample_data.json').is_file() else base/'v1.0-test/sample_data.json'
    if metadata.is_file():
        for row in json.loads(metadata.read_text()):
            name=Path(row['filename']).name
            if name in names:records[name]=row
    index={}
    for name in sorted(names):
        parts=name.split('__')
        row=records.get(name)
        candidates=[Path(row['filename'])] if row else []
        if len(parts)==3 and parts[1].startswith('CAM_'):
            candidates.extend(Path(folder)/parts[1]/name for folder in ('sweeps','samples'))
        if not candidates:raise ValueError('Not a nuScenes camera filename: '+name)
        selected=next((p for p in candidates if (root/p).is_file()),candidates[0])
        index[name]=dict(filename=str(selected),height=row['height'] if row else 900,width=row['width'] if row else 1600)
    return index
