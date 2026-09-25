"""Compare merged weights with stock: only connector matrices may change."""

import vla_paths
import json
from pathlib import Path
import sys
import torch
from safetensors import safe_open
torch.set_num_threads(2)

model,merged=sys.argv[1:]
base=Path(vla_paths.resolve('models/xvla-libero') if model=='xvla' else vla_paths.resolve('models/openvla-oft'))

def index(root):
    result={}
    for p in root.glob('*.safetensors'):
        with safe_open(p,framework='pt',device='cpu') as f:
            for k in f.keys():result[k]=p
    return result

b,m=index(base),index(Path(merged))
assert set(b)==set(m), (set(b)-set(m),set(m)-set(b))
changed=[]
for key in b:
    with safe_open(b[key],framework='pt',device='cpu') as f:x=f.get_tensor(key)
    with safe_open(m[key],framework='pt',device='cpu') as f:y=f.get_tensor(key)
    assert torch.isfinite(y).all(),key
    if not torch.equal(x.to(y.dtype),y):
        allowed=key=='vlm.image_projection' if model=='xvla' else key.startswith('projector.') and key.endswith('.weight')
        assert allowed, 'Unexpected changed weight: '+key
        changed.append(key)
assert changed, 'No connector weight changed'
report=dict(model=model,changed=changed,verified_frozen_tensors=len(b)-len(changed),all_finite=True)
print(json.dumps(report),flush=True)
(Path(merged)/'connector_frozen_audit.json').write_text(json.dumps(report,indent=2)+'\n')
