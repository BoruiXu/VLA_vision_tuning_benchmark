"""Audited sharded WiSE-FT for X-VLA and OpenVLA-OFT visual parameters only."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def weight_map(path):
    index = path/'model.safetensors.index.json'
    if index.exists(): return json.loads(index.read_text())['weight_map']
    with safe_open(str(path/'model.safetensors'), framework='pt', device='cpu') as f:
        return {k:'model.safetensors' for k in f.keys()}


def build(model, base, tuned, output, alpha):
    assert 0 <= alpha <= 1
    assert not output.exists(), output
    bm, tm = weight_map(base), weight_map(tuned)
    assert bm.keys() == tm.keys(), 'Model tensor key mismatch'
    pattern = (r'^vlm\.(vision_tower|image_proj_norm|image_projection|image_pos_embed|visual_temporal_embed)(\.|$)'
               if model == 'xvla' else r'^(vision_backbone|projector)\.')
    visual = re.compile(pattern)
    # OFT external action/proprio heads must be the same frozen tensors.
    if model == 'oft':
        for prefix in ['action_head','proprio_projector']:
            a = list(base.glob(prefix+'--*_checkpoint.pt')); b = list(tuned.glob(prefix+'--*_checkpoint.pt'))
            assert len(a) == len(b) == 1
            normalize = lambda d: {k.removeprefix('module.'):v for k,v in d.items()}
            sa, sb = [normalize(torch.load(p,map_location='cpu',weights_only=True)) for p in [a[0],b[0]]]
            assert sa.keys() == sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)
        bs, ts = [json.loads((p/'dataset_statistics.json').read_text()) for p in [base,tuned]]
        for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
            assert bs[suite+'_no_noops'] == ts[suite+'_no_noops']
    output.mkdir(parents=True)
    counts = dict(visual=0, changed_visual=0, identical_nonvisual=0)
    with ExitStack() as stack:
        bh = {f:stack.enter_context(safe_open(str(base/f),framework='pt',device='cpu')) for f in set(bm.values())}
        th = {f:stack.enter_context(safe_open(str(tuned/f),framework='pt',device='cpu')) for f in set(tm.values())}
        for shard in sorted(set(tm.values())):
            state = {}
            for k in (k for k,f in tm.items() if f == shard):
                a, b = bh[bm[k]].get_tensor(k), th[shard].get_tensor(k)
                assert a.shape == b.shape and a.dtype == b.dtype, k
                if visual.search(k) and a.is_floating_point():
                    counts['visual'] += 1
                    counts['changed_visual'] += int(not torch.equal(a,b))
                    state[k] = (a.float() + alpha*(b.float()-a.float())).to(a.dtype)
                else:
                    assert torch.equal(a,b), 'Changed frozen tensor: '+k
                    counts['identical_nonvisual'] += 1
                    state[k] = a.contiguous().clone()
            save_file(state,str(output/shard),metadata={'format':'pt'})
            with safe_open(str(output/shard),framework='pt',device='cpu') as f:
                assert all(torch.equal(f.get_tensor(k),v) for k,v in state.items())
            del state
            print('[wiseft] verified',model,alpha,shard,flush=True)
    assert counts['changed_visual'] > 0
    for p in tuned.iterdir():
        if not p.is_file() or p.suffix == '.safetensors' or '.back' in p.name: continue
        if p.name.endswith('_manifest.json'): continue
        if p.suffix in ('.json','.py','.model','.txt','.pt'):
            shutil.copy2(p,output/p.name)
    manifest = dict(model=model,method='visual WiSE-FT',alpha=alpha,base=str(base),tuned=str(tuned),
                    training_data='camera_only',source_projector_update='lora',tensor_counts=counts,
                    output_weights_sha256={s:sha(output/s) for s in set(tm.values())})
    (output/'wiseft_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['xvla','oft'],required=True)
    for k in ['base','tuned','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--alpha',type=float,required=True);a=p.parse_args();build(a.model,a.base,a.tuned,a.output,a.alpha)
