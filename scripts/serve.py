#!/usr/bin/env python3
"""Serve one checkpoint through the pinned harness, in its model environment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['oft','xvla','pi05','nora15','internvla15'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--batch',type=int,default=8)
    p.add_argument('--address',default='127.0.0.1:8100');p.add_argument('--execute',action='store_true');a=p.parse_args()
    adapters={'oft':'oft_curve_compat','xvla':'xvla_batched','pi05':'pi0_stable','nora15':'nora15','internvla15':'internvla15'}
    args=dict(max_batch_size=a.batch,max_wait_time=.02,global_seed=7)
    checkpoint=str(a.checkpoint.resolve())
    if a.model=='xvla':args.update(model_path=checkpoint,benchmark_profile='libero',domain_id=3,denoising_steps=10,chunk_size=30)
    elif a.model=='oft':args.update(pretrained_checkpoint=checkpoint,unnorm_key='libero_spatial_no_noops',use_l1_regression=True,use_diffusion=False,use_film=False,num_images_in_input=2,use_proprio=True,center_crop=True,chunk_size=8)
    elif a.model=='pi05':args.update(config_name='pi05_libero_plus_base',checkpoint=checkpoint,image_key='observation/image',wrist_image_key='observation/wrist_image',state_key='observation/state',state_dim=8,image_resolution=224,chunk_size=5)
    elif a.model=='nora15':args.update(checkpoint=checkpoint,num_flow_steps=10,chunk_size=5)
    else:args.update(checkpoint=checkpoint,qwen=str(ROOT/'models/Qwen3.5-2B'),chunk_size=8)
    script=ROOT/'evaluation/model_servers'/f'{adapters[a.model]}.py'
    cmd=[sys.executable,str(script),f'--address={a.address}']+[f'--args.{k}={json.dumps(v) if isinstance(v,(bool,list,dict)) else v}' for k,v in args.items()]
    paths=[ROOT/'code',ROOT/'evaluation/model_servers',ROOT/'third_party/vla-evaluation-harness/src',ROOT/'third_party/openpi/src',ROOT/'third_party/openvla-oft',ROOT/'third_party/X-VLA']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,VLA_RETAIN_ROOT=str(ROOT),VLA_ROBUSTNESS_ROOT=str(ROOT),PYTHONPATH=os.pathsep.join(map(str,paths)),TOKENIZERS_PARALLELISM='false')
    print(shlex.join(cmd))
    if a.execute:subprocess.run(cmd,env=env,check=True)

if __name__=='__main__':main()
