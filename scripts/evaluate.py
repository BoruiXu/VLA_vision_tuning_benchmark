#!/usr/bin/env python3
"""Render on a separate host/GPU; connect to an explicit model-server URL."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',choices=['plus','screen560','original'],default='plus')
    p.add_argument('--server-url',default='ws://127.0.0.1:8100');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--egl-device',type=int,required=True,help='Physical EGL device index; need not equal CUDA index')
    p.add_argument('--shards',type=int,default=16);p.add_argument('--libero-config',type=Path,required=True)
    p.add_argument('--libero-repo',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    output=a.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Refusing to mix a new evaluation with an existing result directory')
    if a.shards<1:raise ValueError('shards must be positive')
    benchmark='original_libero:OriginalLIBEROBenchmark' if a.benchmark=='original' else 'benchmarks.selected_plus:SelectedLIBEROPlusBenchmark'
    config=dict(server=dict(url=a.server_url,timeout=600),output_dir=str(output),render='gpu',record_video=False,
                benchmarks=[dict(benchmark=benchmark,subname=s,episodes_per_task=50 if a.benchmark=='original' else 1,paced=False,
                                 params=dict(suite=s,seed=7,env_seed=7,num_steps_wait=10)) for s in SUITES])
    paths=[a.libero_repo.resolve(),ROOT/'evaluation',ROOT/'code',ROOT/'third_party/vla-evaluation-harness/src']
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,paths)),LIBERO_CONFIG_PATH=str(a.libero_config.resolve()),
             ORIGINAL_LIBERO_REPO=str(a.libero_repo.resolve()),ORIGINAL_LIBERO_MANIFEST=str(ROOT/'data/manifests/original_libero.json'),
             VLA_TASK_SELECTION=str(ROOT/'data/manifests'/('screen560_tasks.json' if a.benchmark=='screen560' else 'plus_tasks.json')),
             MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES=a.gpu,MUJOCO_EGL_DEVICE_ID=str(a.egl_device),EGL_DEVICE_ID=str(a.egl_device),
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    cfg=output/'evaluation.yaml';eval_id='retain-'+a.benchmark
    common=['-c',str(cfg),'--server-url',a.server_url,'--output-dir',str(output),'--eval-id',eval_id]
    commands=[[sys.executable,'-m','vla_eval.cli.main','run',*common,'--no-docker','--num-shards',str(a.shards),'--shard-id',str(i),'--render','gpu','--no-record-video','-y'] for i in range(a.shards)]
    for cmd in commands:print(shlex.join(cmd))
    if not a.execute:return
    import yaml
    output.mkdir(parents=True,exist_ok=True);cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    processes=[];logs=[]
    try:
        for i,cmd in enumerate(commands):
            log=(output/f'shard-{i}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(f'Shard failure: {codes}; partial results preserved')
        subprocess.run([sys.executable,'-m','vla_eval.cli.main','merge','-c',str(cfg),'--eval-id',eval_id,'--output-dir',str(output)],env=env,check=True)
    finally:
        for proc in processes:
            if proc.poll() is None:proc.terminate()
        for log in logs:log.close()

if __name__=='__main__':main()
