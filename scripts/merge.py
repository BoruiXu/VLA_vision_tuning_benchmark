#!/usr/bin/env python3
"""Standalone merge; refuses an existing output directory."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['oft','xvla','pi05','nora15','internvla15'],required=True)
    for key in ['base','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--connector-only',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.connector_only and a.model not in ['oft','xvla']:raise ValueError('Connector-only is released for OFT and X-VLA only')
    if a.output.exists():raise FileExistsError('Choose a fresh merge directory')
    code=ROOT/'code';entry=code/'connector' if a.connector_only else code
    name='merge_oft_scope.py' if a.model=='oft' else f'merge_{a.model}_lora.py'
    cmd=[sys.executable,str(entry/name),'--base',str(a.base.resolve()),'--output',str(a.output.resolve())]
    cmd+=([str(a.checkpoint.resolve())] if a.model in ['oft','pi05'] else ['--checkpoint',str(a.checkpoint.resolve())])
    env=dict(os.environ,VLA_RETAIN_ROOT=str(ROOT),PYTHONPATH=os.pathsep.join([str(entry),str(code),str(ROOT/'third_party/openpi/src'),os.environ.get('PYTHONPATH','')]))
    print(shlex.join(cmd))
    if a.execute:subprocess.run(cmd,env=env,check=True)

if __name__=='__main__':main()
