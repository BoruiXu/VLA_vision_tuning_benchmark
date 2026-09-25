#!/usr/bin/env python3
"""Clone pinned public repositories and apply the minimal experiment patches."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--component',action='append',choices=['xvla','oft','pi05','nora15','internvla15','harness'],required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    pins=json.loads((ROOT/'config/upstream.json').read_text())
    for name in a.component:
        spec=pins[name];dest=ROOT/spec['directory']
        print(name,spec['url'],spec['commit'],dest)
        if not a.execute:continue
        if dest.exists():raise FileExistsError(f'Use a fresh checkout: {dest}')
        dest.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout',spec['url'],str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',spec['commit']],check=True)
        patch=ROOT/'patches'/('openpi.patch' if name=='pi05' else name+'.patch')
        if patch.exists():
            subprocess.run(['git','-C',str(dest),'apply','--check',str(patch)],check=True)
            subprocess.run(['git','-C',str(dest),'apply',str(patch)],check=True)
        if name=='xvla':shutil.copyfile(ROOT/'patches/xvla_libero_plus.py',dest/'datasets/domain_handler/libero_plus.py')

if __name__=='__main__':main()
