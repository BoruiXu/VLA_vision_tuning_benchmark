#!/usr/bin/env python3
"""Render the paper's paired frame sequences from the bundled videos."""
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    for kind,steps in [('camera',[[0,20,40,60,100,300],[0,15,30,50,65,75]]),('noise',[[0,20,40,60,100,300],[0,15,30,45,60,73]])]:
        fig,axes=plt.subplots(2,6,figsize=(12,4))
        for i,label in enumerate(['stock','tuned']):
            reader=imageio.get_reader(ROOT/'qualitative'/f'{kind}_{label}.mp4')
            assert reader.count_frames()>max(steps[i])
            for j,t in enumerate(steps[i]):axes[i,j].imshow(reader.get_data(t));axes[i,j].set_title(f'{label}, step {t}',fontsize=8);axes[i,j].axis('off')
            reader.close()
        fig.tight_layout();fig.savefig(ROOT/'qualitative'/f'{kind}_frames.pdf',metadata={'Author':None,'Creator':'VLA-Retain','CreationDate':None,'ModDate':None});plt.close(fig)

if __name__=='__main__':main()
