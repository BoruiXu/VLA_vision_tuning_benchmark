#!/usr/bin/env python3
"""Vector-PDF views of the released paper results (no raster heatmaps)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DIMS=['camera','robot','language','light','background','noise','layout']
MODELS=['oft','pi05','internvla15','nora15','xvla']
LABELS=['OpenVLA-OFT',r'$\pi_{0.5}$','InternVLA-A1.5','NORA-1.5','X-VLA']
PDF_METADATA={'Author':None,'Creator':'VLA-Retain','CreationDate':None,'ModDate':None}

def main():
    d=json.loads((ROOT/'results/recomputed/metrics.json').read_text());out=ROOT/'results/figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ['changes','source_adjusted']:
        fig,axes=plt.subplots(1,2,figsize=(12,3.8),layout='constrained')
        for ax,split,title in zip(axes,['camera_only','noise_only'],['Camera-view tuning','Sensor-noise tuning']):
            values=np.array([[d['paired'][f'{m}-plus-{split}']['metrics'][dim]['delta_pp'] if kind=='changes' else d[kind][f'{m}-plus-{split}'][dim] for dim in DIMS] for m in MODELS])
            mesh=ax.pcolormesh(values,cmap='RdBu',vmin=-70,vmax=70,edgecolors='white')
            for i in range(5):
                for j in range(7):ax.text(j+.5,i+.5,f'{values[i,j]:+.1f}',ha='center',va='center',fontsize=8)
            ax.set(xticks=np.arange(7)+.5,xticklabels=DIMS,yticks=np.arange(5)+.5,yticklabels=LABELS,title=title);ax.invert_yaxis();ax.tick_params(axis='x',rotation=35)
        fig.colorbar(mesh,ax=axes,label='Change (pp)');fig.savefig(out/f'{kind}.pdf',metadata=PDF_METADATA);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    for m in ['oft','xvla']:
        steps=[0,2000,4000,8000,12000];target=[];off=[];overall=[]
        for step in steps:
            ident=f'{m}-screen-step{step}';target.append(d['absolute_rates'][ident]['camera']);overall.append(d['absolute_rates'][ident]['overall'])
            off.append(0 if step==0 else d['paired'][ident]['metrics']['off_target_macro']['delta_pp'])
        for ax,y in zip(axes,[target,off,overall]):ax.plot(steps,y,'o-',label=m)
    for ax,title in zip(axes,['Camera success (%)','Off-target change (pp)','Overall success (%)']):ax.set(title=title,xlabel='Optimizer updates');ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(out/'checkpoint_curves.pdf',metadata=PDF_METADATA);plt.close(fig)
    for name,methods in [('scope',['connector_lora','backbone_lora']),('wiseft',['wiseft_025','wiseft_050','wiseft_075']),('delta',['delta'])]:
        fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
        for ax,m in zip(axes,['oft','xvla']):
            ids=[f'{m}-plus-camera-{x}' for x in methods]+[f'{m}-plus-camera_only'];labels=methods+['dual_lora']
            for ident,label in zip(ids,labels):
                met=d['paired'][ident]['metrics'];ax.scatter(met['camera']['delta_pp'],met['off_target_macro']['delta_pp'],label=label)
            ax.axhline(0,color='gray',ls='--',lw=.7);ax.set(title=m,xlabel='Camera gain (pp)',ylabel='Off-target change (pp)');ax.legend(fontsize=7)
        fig.savefig(out/f'{name}.pdf',metadata=PDF_METADATA);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    x=np.arange(len(MODELS))
    for i,(split,label) in enumerate([('camera_only','Camera-view'),('noise_only','Sensor-noise'),('mixed','Mixed')]):
        values=[d['paired'][f'{m}-plus-{split}']['metrics']['overall_micro']['delta_pp'] for m in MODELS]
        axes[0].bar(x+(i-1)*.24,values,width=.24,label=label)
    axes[0].set(xticks=x,xticklabels=LABELS,ylabel='Overall change (pp)',title='Training-data coverage');axes[0].tick_params(axis='x',rotation=30);axes[0].legend(fontsize=7)
    for i,(split,label) in enumerate([('stock','Base'),('camera-only','Camera-view'),('noise-only','Sensor-noise')]):
        values=[d['absolute_rates'][f'{m}-original-{split}']['overall'] for m in MODELS]
        axes[1].bar(x+(i-1)*.24,values,width=.24,label=label)
    axes[1].set(xticks=x,xticklabels=LABELS,ylabel='Success (%)',title='Original LIBERO',ylim=(0,100));axes[1].tick_params(axis='x',rotation=30);axes[1].legend(fontsize=7)
    fig.savefig(out/'data_coverage_and_original_libero.pdf',metadata=PDF_METADATA);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,split in zip(axes,['camera_only','noise_only']):
        for i,m in enumerate(MODELS):
            met=d['paired'][f'{m}-plus-{split}']['metrics']['off_target_macro'];value=met['delta_pp'];lo,hi=met['delta_ci95_pp']
            ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',capsize=3)
        ax.set(yticks=range(5),yticklabels=LABELS,title=split,xlabel='Off-target change (pp), 95% CI');ax.axvline(0,color='gray',ls='--',lw=.8);ax.invert_yaxis()
    fig.savefig(out/'off_target_ci.pdf',metadata=PDF_METADATA);plt.close(fig)
    print(out)

if __name__=='__main__':main()
