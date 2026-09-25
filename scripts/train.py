#!/usr/bin/env python3
"""Portable single-GPU launch plans for the paper's training configurations."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))

def plan(a):
    code=ROOT/'code';entry=code/'connector' if a.method=='connector_lora' else code
    scope={'dual_lora':'vision_lora','delta':'vision_lora','backbone_lora':'tower_lora','connector_lora':'projector_lora'}[a.method]
    if a.method!='dual_lora' and (a.model not in ['oft','xvla'] or a.split!='camera_only'):
        raise ValueError('The paper tests these methods on Camera-only OFT/X-VLA only')
    if a.seed!=42 and (a.model not in ['oft','xvla'] or a.split!='camera_only' or a.method!='dual_lora'):
        raise ValueError('The paper adds seeds 43/44 only for Camera-only OFT/X-VLA')
    name=a.name or f'{a.model}-{a.split}-{a.method}-seed{a.seed}'
    dataset=(ROOT/'data/lerobot/local/libero_plus_mix' if a.split=='mixed' else ROOT/'data/libero_plus_domain_splits'/a.split)
    if a.dataset:dataset=a.dataset.resolve()
    env=dict(os.environ,VLA_RETAIN_ROOT=str(ROOT),VLA_ROBUSTNESS_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpu,
             PYTHONPATH=os.pathsep.join([str(entry),str(code),os.environ.get('PYTHONPATH','')]),
             VLA_PROJECTOR_UPDATE='full' if a.method=='backbone_lora' else 'lora',WANDB_MODE='disabled',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    # Never reuse launcher-distributed rank variables as a LoRA rank.
    for key in ['RANK','LOCAL_RANK','WORLD_SIZE']:env.pop(key,None)
    env.update(HF_LEROBOT_HOME=str(ROOT/'data/lerobot'),HF_HOME=str(ROOT/'cache/huggingface'))
    cmd=[sys.executable];run=ROOT/'runs'/name
    if a.model=='xvla':
        repo=ROOT/'third_party/X-VLA'
        env.update(XVLA_REPO=str(repo),XVLA_SCOPE=scope,XVLA_TRAIN_PROJ='0' if a.method=='backbone_lora' else '1',
                   XVLA_LORA_RANK='32',XVLA_LORA_ALPHA='16',XVLA_LORA_DROPOUT='0',
                   XVLA_DELTA_LAMBDA='1' if a.method=='delta' else '0',XVLA_DELTA_ATTENTION='taylor',XVLA_DELTA_TEMPERATURE='1',
                   XVLA_WORKER_BASE_SEED=str(3228531506+a.seed-42),XVLA_PIN_MEMORY='0',XVLA_SHARING_STRATEGY='file_system')
        cmd+=['-m','accelerate.commands.launch','--num_processes','1','--mixed_precision','bf16',str(entry/'train_xvla_seeded.py'),
              '--models',str(a.base),'--train_metas_path',str(dataset/'meta/info.json'),'--output_dir',str(run),
              '--batch_size','32','--learning_rate','1e-4','--learning_coef','1','--weight_decay','0','--betas','.9','.95',
              '--max_grad_norm','1','--freeze_steps','0','--warmup_steps','2000','--iters',str(a.steps),'--use_cosine_decay',
              '--min_lr_ratio','.1','--save_interval','2000','--log_interval','50','--seed',str(a.seed)]
        if a.resume!='none':cmd+=['--resume_from_checkpoint',a.resume]
        cwd=repo
    elif a.model=='oft':
        repo=ROOT/'third_party/openvla-oft';env['PYTHONPATH']+=os.pathsep+str(repo)
        env.update(OFT_SCOPE=scope,OFT_TRAINABLE='encoder',OFT_HEADS_DIR=str(a.base),OFT_HEADS_STEP='300000',OFT_FREEZE_HEADS='1',
                   OFT_WEIGHT_DECAY='.1',OFT_WARMUP_STEPS='1000',OFT_MAX_STEPS=str(a.steps),OFT_MIN_LR_RATIO='0',OFT_SEED=str(a.seed),
                   OFT_NORM_STATS=str(a.base/'dataset_statistics.json'),OFT_TASK_SUITE_MAP=str(ROOT/'config/libero_task_suite_map.json'),
                   OFT_DOMAIN_FILTER={'mixed':'','camera_only':'camera_view','noise_only':'noise'}[a.split],
                   OFT_DOMAIN_AUDIT=str(ROOT/'data/libero_plus_domain_splits/audit.json'),OFT_DELTA_LAMBDA='1' if a.method=='delta' else '0',OFT_DELTA_TEMPERATURE='1')
        resume=None
        if a.resume=='auto':
            candidates=list((ROOT/'runs').glob(name+'--*_chkpt'))
            candidates=[p for p in candidates if (p/'training_state.pt').exists() and (p/'lora_adapter/adapter_config.json').exists()]
            if candidates:resume=max(candidates,key=lambda p:int(p.name.rsplit('--',1)[1].split('_')[0]))
        elif a.resume!='none':resume=Path(a.resume).resolve()
        cmd+=['-m','torch.distributed.run','--standalone','--nnodes','1','--nproc-per-node','1',str(entry/'train_oft.py'),
              '--vla_path',str(a.base),'--data_root_dir',str(a.rlds or ROOT/'data/rlds'),'--dataset_name','libero_mix',
              '--run_root_dir',str(ROOT/'runs'),'--run_id_override',name,'--use_l1_regression','True','--use_diffusion','False',
              '--use_film','False','--num_images_in_input','2','--use_proprio','True','--batch_size','16','--grad_accumulation_steps','2',
              '--learning_rate','5e-4','--lr_warmup_steps','0','--num_steps_before_decay','100000','--max_steps',str(a.steps),
              '--save_freq','2000','--save_latest_checkpoint_only','False','--image_aug','True','--use_lora','True','--lora_rank','32',
              '--lora_dropout','0','--merge_lora_during_training','False']
        if resume:
            step=int(resume.name.rsplit('--',1)[1].split('_')[0]);env.update(OFT_RESUME_DIR=str(resume),OFT_RESUME_STEP=str(step))
            cmd+=['--resume','True','--resume_step',str(step)]
        cwd=repo
    elif a.model=='pi05':
        cwd=ROOT/'third_party/openpi';env['PYTHONPATH']+=os.pathsep+str(cwd/'src')
        env.update(PI05_LORA_ENABLED='1',PI05_LORA_RANK='32',PI05_LORA_ALPHA='16',PI05_LORA_ADAPTER_SEED='420032',
                   PI05_BASE=str(a.base),PI05_ASSETS=str(a.assets or ROOT/'models/pi05-libero/assets'),
                   PI05_LORA_TARGET_REGEX=r'^paligemma_with_expert\.paligemma\.model\.(vision_tower|multi_modal_projector)(\.|$)',JAX_PLATFORMS='cpu')
        if a.split!='mixed':env['PI05_DOMAIN_META']=str(dataset/'meta')
        run=ROOT/'runs/pi05_visonly'/name
        cmd+=[str(code/'train_pi05_lora_pytorch.py'),'pi05_visonly','--exp-name',name,'--checkpoint-base-dir',str(ROOT/'runs'),
              '--batch-size','32','--num-train-steps',str(a.steps),'--save-interval','2000','--keep-period','2000','--num-workers','8','--no-wandb-enabled']
        if a.resume!='none' and run.exists() and any(p.name.isdigit() for p in run.iterdir()):cmd+=['--resume']
    else:
        cwd=ROOT
        cmd+=[str(code/f'train_{a.model}_encoder.py'),'--base',str(a.base),'--dataset',str(dataset),'--output',str(run),
              '--max-steps',str(a.steps),'--effective-batch-size','32','--micro-batch-size','32','--num-workers','8',
              '--save-every','2000','--lora-rank','32','--lora-alpha','16','--seed',str(a.seed),'--resume',a.resume]
        if a.model=='internvla15':cmd+=['--learning-rate','5e-5','--warmup-steps','2000','--decay-learning-rate','5e-6','--weight-decay','.01','--max-grad-norm','1']
        else:
            cmd+=['--learning-rate','8e-5','--warmup-steps','1000','--weight-decay','1e-8','--max-grad-norm','1']
            if a.stats:cmd+=['--base-stats',str(a.stats)]
    return cmd,env,cwd,run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['oft','xvla','pi05','nora15','internvla15'],required=True)
    p.add_argument('--split',choices=['camera_only','noise_only','mixed'],required=True)
    p.add_argument('--method',choices=['dual_lora','backbone_lora','connector_lora','delta'],default='dual_lora')
    p.add_argument('--base',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--name')
    p.add_argument('--dataset',type=Path);p.add_argument('--rlds',type=Path);p.add_argument('--assets',type=Path);p.add_argument('--stats',type=Path)
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42);p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--resume',default='auto');p.add_argument('--execute',action='store_true')
    a=p.parse_args();a.base=a.base.resolve()
    if ',' in a.gpu:raise ValueError('Select one training GPU')
    cmd,env,cwd,run=plan(a);print(shlex.join(cmd))
    print(json.dumps({k:v for k,v in env.items() if k.startswith(('OFT_','XVLA_','PI05_','VLA_PROJECTOR'))},indent=2))
    if not a.execute:return
    if not a.base.is_dir():raise FileNotFoundError(a.base)
    if a.resume=='none' and run.exists() and any(run.iterdir()):raise FileExistsError('Starting without resume requires a fresh run directory')
    run.mkdir(parents=True,exist_ok=True)
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ['execute','resume','steps']}
    path=run/'release_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Resume protocol differs; use a new run name')
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    if a.model=='xvla':subprocess.run([sys.executable,str(ROOT/'code/verify_xvla_data_alignment.py'),'--meta',cmd[cmd.index('--train_metas_path')+1]],env=env,check=True)
    subprocess.run(cmd,cwd=cwd,env=env,check=True)

if __name__=='__main__':main()
