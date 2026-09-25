"""Register only the two OpenPI configurations used in this release."""
import os
from openpi.training import config as c
from openpi.models import pi0_config
from openpi.training import optimizer, weight_loaders
import vla_paths

def install():
    for name in ['pi05_visonly','pi05_libero_plus_base']:
        cfg=c.TrainConfig(
            name=name,model=pi0_config.Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False),
            pytorch_trainable_regex=r'^(paligemma_with_expert\.paligemma\.model\.(vision_tower|multi_modal_projector))',
            data=c.LeRobotLiberoDataConfig(repo_id='local/libero_plus_mix',
                assets=c.AssetsConfig(assets_dir=os.environ.get('PI05_ASSETS',vla_paths.resolve('models/pi05-libero/assets')),asset_id='physical-intelligence/libero'),
                base_config=c.DataConfig(prompt_from_task=True),extra_delta_transform=False),
            batch_size=32,lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000,peak_lr=5e-5,decay_steps=30000,decay_lr=5e-5),
            optimizer=optimizer.AdamW(clip_gradient_norm=1.0),ema_decay=None,
            weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_libero/params'),
            pytorch_weight_path=os.environ.get('PI05_BASE',vla_paths.resolve('models/pi05-libero-pytorch')),
            num_train_steps=12000,num_workers=8,save_interval=2000,wandb_enabled=False,
            checkpoint_base_dir=vla_paths.resolve('runs'),assets_base_dir=vla_paths.resolve('assets'))
        c._CONFIGS_DICT[name]=cfg
