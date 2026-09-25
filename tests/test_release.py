import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import numpy as np
from bootstrap_core import estimate

class ReleaseTests(unittest.TestCase):
    def test_bootstrap_identity(self):
        counts=np.full((40,7),2);base=np.ones((40,7));weights=np.ones((100,40))
        metrics=estimate(base,base,counts,weights,'camera')
        for value in metrics.values():
            self.assertAlmostEqual(value['delta_pp'],0)
            self.assertEqual(value['delta_ci95_pp'],[0,0])

    def test_bootstrap_constant_improvement(self):
        counts=np.full((40,7),2);base=np.zeros((40,7));tuned=np.ones((40,7));weights=np.ones((100,40))
        metrics=estimate(base,tuned,counts,weights,'camera')
        self.assertEqual(metrics['overall_micro']['delta_pp'],50)
        self.assertEqual(metrics['off_target_sum']['delta_pp'],300)
        self.assertEqual(metrics['worst_off_target']['delta_ci95_pp'],[50,50])

    def test_paper_reference_counts(self):
        runs={r['id']:r for r in json.loads((ROOT/'data/catalog.json').read_text())['runs']}
        expected={'oft-plus-stock':6853,'pi05-plus-stock':8428,'internvla15-plus-stock':8399,'nora15-plus-stock':4757,'xvla-plus-stock':6950,
                  'xvla-plus-camera_only':7652,'oft-plus-camera-delta':5190,'xvla-plus-camera-delta':8046,'nora15-original-camera-only':1626}
        for key,value in expected.items():self.assertEqual(runs[key]['successes'],value)

    def test_all_training_plans(self):
        spec=importlib.util.spec_from_file_location('train_plan',ROOT/'scripts/train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from argparse import Namespace
        for model in ['oft','xvla','pi05','nora15','internvla15']:
            for split in ['camera_only','noise_only','mixed']:
                for method in (['dual_lora','connector_lora','backbone_lora','delta'] if model in ['oft','xvla'] and split=='camera_only' else ['dual_lora']):
                    a=Namespace(model=model,split=split,method=method,seed=42,name=None,dataset=None,base=Path('/path/to/base'),gpu='0',steps=12000,resume='none',rlds=None,assets=None,stats=None)
                    command,env,cwd,run=module.plan(a)
                    self.assertTrue(Path(command[0]).exists())
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
                    self.assertEqual(env['VLA_PROJECTOR_UPDATE'],'full' if method=='backbone_lora' else 'lora')
                    scripts=[Path(s) for s in command if s.endswith('.py')];self.assertTrue(scripts)
                    for script in scripts:self.assertTrue(script.exists(),str(script))
                    if method=='delta':self.assertEqual(env[model.upper()+'_DELTA_LAMBDA'],'1')

    def test_cli_help_without_gpu_dependencies(self):
        for script in ['train.py','merge.py','serve.py','evaluate.py','setup_sources.py']:
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
