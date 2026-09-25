import os
import unittest
import torch
from torch import nn
from peft import LoraConfig,get_peft_model
import encoder_mask as mask
import unified_lora_contract as contract


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone=nn.Linear(4,4)
        self.projector=nn.Sequential(nn.Linear(4,8),nn.GELU(),nn.Linear(8,4))
        self.language_model=nn.Linear(4,2)
    def forward(self,x):
        return self.language_model(self.projector(self.vision_backbone(x)))


class Tests(unittest.TestCase):
    def test_projector_only_lora_and_merge(self):
        os.environ['OFT_SCOPE']='projector_lora'
        base=Toy();original={n:p.detach().clone() for n,p in base.named_parameters()}
        names=mask.linear_modules(base,'projector_lora')
        self.assertEqual(names,['projector.0','projector.2'])
        m=get_peft_model(base,LoraConfig(r=2,lora_alpha=1,target_modules=names,bias='none'))
        groups=contract.audit(m,mask.is_vision,mask.is_projector)
        self.assertFalse(groups['tower'])
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
        m(torch.ones(2,4)).sum().backward();opt.step()
        merged=m.merge_and_unload()
        changed=[]
        for n,p in merged.named_parameters():
            if not torch.equal(original[n],p):changed.append(n)
        self.assertTrue(changed)
        self.assertTrue(all(n.startswith('projector.') and n.endswith('weight') for n in changed))


if __name__=='__main__':unittest.main()
