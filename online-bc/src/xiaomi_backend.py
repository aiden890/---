import sys,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

class Backend:
    def __init__(self, checkpoint, lr=1e-4):
        sys.path[:0]=['/train/src','/rl_env/src','/work']
        import grpo_trainer_server as native
        native.SKILLS=('coffee',)
        args=dict(model=checkpoint,rank=8,alpha=32,adapter_skills='coffee',lora_targets='qkv_proj',train_mode='adapter_only',
            lr=lr,optimizer='adamw',weight_decay=0.,expert_lr=None,grad_clip=.5,grad_checkpoint=True,anchor_coef=0.,
            eta=0.,num_steps=5,sampler='fixed_noise',replan_steps=16,real_action_dim=12,clip=.1,kl_coef=0.,ratio_max=10.,adv_clip=3.,update_epochs=1,
            source_commit='online-bc-native-reuse',source_manifest_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
        self.native=native;self.server=native.GRPOTrainerServer(SimpleNamespace(**args))
        from transformers import AutoProcessor
        from skill_sft_dataset import SFTClient
        self.builder=object.__new__(SFTClient);self.builder.processor=AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,use_fast=False)

    def update(self, sample, seed=0):
        from skill_sft_dataset import CAMERA_KEYS,center_crop_np
        obs=sample['obs'];images={key:[center_crop_np(frame,.95) for frame in obs['xiaomi/'+key.replace('observation.images.','video.')]] for key in CAMERA_KEYS}
        inputs=self.builder.build_inputs(images,obs['xiaomi/state_history'],sample['prompt'])
        horizon=inputs['action_mask'].shape[-2] if inputs['action_mask'].ndim==3 else 50
        x=np.zeros((1,horizon,60),np.float32);mask=np.zeros_like(x);n=min(horizon,len(sample['actions']))
        x[0,:n,:12]=sample['actions'][:n];mask[0,:n,:12]=sample['valid'][:n,None]
        return self.server.op_sft_update(dict(inputs=inputs,x1=torch.from_numpy(x),loss_mask=torch.from_numpy(mask),skill='coffee',seed=seed,sft_steps=1))

    def save(self,path,step):
        return self.server.op_save(dict(path=str(Path(path)/'xiaomi.pt'),update_index=step))

    def load(self,path):
        from checkpoint_schema import validate_checkpoint_metadata
        # This checkpoint was produced by this learner, including native NumPy
        # RNG state; torch>=2.6's weights-only reader cannot decode that state.
        blob=torch.load(Path(path)/'xiaomi.pt',map_location='cuda',weights_only=False)
        validate_checkpoint_metadata(blob['metadata'],adapter_skills=self.server.adapter_skills,
            targets=self.server.lora_target_modules,rank=self.server.a.rank,alpha=self.server.a.alpha)
        self.native.load_lora_state_dict(self.server.wrappers,blob['lora'])
        self.server.opt.load_state_dict(blob['optimizer'])

    def parameters(self):
        return {n:p.detach().cpu().float().clone() for n,p in self.server.model.named_parameters() if p.requires_grad}

    def infer(self, sample, seed=0):
        from skill_sft_dataset import CAMERA_KEYS,center_crop_np
        o=sample['obs'];images={k:[center_crop_np(frame,.95) for frame in o['xiaomi/'+k.replace('observation.images.','video.')]] for k in CAMERA_KEYS}
        inputs=self.builder.build_inputs(images,o['xiaomi/state_history'],sample['prompt'])
        r=self.server.op_sample(dict(inputs=inputs,skill='coffee',eta=0.,seed=seed,chunk_index=0))
        return r['actions'][0,:,:12].float().numpy()
