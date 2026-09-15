import json
import time
import torch
import transformers
import flash_attn
from transformers import AutoModel, AutoProcessor

print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda,'transformers':transformers.__version__,'flash_attn':flash_attn.__version__,'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0)}), flush=True)
assert torch.version.cuda == '12.1'
x=torch.randn(1,128,8,64,device='cuda',dtype=torch.bfloat16)
y=flash_attn.flash_attn_func(x,x,x)
torch.cuda.synchronize()
assert y.shape == x.shape and torch.isfinite(y).all()
print('FLASH_ATTN_CUDA_KERNEL_OK', flush=True)
start=time.monotonic()
processor=AutoProcessor.from_pretrained('/checkpoint',trust_remote_code=True,local_files_only=True,use_fast=False)
model=AutoModel.from_pretrained('/checkpoint',trust_remote_code=True,local_files_only=True,attn_implementation='flash_attention_2',dtype=torch.bfloat16).cuda().to(torch.bfloat16).eval()
print(json.dumps({'stage':'MODEL_LOADED','seconds':time.monotonic()-start,'vram_allocated':torch.cuda.memory_allocated(),'vram_peak':torch.cuda.max_memory_allocated(),'parameters':sum(p.numel() for p in model.parameters())}),flush=True)

# Real simulator observation; upstream message/state formatting, no fake policy.
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/work/upstream/eval_robocasa365')
from entry import EvalClient, observation_to_state, CAMERA_KEYS
obs=dict(np.load('/output/observation.npz',allow_pickle=False))
meta=json.loads(Path('/output/observation.json').read_text())
state=np.zeros((1,4,60),dtype=np.float32)
state[0,:,:14]=observation_to_state(obs)
images={key:np.repeat(obs[key][None],4,axis=0) for key in CAMERA_KEYS}
formatter=EvalClient.__new__(EvalClient)
formatter.crop_ratio=0.95
messages=formatter._build_messages(images,meta['instruction'])
inputs=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',do_resize=False,state=state,robot_type='robocasa365')
data={key:(value.to(device=model.device,dtype=model.dtype) if value.is_floating_point() else value.to(model.device)) if isinstance(value,torch.Tensor) else value for key,value in inputs.items()}
data['task_id']='robocasa365'
start=time.monotonic()
with torch.inference_mode():
    result=model(**data)
    actions=processor.decode_action(result.actions.cpu(),robot_type='robocasa365')[0,:,:12].float().numpy()
torch.cuda.synchronize()
assert actions.ndim==2 and actions.shape[1]==12 and len(actions)>=16 and np.isfinite(actions).all()
np.save('/output/preflight-actions.npy',actions)
report={'stage':'REAL_OBSERVATION_INFERENCE_OK','shape':list(actions.shape),'seconds':time.monotonic()-start,'vram_allocated':torch.cuda.memory_allocated(),'vram_peak':torch.cuda.max_memory_allocated(),'task':meta['task'],'seed':meta['seed'],'instruction':meta['instruction']}
Path('/output/inference.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
