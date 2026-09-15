"""Integration test: pinned AutoProcessor accepts real reset observation."""
import sys
import json
from pathlib import Path
import numpy as np
from transformers import AutoProcessor
sys.path.insert(0,'/work/upstream/eval_robocasa365')
from entry import EvalClient, observation_to_state, CAMERA_KEYS
processor=AutoProcessor.from_pretrained('/checkpoint',trust_remote_code=True,local_files_only=True,use_fast=False)
assert 'robocasa365' in processor.list_robot_types()
obs=dict(np.load('/output/observation.npz',allow_pickle=False))
meta=json.loads(Path('/output/observation.json').read_text())
state=np.zeros((1,4,60),dtype=np.float32)
state[0,:,:14]=observation_to_state(obs)
images={key:np.repeat(obs[key][None],4,axis=0) for key in CAMERA_KEYS}
formatter=EvalClient.__new__(EvalClient)
formatter.crop_ratio=0.95
inputs=processor.apply_chat_template(formatter._build_messages(images,meta['instruction']),tokenize=True,return_dict=True,return_tensors='pt',do_resize=False,state=state,robot_type='robocasa365')
print({k:list(v.shape) if hasattr(v,'shape') else type(v).__name__ for k,v in inputs.items()},flush=True)
print('PROCESSOR_REAL_OBSERVATION_OK',flush=True)
