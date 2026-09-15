import os, numpy as np, logging
logging.basicConfig(level=logging.INFO)
import openpi.training.config as _config
import openpi.policies.policy_config as pc
cfg=_config.get_config("pi05_pretrain_human300")
pol=pc.create_trained_policy(cfg, "/ckpt")
ex={"observation/image":np.random.randint(256,size=(224,224,3),dtype=np.uint8),
    "observation/wrist_image":np.random.randint(256,size=(224,224,3),dtype=np.uint8),
    "observation/right_image":np.random.randint(256,size=(224,224,3),dtype=np.uint8),
    "observation/state":np.random.rand(19).astype(np.float32),
    "prompt":"close the blender lid"}
print("INFER...")
out=pol.infer(ex)
print("OK", out["actions"].shape)
