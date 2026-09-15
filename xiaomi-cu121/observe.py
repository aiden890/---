"""Record a real seeded RoboCasa reset for independent model preflight."""
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
import robocasa
from OpenGL import GL
from robocasa.utils.dataset_registry_utils import get_task_horizon

out=Path('/output')
out.mkdir(parents=True,exist_ok=True)
env=gym.make('robocasa/CloseBlenderLid',split='pretrain',seed=7)
try:
    obs,info=env.reset(seed=7)
    renderer=GL.glGetString(GL.GL_RENDERER)
    renderer=renderer.decode() if renderer else ''
    assert 'NVIDIA' in renderer, renderer
    keys=['state.end_effector_position_relative','state.end_effector_rotation_relative','state.gripper_qpos','state.base_position','state.base_rotation','video.robot0_agentview_left','video.robot0_agentview_right','video.robot0_eye_in_hand']
    for key in keys:
        assert key in obs,key
        assert np.isfinite(obs[key]).all(),key
    np.savez_compressed(out/'observation.npz',**{key:obs[key] for key in keys})
    meta={'task':'CloseBlenderLid','seed':7,'split':'pretrain','instruction':obs['annotation.human.task_description'],'horizon':get_task_horizon('CloseBlenderLid'),'renderer':renderer,'shapes':{k:list(obs[k].shape) for k in keys}}
    (out/'observation.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)
finally:
    env.close()
