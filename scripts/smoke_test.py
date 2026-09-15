"""Real container integration test: create, reset, step, optional RGB."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Kitchen')
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--render', action='store_true')
parser.add_argument('--require-gpu', action='store_true')
parser.add_argument('--output', default='/output')
args = parser.parse_args()
if args.steps < 1:
    parser.error('--steps must be positive')

import numpy as np
import mujoco
import robosuite
import robocasa
from robosuite.controllers import load_composite_controller_config

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
env = robosuite.make(
    env_name=args.env, robots='PandaOmron',
    controller_configs=load_composite_controller_config(robot='PandaOmron'),
    layout_ids=[1], style_ids=[1], seed=0,
    has_renderer=False, has_offscreen_renderer=args.render,
    use_camera_obs=args.render, camera_names=['robot0_agentview_left'],
    camera_heights=128, camera_widths=128,
    control_freq=20, ignore_done=True,
)
try:
    obs = env.reset()
    assert isinstance(obs, dict) and obs, 'reset returned no observations'
    before = float(env.sim.data.time)
    for _ in range(args.steps):
        obs, reward, done, info = env.step(np.zeros(env.action_dim))
        assert np.isfinite(reward)
        assert np.isfinite(env.sim.data.qpos).all()
    assert env.sim.data.time > before, 'physics did not advance'
    result = dict(status='passed', environment=args.env, robot='PandaOmron',
                  layout=1, style=1, seed=0, steps=args.steps,
                  action_dim=env.action_dim, simulation_time=float(env.sim.data.time),
                  python=platform.python_version(), mujoco=mujoco.__version__,
                  robocasa=robocasa.__version__, robosuite=robosuite.__version__,
                  numpy=np.__version__, backend=os.environ.get('MUJOCO_GL'),
                  rendered=args.render, observation_keys=sorted(obs))
    if args.render:
        from PIL import Image
        from OpenGL import GL
        image = obs['robot0_agentview_left_image']
        assert image.shape == (128, 128, 3) and image.dtype == np.uint8
        assert image.std() > 1, 'blank render'
        Image.fromarray(image[::-1]).save(out / 'frame.png')
        result.update(image_shape=list(image.shape), image_std=float(image.std()),
                      gl_renderer=GL.glGetString(GL.GL_RENDERER).decode())
    if args.require_gpu:
        assert args.render and 'NVIDIA' in result['gl_renderer'], result.get('gl_renderer')
    result['wall_seconds'] = time.monotonic() - started
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
finally:
    env.close()
