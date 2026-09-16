"""RoboCasa environment adapter -- reset / step / observation / predicate only.

Wraps the parent ``skill_eval.Sim`` (predicate single-source-of-truth) and
``rollout`` (history sampling, image/state encoding, video frames) by IMPORT,
never forking them. Owns the obs-history queues and the recorded video frames;
contains no control decisions, no skill logic, no policy calls.

Heavy imports (numpy/gym/robocasa/rollout/skill_eval) are lazy so the rest of
the package imports on a CPU box. ``MockEnvironment`` (numpy-free) provides the
same interface for the offline control-loop check, driving a scripted predicate
timeline so the planner/verifier/executor exercise the full GRASP->MOVE->PLACE
handoff without a simulator.
"""
from __future__ import annotations

from schemas import PolicyInput


class RoboCasaEnvironment:
    """One CloseBlenderLid env instance + obs-history queues + video frames."""

    def __init__(self, seed, split, replan_steps, obs_history, obs_interval,
                 crop_ratio, video_stride, video_fps, env_id="robocasa/CloseBlenderLid"):
        self.seed = seed
        self.split = split
        self.replan_steps = replan_steps
        self.obs_history = obs_history
        self.obs_interval = obs_interval
        self.video_stride = video_stride
        self.video_fps = video_fps
        self._env_id = env_id

        import collections
        import gymnasium as gym
        import robocasa  # noqa: F401  (registers robocasa/* envs)
        import rollout
        import skill_eval

        self._collections = collections
        self._rollout = rollout
        self._skill_eval = skill_eval
        self._convert_action = __import__(
            "robocasa.utils.env_utils", fromlist=["convert_action"]).convert_action

        self._genv = gym.make(env_id, split=split, seed=seed)
        self._sim = skill_eval.Sim(self._genv)
        self._q_len = (obs_history - 1) * obs_interval + 1
        self._image_queues = {k: collections.deque(maxlen=self._q_len)
                              for k in rollout.CAMERA_KEYS}
        self._state_queue = collections.deque(maxlen=self._q_len)
        self.frames = []
        self._step_count = 0
        self._obs = None

    # ---- lifecycle -------------------------------------------------------- #
    def reset(self):
        rollout = self._rollout
        obs, _ = rollout.reset_env(self._genv, self.seed)
        self._sim.rest_lid_pos = self._sim.lid_pos()
        self._image_queues = {k: self._collections.deque(maxlen=self._q_len)
                              for k in rollout.CAMERA_KEYS}
        self._state_queue = self._collections.deque(maxlen=self._q_len)
        for key, image in rollout.collect_images(obs).items():
            self._image_queues[key].append(image)
        self._state_queue.append(rollout.observation_to_state(obs))
        self._obs = obs
        self.frames = [rollout.make_video_frame(obs)]
        self._step_count = 0
        return obs

    def scene_meta(self):
        return self._sim.scene_meta()

    def predicates(self):
        return self._sim.predicates()

    def observation_ref(self):
        """Compact, machine-readable reference to the current observation."""
        return f"seed{self.seed}:step{self._step_count}"

    def obs_for_verifier(self):
        """Build the obs-only verifier input from the CURRENT observation.

        Returns exactly what the real robot receives: the 3 camera images
        (rollout.collect_images -> CAMERA_KEYS) + the 14-D proprio state
        (rollout.observation_to_state). NO privileged simulator predicate is
        included, so the runtime VLM verifier is obs-only by construction.
        """
        from obs_verifier import ObsInput
        rollout = self._rollout
        images = {k: v for k, v in rollout.collect_images(self._obs).items()}
        proprio = list(rollout.observation_to_state(self._obs))
        return ObsInput(images=images, proprio=proprio, step=self._step_count)

    def processor(self):
        """The policy client's HF processor (single source of truth for VQA inputs)."""
        return None

    # ---- policy I/O ------------------------------------------------------- #
    def build_policy_input(self, instruction, adapter_mode, adapter_checkpoint=None):
        rollout = self._rollout
        states = rollout.sample_history(self._state_queue, self.obs_history, self.obs_interval)
        images = {k: rollout.sample_history(q, self.obs_history, self.obs_interval)
                  for k, q in self._image_queues.items()}
        return PolicyInput(instruction=instruction, state_history=states, image_history=images,
                           adapter_mode=adapter_mode, adapter_checkpoint=adapter_checkpoint)

    def step(self, action):
        """Execute one env action; update queues; maybe record a frame."""
        rollout = self._rollout
        obs, _, done, trunc, info = self._genv.step(self._convert_action(action))
        self._step_count += 1
        for key, image in rollout.collect_images(obs).items():
            self._image_queues[key].append(image)
        self._state_queue.append(rollout.observation_to_state(obs))
        self._obs = obs
        return obs, bool(done), bool(trunc), info

    def maybe_record_frame(self, force=False):
        if force or self._step_count % self.video_stride == 0:
            self.frames.append(self._rollout.make_video_frame(self._obs))
            return len(self.frames) - 1
        return None

    def save_video(self, path):
        import imageio.v2 as imageio
        imageio.mimsave(path, self.frames, fps=self.video_fps)
        return len(self.frames)

    def close(self):
        self._genv.close()


class MockEnvironment:
    """Numpy-free scripted env for the offline control-loop check.

    Drives a deterministic predicate timeline: after GRASP_SUCCEED_AT policy
    steps lid_grasped flips true, after +MOVE flips in_preplace_region, after
    +PLACE flips official_check_success. Lets the executor demonstrate the full
    planner->skill->verify->advance->planner chain and terminal success without
    a simulator.
    """

    def __init__(self, seed=0, replan_steps=4, video_stride=2, video_fps=20,
                 grasp_after=4, grasp_hold=20, move_after=4, move_hold=3, place_after=3):
        self.seed = seed
        self.replan_steps = replan_steps
        self.video_stride = video_stride
        self.video_fps = video_fps
        self.frames = []
        self._step_count = 0
        # milestones unlock strictly in sequence: grasped stays held long enough
        # to count as GRASP success before in_preplace can even begin, etc.
        self._grasp_after = grasp_after        # steps of stepping before lid_grasped latches
        self._grasp_hold = grasp_hold          # grasped must hold this long = GRASP success
        self._move_after = move_after          # steps after GRASP success before in_preplace
        self._move_hold = move_hold            # in_preplace holds this long = MOVE success
        self._place_after = place_after        # steps after MOVE success before task success
        self._grasped_since = None
        self._preplace_since = None
        self._pred = self._blank()

    def _blank(self):
        return {"lid_grasped": False, "lid_lifted": False, "in_preplace_region": False,
                "lid_on_blender": False, "lid_upright_7deg": True,
                "gripper_lid_far_0.15": False, "official_check_success": False,
                "lid_xy_to_closed_pos": 0.30, "lid_dz_to_closed_pos": 0.05, "eef_lid_dist": 0.20}

    def _compute(self):
        s = self._step_count
        grasped = s >= self._grasp_after
        if grasped and self._grasped_since is None:
            self._grasped_since = s
        grasp_success = grasped and (s - (self._grasped_since or s)) >= self._grasp_hold
        in_pp = grasp_success and (s - (self._grasped_since + self._grasp_hold)) >= self._move_after
        if in_pp and self._preplace_since is None:
            self._preplace_since = s
        move_success = in_pp and (s - (self._preplace_since or s)) >= self._move_hold
        success = move_success and (s - (self._preplace_since + self._move_hold)) >= self._place_after
        return {
            "lid_grasped": grasped, "lid_lifted": grasped, "in_preplace_region": in_pp,
            "lid_on_blender": success, "lid_upright_7deg": True,
            "gripper_lid_far_0.15": success, "official_check_success": success,
            "lid_xy_to_closed_pos": 0.02 if in_pp else 0.30,
            "lid_dz_to_closed_pos": 0.05, "eef_lid_dist": 0.01 if grasped else 0.20,
        }

    def reset(self):
        self._step_count = 0
        self._grasped_since = None
        self._preplace_since = None
        self.frames = [("frame", 0)]
        self._pred = self._blank()
        return {"mock_obs": True}

    def scene_meta(self):
        return {"mock": True, "seed": self.seed}

    def predicates(self):
        return dict(self._pred)

    def observation_ref(self):
        return f"mock-seed{self.seed}:step{self._step_count}"

    def build_policy_input(self, instruction, adapter_mode, adapter_checkpoint=None):
        return PolicyInput(instruction=instruction, state_history=[], image_history={},
                           adapter_mode=adapter_mode, adapter_checkpoint=adapter_checkpoint)

    def step(self, action):
        self._step_count += 1
        self._pred = self._compute()
        return {"mock_obs": True}, False, False, {"success": self._pred["official_check_success"]}

    def maybe_record_frame(self, force=False):
        if force or self._step_count % self.video_stride == 0:
            self.frames.append(("frame", self._step_count))
            return len(self.frames) - 1
        return None

    def save_video(self, path):
        from pathlib import Path
        Path(path).write_text(f"MOCK VIDEO {len(self.frames)} frames\n")
        return len(self.frames)

    def close(self):
        pass
