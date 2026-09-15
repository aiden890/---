"""GRPO training loop (sim-client side, xiaomi-client image) for CloseBlenderLid.

Runs INSIDE the CPU-torch RoboCasa client and drives the GPU trainer server
(src/grpo_trainer_server.py) over the socket. Reuses the env card
(rl-env-t_4f3f2b20) as the single source of truth for the rollout harness, reward,
skill FSM, and randomization -- this card adds only the GRPO training *loop*:

  Phase EVAL(before)  : N deterministic (eta=0) full-plan rollouts on the eval seed
                        pool -> official-task success + grasp-stage success, videos, json.
  Phase TRAIN         : for each iteration, pick a TRAIN-pool seed, roll out `group`
                        members from the SAME initial state (identical reset seed;
                        flow-SDE eta>0 diversifies them), score each with the
                        simulator-predicate reward (env card RewardManager), send
                        group-relative advantages, call op=update. Log loss/reward/ratio.
  Phase EVAL(after)   : same as before-eval on the SAME eval pool -> before/after delta.
  Phase HELDOUT       : eval on the TEST pool (disjoint lid-handle geometry + scene
                        seeds; env card randomization.splits_disjoint()==True).

Reward-only ablation (task requirement): `--reward-variant` selects
  simulator_terminal_only  : RewardConfig(use_milestones=False)  (pure terminal predicate)
  simulator_milestones     : RewardConfig(use_milestones=True)   (terminal + simulator milestones)
Both are pure-simulator, a config switch (env card RewardConfig). The env card also
carries a VLM-auxiliary channel (weight 0, diagnostic only); a real VLM scorer is out
of scope for this pilot, so the genuine reward ablation we execute is terminal-only vs
terminal+milestone-shaping. This is stated honestly in the report.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work")               # rollout.py (harness)
sys.path.insert(0, "/skill_eval_tools")   # skill_eval.py (predicates/sim)
sys.path.insert(0, "/rl_env/src")         # env card: reward, skill_manager, randomization
import rollout  # noqa: E402
import skill_eval  # noqa: E402
from reward import RewardConfig, RewardManager, official_success, HoldConfig, hold_step_reward  # noqa: E402
from skill_manager import (  # noqa: E402
    MonitorConfig, OraclePlanner, Skill, SkillMonitor, SkillOutcome,
    SKILL_INSTRUCTION,
)

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
from robocasa.utils.env_utils import convert_action  # noqa: E402

SKILL_KEY = {Skill.GRASP: "grasp", Skill.MOVE_HOLDING: "move_holding", Skill.PLACE: "place"}


def _scoring_image(obs):
    """The RGB frame handed to the VLM scorer: the 3-camera panorama the POLICY itself
    sees (left + right agentview + wrist), concatenated. The wrist view is what makes the
    official predicate legible to a vision model -- it shows whether the gripper is still
    holding the lid or has retreated (the >0.15m clearance the terminal predicate needs),
    which a single agentview cannot disambiguate. Using the policy's own visual input keeps
    the scorer honest (no privileged camera) and maximises evidence.
    """
    return rollout.make_video_frame(obs)


# --------------------------------------------------------------------------- #
# Client speaking the trainer server's op-protocol.
# --------------------------------------------------------------------------- #
class TrainerClient:
    def __init__(self, model_path, host, port, robot_type, crop):
        import torch
        from transformers import AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.robot_type = robot_type
        self.crop = crop
        self.STATE_DIM = rollout.STATE_DIM
        self.ACTION_DIM = rollout.ACTION_DIM
        self.host, self.port = host, port
        self._connect()

    def _connect(self):
        for _ in range(600):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                return
            except OSError:
                time.sleep(1)
        raise ConnectionError(f"cannot reach trainer server {self.host}:{self.port}")

    def _rpc(self, request):
        blob = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(blob)) + blob)
        ln = self.sock.recv(4)
        n = struct.unpack(">I", ln)[0]
        data = b""
        while len(data) < n:
            data += self.sock.recv(n - len(data))
        resp = pickle.loads(data)
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(f"server error: {resp['error']}")
        return resp

    def _build_inputs(self, states, images, instruction):
        np_ = np
        state_history = np_.asarray(states, dtype=np_.float32)
        state = np_.zeros((1, state_history.shape[0], self.STATE_DIM), dtype=np_.float32)
        state[0, :, : state_history.shape[-1]] = state_history
        videos = {k: [rollout.center_crop(f, self.crop) for f in images[k]] for k in rollout.CAMERA_KEYS}
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Left camera: "}, {"type": "video", "video": videos[rollout.CAMERA_KEYS[0]]},
                {"type": "text", "text": "\nRight camera: "}, {"type": "video", "video": videos[rollout.CAMERA_KEYS[1]]},
                {"type": "text", "text": "\nWrist camera: "}, {"type": "video", "video": videos[rollout.CAMERA_KEYS[2]]},
                {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            do_resize=False, state=state, robot_type=self.robot_type)
        d = dict(inputs)
        d["task_id"] = self.robot_type
        return d

    def infer(self, states, images, instruction, *, eta, traj_id=None, seed=None, skill=None,
              chunk_index=0):
        inputs = self._build_inputs(states, images, instruction)
        req = {"op": "sample", "inputs": inputs, "eta": eta, "traj_id": traj_id, "seed": seed,
               "skill": skill, "chunk_index": chunk_index}
        resp = self._rpc(req)
        actions = resp["actions"]
        decoded = self.processor.decode_action(actions, robot_type=self.robot_type)
        decoded = decoded[0, :, : self.ACTION_DIM]
        decoded = decoded.float().cpu().numpy() if hasattr(decoded, "float") else np.asarray(decoded)
        return np.asarray(decoded, dtype=np.float32), resp.get("logprob")

    def update(self, advantages, clip=0.1, kl_coef=0.005, ratio_max=10.0, adv_clip=3.0,
               update_epochs=1):
        return self._rpc({"op": "update", "advantages": advantages, "clip": clip,
                          "kl_coef": kl_coef, "ratio_max": ratio_max, "adv_clip": adv_clip,
                          "update_epochs": update_epochs})

    def save(self, path):
        return self._rpc({"op": "save", "path": path})

    def load(self, path):
        return self._rpc({"op": "load", "path": path})

    def vlm_score(self, image, question_key):
        """Real VQA score P(yes) for an image + yes/no question, via the trainer server.

        Builds the VLM inputs with THIS client's processor (single source of truth for
        the prompt lives in the env card vlm_scorer), sends op=vlm_score, returns P(yes).
        """
        import vlm_scorer  # /rl_env/src on sys.path
        inputs = vlm_scorer.build_vqa_inputs(self.processor, image, question_key,
                                             robot_type=self.robot_type,
                                             state_dim=self.STATE_DIM)
        resp = self._rpc({"op": "vlm_score", "inputs": inputs, "question": question_key})
        return float(resp["prob"])

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Rollout primitives.
# --------------------------------------------------------------------------- #
def _make_env(split, seed):
    genv = gym.make("robocasa/CloseBlenderLid", split=split, seed=seed)
    return genv, skill_eval.Sim(genv)


def _run_one_skill(sim, client, obs, args, skill, reward_mgr, *, eta, traj_id, seed,
                   frames=None, save_video=None, approach_coef=0.0, timeout_penalty=0.0,
                   hold_cfg=None, hold_steps=0):
    """Run ONE skill's VLA loop; return (obs, outcome, reward, steps, predicates).

    Training-only dense shaping (not used in eval; all default to off):
      approach_coef: reward += approach_coef * (prev_dist - curr_dist) per step toward lid.
        Creates return variance even when all members fail to grasp, so GRPO has signal.
      timeout_penalty: flat penalty added when outcome==TIMEOUT (skill horizon exceeded).
        Ensures failed trajectories are numerically distinguished from successful ones.
      hold_cfg/hold_steps (operator decision B, run30): the BOUNDARY-COMPLIANCE term.
        When the skill first reaches SUCCESS, instead of breaking immediately we keep
        stepping for `hold_steps` more steps with the SAME instruction and add
        hold_step_reward per step (+ for staying put, - for drifting / breaking the
        success state). This directly teaches "stop after you succeed" -- the failure
        SFT could not fix. hold_steps=0 disables it (prior behaviour, break on success).
    Only affects the training return; eval passes hold_steps=0 so official reward is unchanged.
    Also returns post-success drift stats via the `_hold_stats` attribute on the returned
    predicate dict is avoided; instead we stash them on the function's return tuple caller
    reads from reward_mgr? -> simplest: attach to a mutable, see stats dict below.
    """
    instruction = SKILL_INSTRUCTION[skill]
    skill_key = SKILL_KEY[skill]
    monitor = SkillMonitor(skill, MonitorConfig(grasp_hold_steps=skill_eval.GRASP_HOLD_STEPS,
                                                 move_hold_steps=skill_eval.MOVE_HOLD_STEPS))
    horizon = {Skill.GRASP: args.horizon_grasp, Skill.MOVE_HOLDING: args.horizon_move,
               Skill.PLACE: args.horizon_place}[skill]
    ql = (args.obs_history - 1) * args.obs_interval + 1
    image_queues = {k: collections.deque(maxlen=ql) for k in rollout.CAMERA_KEYS}
    state_queue = collections.deque(maxlen=ql)
    for k, im in rollout.collect_images(obs).items():
        image_queues[k].append(im)
    state_queue.append(rollout.observation_to_state(obs))
    action_plan = collections.deque()
    if frames is not None and not frames:
        frames.append(rollout.make_video_frame(obs))
    steps = 0
    outcome = None
    reward_sum = 0.0
    chunk_idx = 0   # AUDIT FIX #2: monotonic per-skill chunk counter -> unique RNG per replan
    p = sim.predicates()
    prev_eef_lid_dist = float(p.get("eef_lid_dist", 0.0)) if approach_coef > 0.0 else None
    # --- post-success hold bookkeeping (operator decision B) ---
    success_step = None                 # step at which the skill first succeeded
    held_steps = 0                      # steps stepped in the post-success hold window
    hold_stay = 0                       # of those, how many were "stopped" (<= stay_radius)
    hold_reward = 0.0                   # total hold shaping added
    prev_eef = list(p.get("eef_pos", (0.0, 0.0, 0.0)))
    def _success_now(pred):
        # the skill's own success predicate (same source as SkillMonitor): GRASP=lid_grasped,
        # MOVE=grasped & in preplace region, PLACE/terminal = official_check_success.
        if skill is Skill.GRASP:
            return bool(pred.get("lid_grasped"))
        if skill is Skill.MOVE_HOLDING:
            return bool(pred.get("lid_grasped")) and bool(pred.get("in_preplace_region"))
        return bool(pred.get("official_check_success")) or official_success(pred)
    while steps < horizon:
        if not action_plan:
            states = rollout.sample_history(state_queue, args.obs_history, args.obs_interval)
            images = {k: rollout.sample_history(q, args.obs_history, args.obs_interval)
                      for k, q in image_queues.items()}
            chunk, _ = client.infer(states, images, instruction, eta=eta, traj_id=traj_id,
                                    seed=seed, skill=skill_key, chunk_index=chunk_idx)
            chunk_idx += 1
            if len(chunk) < args.replan_steps:
                raise RuntimeError(f"chunk {len(chunk)} < replan {args.replan_steps}")
            action_plan.extend(chunk[: args.replan_steps])
        a = np.asarray(action_plan.popleft(), dtype=np.float32)
        obs, _, done, trunc, info = sim.genv.step(convert_action(a))
        steps += 1
        for k, im in rollout.collect_images(obs).items():
            image_queues[k].append(im)
        state_queue.append(rollout.observation_to_state(obs))
        p = sim.predicates()
        rb = reward_mgr.step_reward(steps, p, done=bool(done), truncated=bool(trunc))
        reward_sum += rb.primary
        # --- post-success boundary hold shaping (training-only) ---
        if success_step is not None and hold_cfg is not None and hold_steps > 0:
            curr_eef = list(p.get("eef_pos", prev_eef))
            hr = hold_step_reward(prev_eef, curr_eef, _success_now(p), hold_cfg)
            reward_sum += hr
            hold_reward += hr
            held_steps += 1
            import math as _m
            disp = _m.dist(prev_eef, curr_eef) if hasattr(_m, "dist") else \
                sum((a - b) ** 2 for a, b in zip(prev_eef, curr_eef)) ** 0.5
            if disp <= hold_cfg.stay_radius_m:
                hold_stay += 1
        prev_eef = list(p.get("eef_pos", prev_eef))
        # Approach shaping: reward getting closer to the lid (training-only)
        if approach_coef > 0.0 and prev_eef_lid_dist is not None:
            curr_dist = float(p.get("eef_lid_dist", prev_eef_lid_dist))
            reward_sum += approach_coef * (prev_eef_lid_dist - curr_dist)
            prev_eef_lid_dist = curr_dist
        outcome = monitor.update(p, steps, horizon)
        if frames is not None and (steps % args.video_stride == 0 or outcome or done or trunc):
            frames.append(rollout.make_video_frame(obs))
        # Boundary-hold: on the FIRST SUCCESS, latch the step and keep going for a hold
        # window instead of breaking (so the post-success stay/drift shaping can score the
        # policy's boundary behaviour). Without a hold window, break on any terminal outcome.
        if outcome is SkillOutcome.SUCCESS and hold_cfg is not None and hold_steps > 0:
            if success_step is None:
                success_step = steps
            if steps - success_step >= hold_steps:
                break
            # keep the success outcome latched; do not break yet
            continue
        if outcome or done or trunc:
            break
    if outcome is None:
        outcome = SkillOutcome.TIMEOUT
        reward_sum += (-timeout_penalty)   # flat cost for skill timeout (training-only)
    if save_video is not None and frames is not None:
        import imageio.v2 as imageio
        imageio.mimsave(save_video, frames, fps=args.video_fps)
    hold_stats = {"success_step": success_step, "held_steps": held_steps,
                  "hold_stay": hold_stay, "hold_reward": round(hold_reward, 4),
                  "hold_stay_frac": (round(hold_stay / held_steps, 3) if held_steps else None)}
    return obs, outcome, reward_sum, steps, p, hold_stats


# --------------------------------------------------------------------------- #
# Entry-state setup (operator MOVE-first decision).
# GRASP is too rarely successful in eta>0 training rollouts (~0.17), so the
# boundary-hold reward never triggers (n_success_hold ~ 0) and GRPO has no signal.
# MOVE_HOLDING is the EASIER skill once the lid is already grasped+lifted, so we
# train/eval it from a grasp-SUCCESS entry-state (operator option 1): reset(seed),
# run GRASP deterministically (eta=0), and only if it reaches SUCCESS use that
# post-grasp obs/sim state as the MOVE start. All group members share the SAME
# deterministic grasp entry (identical seed, eta=0), so the group-relative MOVE
# advantage stays a pure within-entry-state comparison. Backward-compatible:
# --entry-from-grasp defaults OFF (skills start from reset as before).
# --------------------------------------------------------------------------- #
# prerequisite skill chain to reach a skill's entry-state from a fresh reset
ENTRY_PREREQS = {Skill.MOVE_HOLDING: (Skill.GRASP,)}


def _reach_entry_state(sim, client, args, seed, target_skill, init_obs):
    """Advance a freshly-reset env to `target_skill`'s entry-state by running its
    prerequisite skills DETERMINISTICALLY (eta=0). Returns (obs, ok, p) where ok is
    True iff every prerequisite reached SUCCESS. The action-noise seed is derived
    from the env seed (paired/reproducible), matching eval_episode's convention.
    Caller must have already reset the env (passing that reset obs as init_obs) and
    set sim.rest_lid_pos."""
    prereqs = ENTRY_PREREQS.get(target_skill, ())
    obs = init_obs
    p = sim.predicates()
    ok = True
    for i, pre in enumerate(prereqs):
        entry_action_seed = int(seed) * 131 + (i + 1)   # same scheme as eval_episode
        obs, outcome, _r, _steps, p, _h = _run_one_skill(
            sim, client, obs, args, pre,
            RewardManager(_ENTRY_REWARD_CFG), eta=0.0, traj_id=None, seed=entry_action_seed)
        if outcome is not SkillOutcome.SUCCESS:
            ok = False
            break
    return obs, ok, p


def _current_obs(sim):
    """Get a valid observation after a snapshot restore WITHOUT meaningful motion.
    RoboCasa exposes no pure obs getter, so mirror skill_eval's settle convention:
    one hold-action step (gripper CLOSED so the grasped lid is retained, zero arm
    motion) both settles physics after set_state and returns the obs dict the queues
    need. This is the same 'restore + 1 hold step' skill_eval uses for its per-skill
    entry-state datasets, so the MOVE start matches that verified convention."""
    obs, _, _, _, _ = sim.genv.step(convert_action(skill_eval.hold_action(True)))
    return obs


# throwaway reward cfg for the deterministic entry-setup grasp (its reward is unused)
_ENTRY_REWARD_CFG = RewardConfig(mode="simulator", horizon=200, use_milestones=True)


def build_entry_seed_pool(factory, client, args, candidate_seeds, target_skill, need, tag=""):
    """Scan candidate env seeds, keeping only those whose deterministic (eta=0)
    prerequisite chain reaches the target skill's entry-state (e.g. grasp SUCCESS),
    until `need` are found or candidates run out. Returns the list of good seeds.
    This makes MOVE train/eval start from a real grasp-success state instead of the
    OOD lid-on-counter reset."""
    good = []
    for seed in candidate_seeds:
        if len(good) >= need:
            break
        genv, sim = factory(seed)
        try:
            obs, _ = rollout.reset_env(genv, seed)
            sim.rest_lid_pos = sim.lid_pos()
            _obs, ok, _p = _reach_entry_state(sim, client, args, seed, target_skill, obs)
        finally:
            genv.close()
        print(f"[entry-scan{tag}] seed={seed} reached_entry={ok} ({len(good)}/{need})", flush=True)
        if ok:
            good.append(seed)
    return good


def eval_skill_from_entry(sim, client, args, reward_cfg, seed, skill,
                          entry_from_grasp, out_dir=None, tag=""):
    """Single-skill deterministic (eta=0) eval from the skill's entry-state.

    When entry_from_grasp and skill is MOVE_HOLDING: reset(seed) -> run GRASP eta=0
    to reach the grasp entry-state -> run MOVE eta=0 and judge MOVE's own success
    predicate (grasped AND in_preplace_region held MOVE_HOLD_STEPS = the SkillMonitor
    FSM outcome). Returns {seed, entry_ok, skill_success, ...}. If the grasp entry
    is not reached, skill_success is False and entry_ok is False (seed excluded from
    rate by the caller). This isolates MOVE from the full oracle plan so the number
    is a clean MOVE-from-grasp success rate, comparable base vs trained."""
    obs, _ = rollout.reset_env(sim.genv, seed)
    sim.rest_lid_pos = sim.lid_pos()
    reward_mgr = RewardManager(reward_cfg)
    save_dir = None
    if out_dir is not None:
        save_dir = Path(out_dir); save_dir.mkdir(parents=True, exist_ok=True)
    entry_ok = True
    if entry_from_grasp:
        obs, entry_ok, _p = _reach_entry_state(sim, client, args, seed, skill, obs)
        if not entry_ok:
            return {"seed": seed, "entry_ok": False, "skill_success": False,
                    "outcome": "ENTRY_FAILED", "total_reward": 0.0}
    frames = [] if save_dir is not None else None
    vid = (save_dir / f"{tag}seed{seed}_{SKILL_KEY[skill]}.mp4") if frames is not None else None
    eval_action_seed = int(seed) * 131 + 90   # distinct from entry-setup seeds
    obs, outcome, r, steps, p, _hold = _run_one_skill(
        sim, client, obs, args, skill, reward_mgr, eta=0.0, traj_id=None,
        seed=eval_action_seed, frames=frames, save_video=str(vid) if vid else None)
    return {"seed": seed, "entry_ok": entry_ok,
            "skill_success": bool(outcome is SkillOutcome.SUCCESS),
            "outcome": outcome.name if outcome else "NONE",
            "steps": steps, "total_reward": round(r, 4)}


def eval_skill_pool(factory, client, args, reward_cfg, seeds, skill, entry_from_grasp,
                    out_dir, tag):
    """Single-skill eval over a seed pool (MOVE-from-grasp-entry). Rate is over
    seeds whose entry was reached (entry_ok), so it measures the skill in isolation."""
    results = []
    for i, seed in enumerate(seeds):
        genv, sim = factory(seed)
        try:
            save = out_dir if i < args.save_videos else None
            r = eval_skill_from_entry(sim, client, args, reward_cfg, seed, skill,
                                      entry_from_grasp, out_dir=save, tag=tag)
        finally:
            genv.close()
        results.append(r)
        print(f"[{tag}] {i+1}/{len(seeds)} seed={seed} entry_ok={r['entry_ok']} "
              f"skill_success={r['skill_success']} outcome={r['outcome']}", flush=True)
    entered = [r for r in results if r["entry_ok"]]
    n_entered = len(entered)
    succ = sum(r["skill_success"] for r in entered)
    return {"n": len(results), "n_entered": n_entered,
            "skill_success_rate": (succ / n_entered if n_entered else 0.0),
            "skill_success_rate_over_all": (succ / len(results) if results else 0.0),
            "episodes": results}


def eval_episode(sim, client, args, reward_cfg, seed, out_dir=None, tag="", return_final_obs=False):
    """Deterministic full-plan rollout; returns official + grasp-stage success.

    If return_final_obs, also returns the last observation (for the VLM-scorer gate).
    """
    obs, _ = rollout.reset_env(sim.genv, seed)
    sim.rest_lid_pos = sim.lid_pos()
    planner = OraclePlanner(max_skill_calls=args.max_skill_calls)
    reward_mgr = RewardManager(reward_cfg)
    call = planner.initial()
    guard = 0
    grasp_ok = False
    total_reward = 0.0
    save_dir = None
    last_obs = obs
    if out_dir is not None:
        save_dir = Path(out_dir); save_dir.mkdir(parents=True, exist_ok=True)
    while call is not None and guard < args.max_skill_calls:
        guard += 1
        frames = [] if (save_dir is not None and guard <= 3) else None
        vid = (save_dir / f"{tag}seed{seed}_call{guard}_{SKILL_KEY[call.skill]}.mp4") if frames is not None else None
        # AUDIT FIX #1: give eta=0 eval a DETERMINISTIC action-noise seed derived from the
        # env seed + skill-call index, so before/after eval on the same env seed shares the
        # same x0 latent and the comparison is paired (was seed=None -> unseeded x0).
        eval_action_seed = int(seed) * 131 + guard
        obs, outcome, r, steps, p, _hold = _run_one_skill(
            sim, client, obs, args, call.skill, reward_mgr, eta=0.0, traj_id=None,
            seed=eval_action_seed, frames=frames, save_video=str(vid) if vid else None)
        last_obs = obs
        total_reward += r
        if call.skill is Skill.GRASP and outcome is SkillOutcome.SUCCESS:
            grasp_ok = True
        call = planner.propose(call, outcome, p)
    success = bool(sim.predicates().get("official_check_success"))
    res = {"seed": seed, "official_success": success, "grasp_success": grasp_ok,
           "total_reward": round(total_reward, 4)}
    if return_final_obs:
        return res, last_obs
    return res


def train_iteration(client, args, reward_cfg, seed, it):
    """Group rollout from ONE shared initial state; returns metrics.

    Trains a single configured skill (default GRASP -- the measured bottleneck, and the
    curriculum entry point from the research brief: PLACE is data-starved so is unlocked
    later). All group members share the identical reset seed; flow-SDE eta>0 diversifies
    them, so the group-relative advantage is a pure within-start comparison.

    Dense shaping active during training (not during eval):
      approach_coef=0.1: reward for getting closer to the lid per step (GRASP skill only).
      timeout_penalty=0.5: flat cost when the skill times out without success.
    These create return variance in the degenerate 0%-success baseline case so GRPO has
    a gradient signal. Both are training-only; eval uses the official reward unchanged.
    """
    skill = {"grasp": Skill.GRASP, "move_holding": Skill.MOVE_HOLDING, "place": Skill.PLACE}[args.train_skill]
    # approach shaping only makes sense for GRASP (lid proximity); other skills use milestone variety
    train_approach_coef = 0.1 if skill is Skill.GRASP else 0.0
    train_timeout_penalty = 0.5  # flat cost for timing out a skill
    # Boundary-compliance hold (operator decision B): reward stopping after success. Config
    # from args (0 hold_steps disables it -> identical to the pre-B behaviour).
    hold_steps = int(getattr(args, "hold_steps", 0))
    hold_cfg = HoldConfig(stay_bonus=getattr(args, "hold_stay_bonus", 0.05),
                          drift_penalty=getattr(args, "hold_drift_penalty", 0.10),
                          stay_radius_m=getattr(args, "hold_stay_radius", 0.02),
                          drop_success_penalty=getattr(args, "hold_drop_penalty", 0.5)) \
        if hold_steps > 0 else None
    vlm_weight = float(getattr(args, "vlm_weight", 0.0))
    vlm_question = getattr(args, "vlm_question", "grasp")
    returns = []
    traj_ids = []
    vlm_scores = []
    hold_stats_all = []
    entry_from_grasp = getattr(args, "entry_from_grasp", False) and skill in ENTRY_PREREQS

    if entry_from_grasp:
        # Reach the grasp entry-state ONCE (deterministic eta=0), snapshot it, then
        # restore per group member so every member starts byte-identical from the SAME
        # real grasp-success state (skill guidance: group members start byte-identical,
        # diverge under eta>0). Avoids re-running the ~120-step grasp `group` times.
        genv, sim = _make_env(args.split, seed)
        try:
            reset_obs, _ = rollout.reset_env(genv, seed)
            sim.rest_lid_pos = sim.lid_pos()
            _obs, entry_ok, _p = _reach_entry_state(sim, client, args, seed, skill, reset_obs)
            if not entry_ok:
                genv.close()
                # No grasp entry this seed -> no MOVE signal. Return a sentinel the
                # caller logs and skips (no optimizer update on an empty group).
                return {"iter": it, "seed": seed, "skipped": "entry_failed",
                        "n_success_hold": 0, "mean_hold_stay_frac": None,
                        "loss": 0.0, "grad_norm": 0.0, "mean_ratio": 1.0,
                        "mean_return": 0.0, "max_return": 0.0, "reward_std": 0.0,
                        "returns": []}
            entry_snap = sim.snapshot("move_entry", 0)
            for m in range(args.group):
                sim.restore(entry_snap)
                obs = _current_obs(sim)
                reward_mgr = RewardManager(reward_cfg)
                traj_id = f"iter{it}_m{m}"
                final_obs, outcome, r, steps, p, hstats = _run_one_skill(
                    sim, client, obs, args, skill, reward_mgr, eta=args.eta,
                    traj_id=traj_id, seed=args.seed_base + it * 100 + m,
                    approach_coef=train_approach_coef,
                    timeout_penalty=train_timeout_penalty,
                    hold_cfg=hold_cfg, hold_steps=hold_steps)
                hold_stats_all.append(hstats)
                if outcome is SkillOutcome.SUCCESS:
                    r += 1.0
                if vlm_weight > 0.0:
                    vscore = client.vlm_score(_scoring_image(final_obs), vlm_question)
                    vlm_scores.append(vscore)
                    r += vlm_weight * vscore
                returns.append(r)
                traj_ids.append(traj_id)
        finally:
            genv.close()
    else:
        for m in range(args.group):
            genv, sim = _make_env(args.split, seed)
            try:
                obs, _ = rollout.reset_env(genv, seed)   # identical start for all members
                sim.rest_lid_pos = sim.lid_pos()
                reward_mgr = RewardManager(reward_cfg)
                traj_id = f"iter{it}_m{m}"
                final_obs, outcome, r, steps, p, hstats = _run_one_skill(
                    sim, client, obs, args, skill, reward_mgr, eta=args.eta,
                    traj_id=traj_id, seed=args.seed_base + it * 100 + m,
                    approach_coef=train_approach_coef,
                    timeout_penalty=train_timeout_penalty,
                    hold_cfg=hold_cfg, hold_steps=hold_steps)
                hold_stats_all.append(hstats)
                # terminal shaping for the skill unit: bonus if the skill's predicate is met
                if outcome is SkillOutcome.SUCCESS:
                    r += 1.0
                # --- REAL VLM-auxiliary channel (sim+VLM ablation only; weight>0) ---
                # A genuine Qwen3-VL VQA judgement of the FINAL frame: P(yes) that the grasp/
                # closing progressed. Added to the GRPO return, never replacing the simulator
                # reward. weight==0 => pure sim-only (identical to the executed baseline).
                if vlm_weight > 0.0:
                    vscore = client.vlm_score(_scoring_image(final_obs), vlm_question)
                    vlm_scores.append(vscore)
                    r += vlm_weight * vscore
                returns.append(r)
                traj_ids.append(traj_id)
            finally:
                genv.close()
    arr = np.asarray(returns, dtype=np.float64)
    adv = (arr - arr.mean()) / (arr.std() + 1e-8)
    advantages = {tid: float(a) for tid, a in zip(traj_ids, adv)}
    metrics = client.update(advantages, clip=args.clip, kl_coef=args.kl_coef,
                            ratio_max=args.ratio_max, adv_clip=args.adv_clip,
                            update_epochs=getattr(args, "update_epochs", 1))
    metrics["mean_return"] = float(arr.mean())
    metrics["max_return"] = float(arr.max())
    metrics["returns"] = [round(x, 4) for x in returns]
    metrics["reward_std"] = float(arr.std())
    if vlm_scores:
        metrics["vlm_mean"] = round(float(np.mean(vlm_scores)), 4)
        metrics["vlm_scores"] = [round(x, 4) for x in vlm_scores]
    # Boundary-hold diagnostics: how often did group members reach success, and of the
    # post-success hold steps, what fraction were "stopped" (eef displacement <= radius).
    succ_holds = [h for h in hold_stats_all if h.get("success_step") is not None]
    metrics["n_success_hold"] = len(succ_holds)
    stay_fracs = [h["hold_stay_frac"] for h in succ_holds if h.get("hold_stay_frac") is not None]
    metrics["mean_hold_stay_frac"] = round(float(np.mean(stay_fracs)), 4) if stay_fracs else None
    return metrics


def vlm_gate(client, args, reward_cfg, seeds, out_dir, eval_split="target"):
    """Verification GATE for the real VLM scorer (task standing gate, run BEFORE training).

    Rolls out deterministic oracle-planner eval episodes, captures each FINAL frame plus
    the simulator's official-success label, then scores every final frame with the real
    Qwen3-VL VQA scorer under all three questions. A valid auxiliary scorer must assign a
    HIGHER P(yes) to genuinely-successful episodes than to failed ones (positive/negative
    separation). We report per-question mean P(yes | success) vs mean P(yes | fail), the
    separation margin, and a threshold-free ROC-AUC. The gate PASSES if the primary
    'success' question separates the two classes (AUC >= 0.65 and margin > 0).
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    questions = ["success", "grasp", "progress"]
    rows = []
    for i, seed in enumerate(seeds):
        genv, sim = _make_env(eval_split, seed)
        try:
            res, final_obs = eval_episode(sim, client, args, reward_cfg, seed,
                                          return_final_obs=True)
            img = _scoring_image(final_obs)
            scores = {q: client.vlm_score(img, q) for q in questions}
        finally:
            genv.close()
        row = {"seed": seed, "official_success": res["official_success"],
               "grasp_success": res["grasp_success"], "vlm": scores}
        rows.append(row)
        print(f"[vlm-gate] {i+1}/{len(seeds)} seed={seed} succ={res['official_success']} "
              f"grasp={res['grasp_success']} P(yes)={ {q: round(scores[q],3) for q in questions} }",
              flush=True)

    def _auc(pos, neg):
        if not pos or not neg:
            return None
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return wins / (len(pos) * len(neg))

    report = {"n": len(rows), "questions": {}}
    for q in questions:
        # success-vs-fail on the OFFICIAL predicate, and grasp-vs-nograsp for the grasp Q
        succ = [r["vlm"][q] for r in rows if r["official_success"]]
        fail = [r["vlm"][q] for r in rows if not r["official_success"]]
        gyes = [r["vlm"][q] for r in rows if r["grasp_success"]]
        gno = [r["vlm"][q] for r in rows if not r["grasp_success"]]
        mean = lambda xs: (round(sum(xs) / len(xs), 4) if xs else None)
        report["questions"][q] = {
            "n_success": len(succ), "n_fail": len(fail),
            "mean_P_success": mean(succ), "mean_P_fail": mean(fail),
            "success_margin": (round(mean(succ) - mean(fail), 4) if succ and fail else None),
            "auc_success_vs_fail": (round(_auc(succ, fail), 4) if _auc(succ, fail) is not None else None),
            "n_grasp": len(gyes), "n_nograsp": len(gno),
            "mean_P_grasp": mean(gyes), "mean_P_nograsp": mean(gno),
            "grasp_margin": (round(mean(gyes) - mean(gno), 4) if gyes and gno else None),
            "auc_grasp_vs_nograsp": (round(_auc(gyes, gno), 4) if _auc(gyes, gno) is not None else None),
        }
    # PASS criterion: the primary success question separates success from failure.
    sq = report["questions"]["success"]
    gq = report["questions"]["grasp"]
    success_ok = (sq["auc_success_vs_fail"] is not None and sq["auc_success_vs_fail"] >= 0.65
                  and sq["success_margin"] is not None and sq["success_margin"] > 0)
    grasp_ok = (gq["auc_grasp_vs_nograsp"] is not None and gq["auc_grasp_vs_nograsp"] >= 0.65
                and gq["grasp_margin"] is not None and gq["grasp_margin"] > 0)
    report["gate_pass"] = bool(success_ok or grasp_ok)
    report["gate_detail"] = {"success_question_separates": success_ok,
                             "grasp_question_separates": grasp_ok}
    report["rows"] = rows
    (out / "vlm_gate.json").write_text(json.dumps(report, indent=2))
    print("=== VLM GATE:", "PASS" if report["gate_pass"] else "FAIL",
          "| success AUC", sq["auc_success_vs_fail"], "margin", sq["success_margin"],
          "| grasp AUC", gq["auc_grasp_vs_nograsp"], "margin", gq["grasp_margin"], flush=True)
    return report


def eval_pool(sim_factory, client, args, reward_cfg, seeds, out_dir, tag):
    results = []
    for i, seed in enumerate(seeds):
        genv, sim = sim_factory(seed)
        try:
            save = out_dir if i < args.save_videos else None
            r = eval_episode(sim, client, args, reward_cfg, seed, out_dir=save, tag=tag)
        finally:
            genv.close()
        results.append(r)
        print(f"[{tag}] {i+1}/{len(seeds)} seed={seed} official={r['official_success']} grasp={r['grasp_success']}", flush=True)
    off = (sum(r["official_success"] for r in results) / len(results)) if results else 0.0
    grasp = (sum(r["grasp_success"] for r in results) / len(results)) if results else 0.0
    return {"n": len(results), "official_success_rate": off, "grasp_success_rate": grasp,
            "episodes": results}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainer-port", type=int, default=10088)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--split", default="target")
    ap.add_argument("--eval-split", default="target")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--heldout-n", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.6)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--kl-coef", type=float, default=0.005)
    ap.add_argument("--ratio-max", type=float, default=10.0)
    ap.add_argument("--adv-clip", type=float, default=3.0)
    ap.add_argument("--train-skill", default="grasp", choices=("grasp", "move_holding", "place"))
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--eval-seed-base", type=int, default=5000)
    ap.add_argument("--heldout-seed-base", type=int, default=9000)
    ap.add_argument("--reward-variant", default="simulator_milestones",
                    choices=("simulator_milestones", "simulator_terminal_only"))
    ap.add_argument("--vlm-weight", type=float, default=0.0,
                    help="Weight of the REAL Qwen3-VL VQA auxiliary score added to the GRPO "
                         "return during TRAINING. 0.0 = sim-only; >0 = sim+VLM ablation.")
    ap.add_argument("--update-epochs", type=int, default=1,
                    help="PPO/GRPO optimizer epochs per rollout batch (forwarded to trainer "
                         "op=update). 1 = single-step group-relative REINFORCE; pi-RL uses 4.")
    ap.add_argument("--vlm-question", default="grasp",
                    choices=("grasp", "progress", "success"),
                    help="Which VQA question the auxiliary scorer asks each rollout.")
    ap.add_argument("--horizon-grasp", type=int, default=120)
    ap.add_argument("--horizon-move", type=int, default=120)
    ap.add_argument("--horizon-place", type=int, default=150)
    ap.add_argument("--max-skill-calls", type=int, default=3)
    ap.add_argument("--save-videos", type=int, default=6)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--vlm-gate-n", type=int, default=0,
                    help="If >0, run ONLY the VLM-scorer verification gate on this many "
                         "eval seeds (no training), write vlm_gate.json, and exit.")
    ap.add_argument("--ckpt-name", default="grpo_trained.pt")
    ap.add_argument("--load-ckpt", default=None,
                    help="Load a saved LoRA checkpoint before AFTER-eval phase. "
                         "For standalone post-hoc eval of a saved ARM checkpoint.")
    # Boundary-compliance post-success hold (operator decision B, run30)
    ap.add_argument("--hold-steps", type=int, default=0,
                    help="post-success hold window length (steps). 0 = disabled (break on "
                         "success, prior behaviour). >0 keeps stepping the same instruction and "
                         "applies the stay/drift boundary reward.")
    ap.add_argument("--hold-stay-bonus", type=float, default=0.05)
    ap.add_argument("--hold-drift-penalty", type=float, default=0.10)
    ap.add_argument("--hold-stay-radius", type=float, default=0.02)
    ap.add_argument("--hold-drop-penalty", type=float, default=0.5)
    # Operator MOVE-first: start the trained skill from its prerequisite skill's
    # SUCCESS entry-state (e.g. MOVE from a deterministic grasp-success) instead of
    # the fresh lid-on-counter reset. Default OFF (backward-compatible GRASP-from-reset).
    ap.add_argument("--entry-from-grasp", action="store_true",
                    help="For --train-skill move_holding: reach a deterministic (eta=0) "
                         "GRASP-success entry-state before MOVE train/eval (snapshot once, "
                         "restore per group member). No-op for grasp/place.")
    ap.add_argument("--entry-scan-cap", type=int, default=0,
                    help="With --entry-from-grasp: scan up to this many candidate seeds per "
                         "band to prefilter grasp-entry-reaching seeds for the eval/train "
                         "pools (cached to entry_seeds.json). 0 = no prefilter (raw seeds, "
                         "non-entering ones are SKIPPED).")
    return ap


def main():
    my, rest = build_parser().parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    for k in ("group", "eta", "clip", "kl_coef", "ratio_max", "adv_clip", "train_skill",
              "horizon_grasp", "horizon_move", "horizon_place",
              "max_skill_calls", "seed_base", "save_videos", "split",
              "vlm_weight", "vlm_question", "update_epochs",
              "hold_steps", "hold_stay_bonus", "hold_drift_penalty",
              "hold_stay_radius", "hold_drop_penalty", "entry_from_grasp"):
        setattr(args, k, getattr(my, k))
    out = Path(my.out); out.mkdir(parents=True, exist_ok=True)

    reward_cfg = RewardConfig(mode="simulator", horizon=my.horizon_place,
                              use_milestones=(my.reward_variant == "simulator_milestones"))

    client = TrainerClient(my.model_path, my.server_addr, my.trainer_port,
                           args.robot_type, args.crop_ratio)

    def train_factory(seed):
        return _make_env(my.split, seed)

    def eval_factory(seed):
        return _make_env(my.eval_split, seed)

    def heldout_factory(seed):
        return _make_env(my.eval_split, seed)

    log = {"config": {**vars(my), "reward_variant": my.reward_variant,
                      "vlm_weight": my.vlm_weight, "vlm_question": my.vlm_question},
           "phases": {}}
    (out / "train_log.jsonl").write_text("")

    eval_seeds = [my.eval_seed_base + i for i in range(my.eval_n)]
    heldout_seeds = [my.heldout_seed_base + i for i in range(my.heldout_n)]

    # MOVE-first mode: the trained skill is a single skill evaluated from its
    # grasp-success entry-state, so before/after/heldout use eval_skill_pool (isolated
    # single-skill success rate) instead of the full-plan eval_pool.
    entry_mode = bool(my.entry_from_grasp) and my.train_skill in ("move_holding",)
    train_skill_enum = {"grasp": Skill.GRASP, "move_holding": Skill.MOVE_HOLDING,
                        "place": Skill.PLACE}[my.train_skill]

    def _rate_str(d):
        if entry_mode:
            return f"skill_success={d.get('skill_success_rate')} (n_entered={d.get('n_entered')}/{d.get('n')})"
        return f"official={d.get('official_success_rate')} grasp={d.get('grasp_success_rate')}"

    def run_eval(factory, seeds, subdir, tag):
        if entry_mode:
            return eval_skill_pool(factory, client, args, reward_cfg, seeds,
                                   train_skill_enum, True, out / subdir, tag)
        return eval_pool(factory, client, args, reward_cfg, seeds, out / subdir, tag)

    # ---- Entry-seed prefilter (MOVE-first) ----
    # In entry_mode, most raw seeds do NOT reach the deterministic grasp entry-state
    # (base grasp ~17%), so a raw eval/train pool is mostly SKIPPED and n_entered is
    # tiny/noisy. Prefilter: scan candidate seeds, keep only grasp-entry-reaching ones,
    # split disjointly into eval (paired before/after) and train pools. Cache the found
    # pool to entry_seeds.json for reproducibility + rerun skip. Only when --entry-scan-cap>0.
    if entry_mode and my.entry_scan_cap > 0:
        cache = out / "entry_seeds.json"
        if cache.exists():
            pool = json.loads(cache.read_text())
            eval_seeds, train_seeds = pool["eval_seeds"], pool["train_seeds"]
            print(f"[entry-prefilter] reuse cache: {len(eval_seeds)} eval + {len(train_seeds)} train", flush=True)
        else:
            need_eval, need_train = my.eval_n, my.iters
            # eval pool from the eval-seed band, train pool from the train-seed band (disjoint bands)
            eval_cands = [my.eval_seed_base + i for i in range(my.entry_scan_cap)]
            train_cands = [my.seed_base + i for i in range(my.entry_scan_cap)]
            eval_seeds = build_entry_seed_pool(eval_factory, client, args, eval_cands,
                                               train_skill_enum, need_eval, tag="_eval")
            train_seeds = build_entry_seed_pool(train_factory, client, args, train_cands,
                                                train_skill_enum, need_train, tag="_train")
            cache.write_text(json.dumps({"eval_seeds": eval_seeds, "train_seeds": train_seeds,
                                         "scan_cap": my.entry_scan_cap}, indent=2))
            print(f"[entry-prefilter] found {len(eval_seeds)}/{need_eval} eval + "
                  f"{len(train_seeds)}/{need_train} train grasp-entry seeds (cap {my.entry_scan_cap})", flush=True)
        log["entry_prefilter"] = {"eval_seeds": eval_seeds, "train_seeds": train_seeds}
    else:
        train_seeds = None  # non-entry mode: train uses seed_base+it inline

    # ---- VLM-scorer verification GATE (standing gate: run BEFORE any ablation train) ----
    if my.vlm_gate_n > 0:
        gate_seeds = [my.eval_seed_base + i for i in range(my.vlm_gate_n)]
        rep = vlm_gate(client, args, reward_cfg, gate_seeds, out, eval_split=my.eval_split)
        (out / "run_summary.json").write_text(json.dumps({"vlm_gate": rep}, indent=2, default=str))
        client.close()
        print("=== DONE (vlm-gate) ===", flush=True)
        return

    # ---- EVAL(before) ----
    if not my.skip_eval:
        t0 = time.time()
        before = run_eval(eval_factory, eval_seeds, "eval_before", "before_")
        before["seconds"] = round(time.time() - t0, 1)
        log["phases"]["eval_before"] = before
        (out / "eval_before.json").write_text(json.dumps(before, indent=2))
        print("EVAL(before):", _rate_str(before), flush=True)

    # ---- TRAIN ----
    curve = []
    with open(out / "train_log.jsonl", "a") as fh:
        for it in range(my.iters):
            # In entry_mode with a prefiltered pool, iterate over grasp-entry seeds
            # (cycled if fewer were found than iters) so every train step gets a real
            # entry-state instead of being SKIPPED. Otherwise seed_base+it as before.
            if train_seeds:
                seed = train_seeds[it % len(train_seeds)]
            else:
                seed = my.seed_base + it
            m = train_iteration(client, args, reward_cfg, seed, it)
            m["iter"] = it
            m["seed"] = seed
            curve.append({"iter": it, "loss": m["loss"], "mean_return": m["mean_return"],
                          "grad_norm": m["grad_norm"], "mean_ratio": m["mean_ratio"],
                          "skipped": m.get("skipped")})
            fh.write(json.dumps(m) + "\n"); fh.flush()
            if m.get("skipped"):
                print(f"[train] it={it} SKIPPED ({m['skipped']}) — no entry-state this seed", flush=True)
            else:
                print(f"[train] it={it} loss={m['loss']:.4f} mean_return={m['mean_return']:.3f} "
                      f"grad_norm={m['grad_norm']:.3f} ratio={m['mean_ratio']:.3f} "
                      f"n_succ_hold={m.get('n_success_hold')} mem={m.get('peak_mem_gb')}", flush=True)
    log["phases"]["train_curve"] = curve

    # Checkpoint path must be on the TRAINER SERVER's filesystem, which has /train mounted
    # (= /home/v4/rl-train-t_3ed65912 on the host). /out is only visible to the CLIENT.
    ckpt_server_path = f"/train/results/{Path(my.out).name}/{my.ckpt_name}"
    save_res = client.save(ckpt_server_path)
    log["checkpoint"] = save_res

    # ---- EVAL(after) ----
    if not my.skip_eval:
        # If --load-ckpt is given (post-hoc eval mode), load that checkpoint now
        if my.load_ckpt:
            load_res = client.load(my.load_ckpt)
            print(f"Loaded checkpoint {my.load_ckpt}: {load_res}", flush=True)
        t0 = time.time()
        after = run_eval(eval_factory, eval_seeds, "eval_after", "after_")
        after["seconds"] = round(time.time() - t0, 1)
        log["phases"]["eval_after"] = after
        (out / "eval_after.json").write_text(json.dumps(after, indent=2))
        print("EVAL(after):", _rate_str(after), flush=True)

        # ---- HELDOUT (disjoint geometry/seed) ----
        t0 = time.time()
        heldout = run_eval(heldout_factory, heldout_seeds, "eval_heldout", "heldout_")
        heldout["seconds"] = round(time.time() - t0, 1)
        log["phases"]["eval_heldout"] = heldout
        (out / "eval_heldout.json").write_text(json.dumps(heldout, indent=2))

    (out / "run_summary.json").write_text(json.dumps(log, indent=2, default=str))
    client.close()
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
