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
Both are pure-simulator configurations from the environment RewardManager.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
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
from advantage import compute_group_advantages  # noqa: E402  (this card's src dir on sys.path[0])
from adaptive_curriculum import AdaptiveCurriculum, CurriculumTransaction  # noqa: E402
from update_batch import (  # noqa: E402
    execute_batched_update, group_env_seed, recv_exact, retry_idempotent_rpc,
    trajectory_action_seed, validate_batch_config,
)
from training_correctness import (  # noqa: E402
    ExactHoldWindow, RewardComponents, append_progress, hold_enabled_for_variant,
    nonduplicated_skill_reward,
    reset_gated_store, skill_terminal_enabled_for_variant, skill_timeout_reward,
    verify_deployment_manifest,
)
from reward import RewardConfig, RewardManager, official_success, HoldConfig, hold_step_reward  # noqa: E402
from skill_manager import (  # noqa: E402
    MonitorConfig, OraclePlanner, Skill, SkillMonitor, SkillOutcome,
    SKILL_INSTRUCTION,
)

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
from robocasa.utils.env_utils import convert_action  # noqa: E402

SKILL_KEY = {Skill.GRASP: "grasp", Skill.MOVE_HOLDING: "move_holding", Skill.PLACE: "place"}


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
        self.sample_rpc_seconds = 0.0
        self.sample_rpc_calls = 0
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
        ln = recv_exact(self.sock, 4)
        n = struct.unpack(">I", ln)[0]
        data = recv_exact(self.sock, n)
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
        t_rpc = time.perf_counter()
        resp = self._rpc(req)
        self.sample_rpc_seconds += time.perf_counter() - t_rpc
        self.sample_rpc_calls += 1
        actions = resp["actions"]
        decoded = self.processor.decode_action(actions, robot_type=self.robot_type)
        decoded = decoded[0, :, : self.ACTION_DIM]
        decoded = decoded.float().cpu().numpy() if hasattr(decoded, "float") else np.asarray(decoded)
        return np.asarray(decoded, dtype=np.float32), resp.get("logprob")

    def update(self, advantages, clip=0.1, kl_coef=0.005, ratio_max=10.0, adv_clip=3.0,
               update_epochs=1, target_kl=None, update_id=None, curriculum_state=None):
        request = {"op": "update", "advantages": advantages, "clip": clip,
                   "kl_coef": kl_coef, "ratio_max": ratio_max, "adv_clip": adv_clip,
                   "update_epochs": update_epochs, "target_kl": target_kl,
                   "update_id": update_id, "curriculum_state": curriculum_state}
        def reconnect():
            self.close()
            self._connect()
        # The trainer caches completed update IDs. If the first response was lost,
        # reconnecting and replaying this exact request does not execute the optimizer again.
        return retry_idempotent_rpc(self._rpc, reconnect, request)

    def save(self, path, *, update_index=0, train_meta=None):
        return self._rpc({"op": "save", "path": path, "update_index": update_index,
                          "train_meta": train_meta})

    def reset_store(self):
        """Clear the trainer's buffered rollout chunks without an optimizer step.
        Called after the difficulty-band prefilter so its eta>0 scan rollouts do not
        leak into the first real training update."""
        return self._rpc({"op": "reset"})

    def metrics(self):
        return self._rpc({"op": "metrics"})

    def export_store(self, trajectory_ids, path, *, drop_after_export=False):
        """Persist selected sampled chunks without asking the optimizer to update."""
        return self._rpc({"op": "export_store", "trajectory_ids": list(trajectory_ids),
                          "path": str(path), "drop_after_export": bool(drop_after_export)})

    def import_store(self, path, expected_sha256):
        """Restore a collector payload for the existing update path to consume later."""
        return self._rpc({"op": "import_store", "path": str(path),
                          "expected_sha256": expected_sha256})

    def discard_store(self, trajectory_ids):
        """Drop failed-attempt chunks without performing an optimizer update."""
        return self._rpc({"op": "discard_store", "trajectory_ids": list(trajectory_ids)})

    def config(self, code_rev=None):
        """Fetch the full trainer configuration (optimizer/LR/grad-clip/LoRA/trainable count/
        sampler/eta) for the run summary. Measurement fix: every run summary is self-describing."""
        return self._rpc({"op": "config", "code_rev": code_rev})

    def load(self, path):
        return self._rpc({"op": "load", "path": path})

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
    reward_components = RewardComponents()
    chunk_idx = 0   # AUDIT FIX #2: monotonic per-skill chunk counter -> unique RNG per replan
    p = sim.predicates()
    prev_eef_lid_dist = float(p.get("eef_lid_dist", 0.0)) if approach_coef > 0.0 else None
    # --- post-success hold bookkeeping (operator decision B) ---
    success_step = None                 # environment step at first success (diagnostics)
    success_chunk = None                # elapsed action chunks at first success (Z-1 decay)
    hold_window = ExactHoldWindow(hold_steps)
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
    while steps < horizon or (hold_window.latched and hold_window.held_steps < hold_steps):
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
        # Z-1 completion decay is indexed by elapsed action chunks, not 20 Hz env steps.
        rb = reward_mgr.step_reward(chunk_idx, p, done=bool(done), truncated=bool(trunc))
        reward_sum += rb.primary
        reward_components.official_terminal += rb.terminal
        reward_components.milestone += rb.milestone
        reward_components.drop += getattr(rb, "object_dropped", 0.0)
        reward_components.collision += getattr(rb, "disallowed_collision", 0.0)
        reward_components.timeout += getattr(rb, "timeout", 0.0)
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
                reward_components.hold_stay += hold_cfg.stay_bonus
            else:
                reward_components.hold_drift += -hold_cfg.drift_penalty * (
                    disp / max(hold_cfg.stay_radius_m, 1e-6) - 1.0)
            if not _success_now(p):
                reward_components.hold_lapse += -hold_cfg.drop_success_penalty
        prev_eef = list(p.get("eef_pos", prev_eef))
        # Approach shaping: reward getting closer to the lid (training-only)
        if approach_coef > 0.0 and prev_eef_lid_dist is not None:
            curr_dist = float(p.get("eef_lid_dist", prev_eef_lid_dist))
            approach_reward = approach_coef * (prev_eef_lid_dist - curr_dist)
            reward_sum += approach_reward
            reward_components.approach += approach_reward
            prev_eef_lid_dist = curr_dist
        outcome = monitor.update(p, steps, horizon)
        # Binary task reward ends PLACE as soon as stable lid closure has been debounced.
        # Robot/gripper motion is intentionally not part of this success condition.
        if (getattr(args, "reward_variant", "simulator_terminal_only") == "simulator_terminal_only"
                and skill is Skill.PLACE and rb.success):
            outcome = SkillOutcome.SUCCESS
        if frames is not None and (steps % args.video_stride == 0 or outcome or done or trunc):
            frames.append(rollout.make_video_frame(obs))
        # Latch the first success once, pay the decayed skill terminal once, then execute
        # exactly hold_steps subsequent env steps even if the predicate lapses.
        if outcome is SkillOutcome.SUCCESS and success_step is None:
            success_step = steps
            success_chunk = chunk_idx
            terminal_reward = (
                nonduplicated_skill_reward(
                    reward_components.official_terminal,
                    success_chunk,
                    getattr(args, "skill_success_reward", 1.0),
                    getattr(args, "skill_success_gamma", 0.998),
                    getattr(args, "skill_success_decay", True),
                )
                if skill_terminal_enabled_for_variant(
                    getattr(args, "reward_variant", "simulator_terminal_only")
                )
                else 0.0
            )
            reward_sum += terminal_reward
            reward_components.skill_terminal += terminal_reward
            if hold_cfg is None or hold_window.observe(True, terminated=bool(done or trunc)):
                break
            continue
        if hold_window.latched:
            outcome = SkillOutcome.SUCCESS
            if hold_window.observe(_success_now(p), terminated=bool(done or trunc)):
                break
            continue
        if outcome or done or trunc:
            break
    if outcome is None:
        outcome = SkillOutcome.TIMEOUT
    timeout_reward = skill_timeout_reward(outcome.name, timeout_penalty)
    reward_sum += timeout_reward
    reward_components.timeout += timeout_reward
    if save_video is not None and frames is not None:
        import imageio.v2 as imageio
        imageio.mimsave(save_video, frames, fps=args.video_fps)
    hold_stats = {"success_step": success_step, "success_chunk": success_chunk,
                  "held_steps": held_steps,
                  "hold_stay": hold_stay, "hold_reward": round(hold_reward, 4),
                  "hold_stay_frac": (round(hold_stay / held_steps, 3) if held_steps else None),
                  "hold_lapse_steps": hold_window.lapse_steps,
                  "reward_components": reward_components.as_dict()}
    if abs(reward_components.total() - reward_sum) > 1e-5:
        raise RuntimeError(
            f"reward component mismatch: components={reward_components.total()} total={reward_sum}")
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
_ENTRY_REWARD_CFG = RewardConfig( horizon=200, use_milestones=True)


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


def scan_seed_difficulty(factory, client, args, reward_cfg, seed, skill):
    """Roll out `group` STOCHASTIC (eta>0, the SAME eta training uses) members of the
    trained skill from seed's fresh reset with the CURRENT (base) policy, and return how
    many succeeded. This is the difficulty signal used by the band prefilter: a seed whose
    base success count is 0/group is TOO HARD (all-failure, GRPO std=0) and 8/8 is TOO EASY
    (all-success, near-zero reward variance) -- both get GATED and their learning signal
    thrown away. The band [LOW,HIGH] keeps only the mixed-difficulty seeds in between.

    Mirrors train_iteration's non-entry group rollout EXACTLY (new env per member, same
    reset seed, same eta, same per-member action seed derivation) so the scanned success
    count matches what the first training iteration on that seed would see. Does NOT call
    op_update; the caller clears the trainer store afterwards via client.reset_store().
    """
    n_succ = 0
    for m in range(args.group):
        genv, sim = factory(seed)
        try:
            obs, _ = rollout.reset_env(genv, seed)
            sim.rest_lid_pos = sim.lid_pos()
            reward_mgr = RewardManager(reward_cfg)
            # distinct traj_id namespace so scan rollouts never collide with train iters
            traj_id = f"bandscan_s{seed}_m{m}"
            # action-noise seed: derived from (seed, member) but offset into a disjoint
            # range from the training seeds (args.seed_base + it*100 + m) to avoid overlap.
            scan_action_seed = int(seed) * 977 + m + 500000
            _obs, outcome, _r, _steps, _p, _h = _run_one_skill(
                sim, client, obs, args, skill, reward_mgr, eta=args.eta,
                traj_id=traj_id, seed=scan_action_seed)
            if outcome is SkillOutcome.SUCCESS:
                n_succ += 1
        finally:
            genv.close()
    return n_succ


def build_difficulty_band_pool(factory, client, args, reward_cfg, candidate_seeds,
                               skill, band, need, tag=""):
    """Prefilter candidate env seeds by BASE-policy success-count difficulty band.

    For each candidate seed, scan_seed_difficulty() rolls out `group` stochastic members
    with the current policy and counts successes; a seed is kept for the train pool only
    if LOW <= n_succ <= HIGH (a mid-difficulty seed that yields a real GRPO advantage
    signal instead of an all-success / all-failure GATED group). Scans until `need` seeds
    are collected or candidates run out.

    Returns (train_seeds, per_seed_nsucc) where per_seed_nsucc maps EVERY scanned seed to
    its success count (for the cache + diagnostics, incl. the rejected 0/8 and 8/8 seeds).
    Backward-compatible: only invoked when --train-difficulty-band is set (GRASP-from-reset
    mode, entry_from_grasp=False); the entry-mode prefilter path is unchanged.
    """
    lo, hi = band
    good = []
    per_seed = {}
    for seed in candidate_seeds:
        if len(good) >= need:
            break
        n_succ = scan_seed_difficulty(factory, client, args, reward_cfg, seed, skill)
        per_seed[seed] = n_succ
        kept = (lo <= n_succ <= hi)
        if kept:
            good.append(seed)
        print(f"[band-scan{tag}] seed={seed} n_succ={n_succ}/{args.group} "
              f"band=[{lo},{hi}] kept={kept} ({len(good)}/{need})", flush=True)
    return good, per_seed


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


def train_iteration(client, args, reward_cfg, seed, it, *, defer_update=False):
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
    # approach shaping disabled (operator 2026-09-16): reward only at final success,
    # no dense per-step distance shaping before success.
    train_approach_coef = 0.0
    train_timeout_penalty = (0.0 if getattr(args, "reward_variant", "") ==
                             "simulator_terminal_only" else 0.5)
    # Boundary-compliance hold (operator decision B): reward stopping after success. Config
    # from args (0 hold_steps disables it -> identical to the pre-B behaviour).
    hold_steps = int(getattr(args, "hold_steps", 0))
    hold_cfg = HoldConfig(stay_bonus=getattr(args, "hold_stay_bonus", 0.05),
                          drift_penalty=getattr(args, "hold_drift_penalty", 0.10),
                          stay_radius_m=getattr(args, "hold_stay_radius", 0.02),
                          drop_success_penalty=getattr(args, "hold_drop_penalty", 0.5)) \
        if hold_enabled_for_variant(getattr(args, "reward_variant", "terminal_plus_hold"),
                                    hold_steps) else None
    returns = []
    traj_ids = []
    successes = []          # per-member: did the trained skill succeed (for group composition)
    hold_stats_all = []
    reset_seconds = []
    sample_rpc_start = float(getattr(client, "sample_rpc_seconds", 0.0))
    sample_calls_start = int(getattr(client, "sample_rpc_calls", 0))
    entry_from_grasp = getattr(args, "entry_from_grasp", False) and skill in ENTRY_PREREQS

    if entry_from_grasp:
        # Reach the grasp entry-state ONCE (deterministic eta=0), snapshot it, then
        # restore per group member so every member starts byte-identical from the SAME
        # real grasp-success state (skill guidance: group members start byte-identical,
        # diverge under eta>0). Avoids re-running the ~120-step grasp `group` times.
        t_reset = time.perf_counter()
        genv, sim = _make_env(args.split, seed)
        try:
            reset_obs, _ = rollout.reset_env(genv, seed)
            reset_seconds.append(time.perf_counter() - t_reset)
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
                    traj_id=traj_id,
                    seed=(trajectory_action_seed(args.seed_base, it, m) if defer_update
                          else args.seed_base + it * 100 + m),
                    approach_coef=train_approach_coef,
                    timeout_penalty=train_timeout_penalty,
                    hold_cfg=hold_cfg, hold_steps=hold_steps)
                hold_stats_all.append(hstats)
                is_succ = outcome is SkillOutcome.SUCCESS
                successes.append(is_succ)
                hstats["reward_components"]["total"] = r
                returns.append(r)
                traj_ids.append(traj_id)
        finally:
            genv.close()
    else:
        for m in range(args.group):
            t_reset = time.perf_counter()
            genv, sim = _make_env(args.split, seed)
            try:
                obs, _ = rollout.reset_env(genv, seed)   # identical start for all members
                reset_seconds.append(time.perf_counter() - t_reset)
                sim.rest_lid_pos = sim.lid_pos()
                reward_mgr = RewardManager(reward_cfg)
                traj_id = f"iter{it}_m{m}"
                final_obs, outcome, r, steps, p, hstats = _run_one_skill(
                    sim, client, obs, args, skill, reward_mgr, eta=args.eta,
                    traj_id=traj_id,
                    seed=(trajectory_action_seed(args.seed_base, it, m) if defer_update
                          else args.seed_base + it * 100 + m),
                    approach_coef=train_approach_coef,
                    timeout_penalty=train_timeout_penalty,
                    hold_cfg=hold_cfg, hold_steps=hold_steps)
                hold_stats_all.append(hstats)
                # terminal shaping for the skill unit: bonus if the skill's predicate is met
                is_succ = outcome is SkillOutcome.SUCCESS
                successes.append(is_succ)
                hstats["reward_components"]["total"] = r
                returns.append(r)
                traj_ids.append(traj_id)
            finally:
                genv.close()
    arr = np.asarray(returns, dtype=np.float64)
    succ_arr = np.asarray(successes, dtype=bool)
    adv_arr, adv_info = compute_group_advantages(
        arr, succ_arr,
        std_gate=getattr(args, "reward_std_gate", 0.05))
    advantages = {tid: float(a) for tid, a in zip(traj_ids, adv_arr)}
    # low-variance / all-failure / all-equal groups are gated: every advantage is 0, so the
    # trainer's active-filter (adv!=0) skips the optimizer step entirely (no noisy update).
    if adv_info["gated"]:
        metrics = {"loss": 0.0, "grad_norm": 0.0, "mean_ratio": 1.0, "n_chunks": 0,
                   "gated": adv_info["reason"], "reward_std": adv_info["reward_std"]}
        reset_gated_store(client, defer_update)
    elif defer_update:
        metrics = {"loss": 0.0, "grad_norm": 0.0, "mean_ratio": 1.0,
                   "n_chunks": 0, "deferred_update": True}
    else:
        metrics = client.update(advantages, clip=args.clip, kl_coef=args.kl_coef,
                                ratio_max=args.ratio_max, adv_clip=args.adv_clip,
                                update_epochs=getattr(args, "update_epochs", 1),
                                target_kl=getattr(args, "target_kl", None))
    metrics["mean_return"] = float(arr.mean())
    metrics["max_return"] = float(arr.max())
    metrics["returns"] = [round(x, 4) for x in returns]
    metrics["reward_std"] = float(arr.std())
    metrics["n_success_group"] = int(succ_arr.sum())
    metrics["group_composition"] = adv_info["composition"]      # all_success|mixed|all_failure|constant
    metrics["advantage_mode"] = adv_info["mode"]                # group_relative|nonneg_min_baseline|gated
    metrics["advantages"] = {tid: round(float(a), 4) for tid, a in zip(traj_ids, adv_arr)}
    metrics["_advantages"] = advantages
    # Boundary-hold diagnostics: how often did group members reach success, and of the
    # post-success hold steps, what fraction were "stopped" (eef displacement <= radius).
    succ_holds = [h for h in hold_stats_all if h.get("success_step") is not None]
    metrics["n_success_hold"] = len(succ_holds)
    stay_fracs = [h["hold_stay_frac"] for h in succ_holds if h.get("hold_stay_frac") is not None]
    metrics["mean_hold_stay_frac"] = round(float(np.mean(stay_fracs)), 4) if stay_fracs else None
    metrics["reward_components"] = [h["reward_components"] for h in hold_stats_all]
    metrics["skill_success_gamma"] = float(getattr(args, "skill_success_gamma", 0.998))
    metrics["skill_success_decay"] = bool(getattr(args, "skill_success_decay", True))
    metrics["reset_seconds"] = round(sum(reset_seconds), 3)
    metrics["sample_store_rpc_seconds"] = round(
        float(getattr(client, "sample_rpc_seconds", 0.0)) - sample_rpc_start, 3)
    metrics["sample_store_rpc_calls"] = (
        int(getattr(client, "sample_rpc_calls", 0)) - sample_calls_start)
    return metrics


def train_batched_update(client, args, reward_cfg, seed_base, update_index,
                         curriculum=None, curriculum_cache=None):
    """Collect multiple independent groups, then perform exactly one logical update RPC.

    Every group uses one shared env seed internally and a distinct env seed across groups.
    Group-relative advantages are computed before aggregation. Collection continues in
    whole groups until both floors are met: ``groups_per_update`` and
    ``min_trainable_chunks``. The trainer may execute multiple configured optimizer
    epochs inside that one logical update.
    """
    progress_path = getattr(args, "group_progress_path", None)
    t_collect = time.time()
    update_timing = {"seconds": 0.0}

    def select_seed(update, group, used):
        if curriculum is not None:
            return curriculum.next_seed(update * int(args.max_groups_per_update) + group,
                                        exclude=used)
        return group_env_seed(seed_base, update, group), "raw"

    def collect_group(group, env_seed, iteration_token):
        t_group = time.time()
        gm = train_iteration(client, args, reward_cfg, env_seed, iteration_token,
                             defer_update=True)
        gm["wall_seconds"] = round(time.time() - t_group, 3)
        return gm

    def update(advantages):
        started = time.time()
        result = client.update(
            advantages, clip=args.clip, kl_coef=args.kl_coef,
            ratio_max=args.ratio_max, adv_clip=args.adv_clip,
            update_epochs=getattr(args, "update_epochs", 1),
            target_kl=getattr(args, "target_kl", None),
            update_id=f"{getattr(args, 'run_id', 'run')}:{update_index}",
            curriculum_state=(curriculum.to_dict() if curriculum is not None else None))
        update_timing["seconds"] = time.time() - started
        return result

    def on_group(group, accumulator):
        store = client.metrics()
        if curriculum is not None:
            curriculum.update(
                group["env_seed"], int(group.get("n_success_group") or 0),
                it=update_index * int(args.max_groups_per_update) + group["group_index"],
                source=group["seed_source"])
            group["adaptive_band_stats"] = curriculum.band_stats()
        if progress_path:
            append_progress(progress_path, {
                "update": update_index, "group": group["group_index"],
                "env_seed": group["env_seed"], "seed_source": group["seed_source"],
                "trajectories": int(args.group),
                "stored_chunks": group["stored_chunks"],
                "trainable_chunks": group["trainable_chunks"],
                "stored_chunks_cumulative": int(store.get("n_chunks", 0)),
                "trainable_chunks_cumulative": accumulator.trainable_chunks,
                "n_success": int(group.get("n_success_group") or 0),
                "composition": group.get("group_composition"),
                "gate_reason": group.get("gated"),
                "reset_seconds": group.get("reset_seconds"),
                "sample_store_rpc_seconds": group.get("sample_store_rpc_seconds"),
                "rollout_seconds": group["wall_seconds"],
                "trainer_free_gb": store.get("free_gb"),
                "adaptive_band_stats": group.get("adaptive_band_stats"),
            })

    transaction = (CurriculumTransaction(curriculum, curriculum_cache)
                   if curriculum is not None else contextlib.nullcontext())
    with transaction as curriculum_transaction:
        result = execute_batched_update(
            update_index=update_index,
            min_groups=int(getattr(args, "groups_per_update", 1)),
            min_trainable_chunks=int(getattr(args, "min_trainable_chunks", 0)),
            max_groups=int(getattr(args, "max_groups_per_update", 64)),
            select_seed=select_seed, collect_group=collect_group,
            get_store=client.metrics, update=update, reset_store=client.reset_store,
            on_group=on_group)
        if curriculum_transaction is not None and curriculum is not None:
            committed_state = result.get("curriculum_state")
            if committed_state is not None:
                curriculum.load_state(committed_state)
            curriculum_transaction.commit()
    groups = result["group_summaries"]
    collect_seconds = time.time() - t_collect - update_timing["seconds"]
    returns = [r for g in groups for r in g.get("returns", [])]
    n_success = sum(int(g.get("n_success_group") or 0) for g in groups)
    n_traj = len(groups) * int(args.group)
    result.update({
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "max_return": float(np.max(returns)) if returns else 0.0,
        "returns": returns,
        "n_success_group": n_success,
        "n_success_hold": sum(int(g.get("n_success_hold") or 0) for g in groups),
        "trajectories_collected": n_traj,
        "group_composition": [g.get("group_composition") for g in groups],
        "advantage_mode": "per_group_then_aggregate",
        "reward_std": float(np.std(returns)) if returns else 0.0,
        "collect_seconds": round(collect_seconds, 3),
        "optimizer_seconds": round(update_timing["seconds"], 3),
        "trajectory_mean_seconds": round(collect_seconds / max(n_traj, 1), 3),
        "reset_seconds": round(sum(float(g.get("reset_seconds", 0.0)) for g in groups), 3),
        "sample_store_rpc_seconds": round(
            sum(float(g.get("sample_store_rpc_seconds", 0.0)) for g in groups), 3),
        "sample_store_rpc_calls": sum(int(g.get("sample_store_rpc_calls", 0)) for g in groups),
    })
    result["chunk_mean_seconds"] = round(
        result["sample_store_rpc_seconds"] / max(result["stored_chunks_collected"], 1), 4)
    return result


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
    ap.add_argument("--train-source-manifest", default="/train/source_manifest.json")
    ap.add_argument("--env-source-manifest", default="/rl_env/source_manifest.json")
    ap.add_argument("--split", default="target")
    ap.add_argument("--eval-split", default="target")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--groups-per-update", type=int, default=1,
                    help="Minimum number of independent env-seed groups collected before one "
                         "optimizer update. Advantages remain normalized within each group.")
    ap.add_argument("--min-trainable-chunks", type=int, default=0,
                    help="Continue collecting whole groups until the trainer store contains at "
                         "least this many action chunks before the optimizer update.")
    ap.add_argument("--max-groups-per-update", type=int, default=64,
                    help="Safety cap while satisfying --min-trainable-chunks.")
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
    ap.add_argument("--reward-variant", default="simulator_terminal_only",
                    choices=("simulator_milestones", "simulator_terminal_only", "terminal_plus_hold"),
                    help="Reward composition label. simulator_milestones: per-milestone bonuses. "
                         "terminal_plus_hold: terminal success (+1.0) PLUS the 20-step post-success "
                         "hold shaping (the ACTUAL reward when --hold-steps>0; the honest rename of "
                         "the mislabeled 'terminal_only'). simulator_terminal_only: pure terminal "
                         "official whole-task predicate only: task failure=0, task success=1; "
                         "no per-skill payment, decay, hold shaping, or failure penalty.")
    ap.add_argument("--reward-std-gate", type=float, default=0.05,
                    help="Minimum within-group reward std for an optimizer update. Groups with a "
                         "smaller std (all-failure or effectively-constant reward) are gated: "
                         "advantage is zeroed and the step is skipped (advantage-handling fix).")
    ap.add_argument("--update-epochs", type=int, default=1,
                    help="PPO/GRPO optimizer epochs per rollout batch (forwarded to trainer "
                         "op=update). 1 = single-step group-relative REINFORCE; pi-RL uses 4.")
    ap.add_argument("--target-kl", type=float, default=None,
                    help="Stop before a later-epoch step when mean KL exceeds this.")
    ap.add_argument("--horizon-grasp", type=int, default=208)
    ap.add_argument("--horizon-move", type=int, default=150)
    ap.add_argument("--horizon-place", type=int, default=200)
    ap.add_argument("--max-skill-calls", type=int, default=3)
    ap.add_argument("--save-videos", type=int, default=6)
    ap.add_argument("--skip-eval", action="store_true")
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
    ap.add_argument("--skill-success-reward", type=float, default=1.0)
    ap.add_argument("--skill-success-gamma", type=float, default=0.998)
    ap.add_argument("--no-skill-success-decay", dest="skill_success_decay",
                    action="store_false", help="Disable gamma^completion_step decay.")
    ap.set_defaults(skill_success_decay=True)
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
    # Difficulty-band seed prefilter (operator redesign t_2907de4f): GRASP-from-reset mode.
    # Keep only MID-difficulty train seeds (base success count inside [LOW,HIGH]) so the
    # group is neither all-failure (0/group, GATED std=0) nor all-success (group/group,
    # near-zero reward variance) -- both throw away the GRPO signal. Independent of the
    # entry-mode (MOVE) --entry-scan-cap prefilter, which filters on ENTRY-REACHED, not
    # success-rate band.
    ap.add_argument("--train-difficulty-band", type=int, nargs=2, default=None,
                    metavar=("LOW", "HIGH"),
                    help="Prefilter train-pool seeds to those whose BASE-policy success "
                         "count over `group` stochastic (eta) rollouts is in [LOW,HIGH]. "
                         "E.g. 1 7 with group 8 keeps mixed seeds, drops 0/8 and 8/8. "
                         "None = no band prefilter (legacy seed_base+it sequential seeds).")
    ap.add_argument("--train-scan-cap", type=int, default=0,
                    help="With --train-difficulty-band: max candidate seeds (seed_base+i) to "
                         "scan for the band. Recommend >= iters*a few so `need` band seeds "
                         "are found. 0 = auto (iters*8).")
    ap.add_argument("--train-pool-need", type=int, default=0,
                    help="How many band-passing train seeds to collect. 0 = auto "
                         "(max(iters, ceil(iters*1.5)) so the pool exceeds iters and every "
                         "train step draws WITHOUT replacement for diversity).")
    ap.add_argument("--seed", type=int, default=12345,
                    help="RNG seed for the per-iter WITHOUT-REPLACEMENT random draw from the "
                         "band-filtered train pool (reproducible; both arms share it so the "
                         "paired comparison uses the same pool + same iter-seed order).")
    # Online adaptive curriculum (operator redesign t_2907de4f, moving band). NO upfront
    # scan: reuse each iter's group rollout n_succ as the seed's current difficulty (EMA
    # cache), and bias the NEXT iter's seed toward the mid band [LOW,HIGH]. The band moves
    # with the policy. Mutually exclusive with the static --train-difficulty-band prefilter.
    ap.add_argument("--adaptive-band", type=int, nargs=2, default=None,
                    metavar=("LOW", "HIGH"),
                    help="Enable the ONLINE adaptive-curriculum sampler: bias per-iter train "
                         "seeds toward those whose CURRENT-policy group n_succ (EMA) is in "
                         "[LOW,HIGH]. E.g. 1 7 with group 8 targets mixed groups, avoiding "
                         "0/8 & 8/8 GATED. The band tracks the policy (no upfront scan).")
    ap.add_argument("--adaptive-universe", type=int, default=0,
                    help="With --adaptive-band: size of the candidate seed universe "
                         "(seed_base .. seed_base+N-1) to sample from. 0 = auto (iters*8).")
    ap.add_argument("--adaptive-explore-frac", type=float, default=0.25,
                    help="With --adaptive-band: probability of drawing an UNSEEN seed to "
                         "probe its difficulty (exploration) vs exploiting known in-band "
                         "seeds. Also forced when no in-band seed is known yet.")
    ap.add_argument("--adaptive-ema", type=float, default=0.5,
                    help="With --adaptive-band: EMA weight on the newest n_succ when "
                         "updating a seed's difficulty (higher = tracks the policy faster).")
    ap.add_argument("--adaptive-avoid-recent", type=int, default=3,
                    help="With --adaptive-band: exclude the last N distinct drawn seeds from "
                         "the exploit pool so no single seed is overfit (per-iter diversity).")
    return ap


def main():
    my, rest = build_parser().parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    for k in ("group", "groups_per_update", "min_trainable_chunks", "max_groups_per_update",
              "eta", "clip", "kl_coef", "ratio_max", "adv_clip", "train_skill",
              "reward_variant",
              "horizon_grasp", "horizon_move", "horizon_place",
              "max_skill_calls", "seed_base", "save_videos", "split",
              "update_epochs", "target_kl", "reward_std_gate",
              "hold_steps", "hold_stay_bonus", "hold_drift_penalty",
              "hold_stay_radius", "hold_drop_penalty", "skill_success_reward",
              "skill_success_gamma", "skill_success_decay", "entry_from_grasp"):
        setattr(args, k, getattr(my, k))
    validate_batch_config(my.groups_per_update, my.min_trainable_chunks,
                          my.max_groups_per_update)
    if my.hold_steps < 0:
        raise ValueError("hold_steps must be non-negative")
    if not (0.0 < my.skill_success_gamma <= 1.0):
        raise ValueError("skill_success_gamma must be in (0, 1]")
    if my.target_kl is not None and my.target_kl <= 0.0:
        raise ValueError("target_kl must be positive when set")
    out = Path(my.out); out.mkdir(parents=True, exist_ok=True)
    args.run_id = out.name
    source_state = {
        "train": verify_deployment_manifest("/train", my.train_source_manifest),
        "env": verify_deployment_manifest("/rl_env", my.env_source_manifest),
    }

    # simulator_terminal_only forces a PURE terminal-only ablation: zero the hold shaping
    # weights (the hold window is retained only as a verifier condition, not a reward).
    # terminal_plus_hold keeps the hold shaping (the honest label for the run that ships
    # +1.0 terminal AND the 20-step hold bonus). simulator_milestones enables milestones.
    if my.reward_variant == "simulator_terminal_only":
        args.hold_stay_bonus = 0.0
        args.hold_drift_penalty = 0.0
        args.hold_drop_penalty = 0.0
        args.skill_success_decay = False

    reward_cfg = (
        RewardConfig.binary(horizon=my.horizon_place)
        if my.reward_variant == "simulator_terminal_only"
        else RewardConfig(
            horizon=my.horizon_place,
            use_milestones=(my.reward_variant == "simulator_milestones"),
        )
    )

    client = TrainerClient(my.model_path, my.server_addr, my.trainer_port,
                           args.robot_type, args.crop_ratio)

    def train_factory(seed):
        return _make_env(my.split, seed)

    def eval_factory(seed):
        return _make_env(my.eval_split, seed)

    def heldout_factory(seed):
        return _make_env(my.eval_split, seed)

    log = {"config": {**vars(my), "reward_variant": my.reward_variant},
           "source": source_state, "phases": {}}
    log["algorithm"] = (
        "group-relative policy gradient (single on-policy epoch)"
        if my.update_epochs == 1 else "PPO-clipped GRPO (multi-epoch)"
    )
    # measurement fix: capture the full trainer-side config (optimizer/LR/grad-clip/LoRA
    # rank-alpha-targets/trainable count/sampler/eta/code rev) so the run summary is
    # self-describing. code_rev from env (set by the launcher) if available.
    try:
        import os as _os
        log["trainer_config"] = client.config(code_rev=_os.environ.get("CODE_REV"))
    except Exception as _e:
        log["trainer_config"] = {"error": str(_e)}
    trainer_cfg = log["trainer_config"]
    server_args = trainer_cfg.get("config", {}) if isinstance(trainer_cfg, dict) else {}
    adapter_skills = [s for s in str(server_args.get("adapter_skills", "")).split(",") if s]
    if my.iters > 0 and adapter_skills and my.train_skill not in adapter_skills:
        raise ValueError(f"trainer has adapters for {adapter_skills}, cannot train {my.train_skill!r}")
    (out / "train_log.jsonl").write_text("")
    args.group_progress_path = str(out / "group_progress.jsonl")
    Path(args.group_progress_path).write_text("")

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
    elif (not entry_mode) and my.train_difficulty_band is not None:
        # ---- Difficulty-band seed prefilter (GRASP-from-reset) ----
        # Scan the train-seed band with the BASE policy and keep only MID-difficulty seeds
        # (base success count inside [LOW,HIGH]); pool > iters so the train loop draws
        # without replacement for diversity. Cached to train_pool.json (reproducibility +
        # rerun skip). eval/heldout seeds are NOT band-filtered (fixed held-out set).
        band = tuple(my.train_difficulty_band)
        need_train = my.train_pool_need or max(my.iters, -(-my.iters * 3 // 2))  # ceil(iters*1.5)
        scan_cap = my.train_scan_cap or (my.iters * 8)
        cache = out / "train_pool.json"
        if cache.exists():
            pool = json.loads(cache.read_text())
            train_seeds = pool["train_seeds"]
            per_seed_nsucc = pool.get("per_seed_nsucc", {})
            print(f"[band-prefilter] reuse cache: {len(train_seeds)} train seeds "
                  f"band={pool.get('band')}", flush=True)
        else:
            train_cands = [my.seed_base + i for i in range(scan_cap)]
            train_seeds, per_seed_nsucc = build_difficulty_band_pool(
                train_factory, client, args, reward_cfg, train_cands,
                train_skill_enum, band, need_train, tag="_train")
            # the eta>0 scan rollouts populated the trainer store; clear it so they do NOT
            # enter the first real op_update batch.
            try:
                cleared = client.reset_store()
                print(f"[band-prefilter] cleared trainer store after scan: {cleared}", flush=True)
            except Exception as _e:
                print(f"[band-prefilter] store reset failed (non-fatal): {_e}", flush=True)
            # distribution of scanned success counts (incl. rejected 0/group & group/group)
            hist = {}
            for k in per_seed_nsucc.values():
                hist[k] = hist.get(k, 0) + 1
            cache.write_text(json.dumps({
                "band": list(band), "scan_cap": scan_cap, "need_train": need_train,
                "eta": args.eta, "group": args.group,
                "n_scanned": len(per_seed_nsucc), "nsucc_hist": {str(k): v for k, v in sorted(hist.items())},
                "train_seeds": train_seeds,
                "per_seed_nsucc": {str(k): v for k, v in per_seed_nsucc.items()},
            }, indent=2))
            print(f"[band-prefilter] found {len(train_seeds)}/{need_train} band seeds "
                  f"(band={band}, scanned {len(per_seed_nsucc)}/{scan_cap}, hist={dict(sorted(hist.items()))})",
                  flush=True)
        if not train_seeds:
            raise RuntimeError(f"band prefilter found 0 seeds in band {band} "
                               f"(scan_cap={scan_cap}); widen the band or raise scan-cap.")
        log["band_prefilter"] = {"band": list(band), "train_seeds": train_seeds,
                                 "scan_cap": scan_cap, "need_train": need_train}
    else:
        train_seeds = None  # non-entry mode: train uses seed_base+it inline

    # ---- EVAL(before) ----
    if not my.skip_eval:
        t0 = time.time()
        before = run_eval(eval_factory, eval_seeds, "eval_before", "before_")
        before["seconds"] = round(time.time() - t0, 1)
        log["phases"]["eval_before"] = before
        (out / "eval_before.json").write_text(json.dumps(before, indent=2))
        print("EVAL(before):", _rate_str(before), flush=True)

    # ---- TRAIN ----
    # Build the per-iter seed schedule. In band mode, draw from the band-filtered pool
    # WITHOUT replacement (reshuffling when exhausted) using the reproducible --seed RNG,
    # so a pool larger than iters gives every step a different mid-difficulty seed and no
    # single seed is overfit. entry_mode keeps its cycle; legacy keeps seed_base+it.
    band_mode = (not entry_mode) and my.train_difficulty_band is not None and bool(train_seeds)
    # Online adaptive curriculum (operator redesign t_2907de4f): NO upfront scan, seed
    # chosen per-iter from a moving band driven by the running per-seed difficulty EMA.
    adaptive_mode = (not entry_mode) and (my.adaptive_band is not None) and (not band_mode)
    batch_mode = my.groups_per_update > 1 or my.min_trainable_chunks > 0
    if batch_mode and (band_mode or entry_mode):
        raise ValueError("multi-group update batching supports raw or adaptive GRASP seeds; "
                         "static difficulty-band and entry-state modes still require a "
                         "per-group scheduler")
    curriculum = None
    cache = None
    if adaptive_mode:
        universe = my.adaptive_universe or (my.iters * 8)
        if batch_mode and universe < my.max_groups_per_update:
            raise ValueError("adaptive universe must be >= max_groups_per_update so every "
                             "group in one update can use a distinct env seed")
        curriculum = AdaptiveCurriculum(
            seed_base=my.seed_base, universe=universe, band=tuple(my.adaptive_band),
            group=my.group, explore_frac=my.adaptive_explore_frac, ema=my.adaptive_ema,
            rng_seed=my.seed, avoid_recent=my.adaptive_avoid_recent)
        cache = out / "seed_difficulty.json"
        if cache.exists():
            try:
                curriculum.load_state(json.loads(cache.read_text()))
                print(f"[adaptive] resumed seed_difficulty.json: "
                      f"{len(curriculum.difficulty)} seeds known", flush=True)
            except Exception as _e:
                print(f"[adaptive] cache load failed (fresh start): {_e}", flush=True)
        log["adaptive_curriculum"] = {
            "band": list(my.adaptive_band), "universe": universe,
            "explore_frac": my.adaptive_explore_frac, "ema": my.adaptive_ema,
            "avoid_recent": my.adaptive_avoid_recent, "rng_seed": my.seed}
        print(f"[train] adaptive moving-band mode: band={tuple(my.adaptive_band)} "
              f"universe={universe} explore_frac={my.adaptive_explore_frac} "
              f"ema={my.adaptive_ema} (rng seed {my.seed})", flush=True)
    iter_seeds = None
    if band_mode:
        import random as _random
        rng = _random.Random(my.seed)
        pool = list(train_seeds)
        order = []
        while len(order) < my.iters:
            shuffled = pool[:]
            rng.shuffle(shuffled)
            order.extend(shuffled)
        iter_seeds = order[:my.iters]
        log["train_seed_schedule"] = iter_seeds
        print(f"[train] band mode: pool={len(pool)} seeds, drawing {my.iters} "
              f"without-replacement (rng seed {my.seed}); first 8={iter_seeds[:8]}", flush=True)
    curve = []
    with open(out / "train_log.jsonl", "a") as fh:
        for it in range(my.iters):
            # In entry_mode with a prefiltered pool, iterate over grasp-entry seeds
            # (cycled if fewer were found than iters) so every train step gets a real
            # entry-state instead of being SKIPPED. In adaptive mode, ASK the curriculum
            # (moving band). In band mode, draw the precomputed without-replacement
            # schedule. Otherwise seed_base+it as before.
            adaptive_source = None
            if adaptive_mode and not batch_mode:
                seed, adaptive_source = curriculum.next_seed(it)
            elif band_mode:
                seed = iter_seeds[it]
            elif train_seeds:
                seed = train_seeds[it % len(train_seeds)]
            else:
                seed = my.seed_base + it
            if batch_mode:
                m = train_batched_update(
                    client, args, reward_cfg, my.seed_base, it,
                    curriculum=curriculum if adaptive_mode else None,
                    curriculum_cache=cache if adaptive_mode else None)
                seed = m["group_env_seeds"][0]
            else:
                m = train_iteration(client, args, reward_cfg, seed, it)
            m["iter"] = it
            m["seed"] = seed
            if adaptive_mode and not batch_mode:
                # Fold this iter's group n_succ into the seed's difficulty EMA so the band
                # moves with the policy; persist the cache each iter for resume + audit.
                n_succ = int(m.get("n_success_group") or 0)
                curriculum.update(seed, n_succ, it=it, source=adaptive_source)
                bstats = curriculum.band_stats()
                m["adaptive_source"] = adaptive_source
                m["adaptive_band_stats"] = bstats
                try:
                    cache.write_text(json.dumps(curriculum.to_dict(), indent=2))
                except Exception:
                    pass
            elif adaptive_mode:
                m["adaptive_source"] = [
                    group.get("seed_source") for group in m["group_summaries"]]
                m["adaptive_band_stats"] = curriculum.band_stats()
            curve.append({"iter": it, "loss": m["loss"], "mean_return": m["mean_return"],
                          "grad_norm": m["grad_norm"], "mean_ratio": m["mean_ratio"],
                          "n_success_group": m.get("n_success_group"),
                          "group_composition": m.get("group_composition"),
                          "advantage_mode": m.get("advantage_mode"),
                          "gated": m.get("gated"),
                          "adaptive_source": m.get("adaptive_source"),
                          "adaptive_band_stats": m.get("adaptive_band_stats"),
                          "post_step_mean_ratio": m.get("post_step_mean_ratio"),
                          "post_step_mean_kl": m.get("post_step_mean_kl"),
                          "post_step_clip_fraction": m.get("post_step_clip_fraction"),
                          "post_step_ess": m.get("post_step_ess"),
                          "adapter_delta_l2": m.get("adapter_delta_l2"),
                          "skipped": m.get("skipped")})
            fh.write(json.dumps(m) + "\n"); fh.flush()
            _asrc = f" src={m['adaptive_source']}" if adaptive_mode else ""
            if m.get("skipped"):
                print(f"[train] it={it} seed={seed}{_asrc} SKIPPED ({m['skipped']}) — no entry-state this seed", flush=True)
            elif m.get("gated"):
                print(f"[train] it={it} seed={seed}{_asrc} GATED ({m['gated']}) comp={m.get('group_composition')} "
                      f"n_succ={m.get('n_success_group')}/{args.group} std={m.get('reward_std', 0.0):.4f} "
                      f"— no optimizer step", flush=True)
            else:
                def _f(x, d=3):
                    return f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x)
                print(f"[train] it={it} seed={seed}{_asrc} comp={m.get('group_composition')}({m.get('advantage_mode')}) "
                      f"n_succ={m.get('n_success_group')}/{m.get('trajectories_collected', args.group)} "
                      f"loss={m['loss']:.4f} return={m['mean_return']:.3f} "
                      f"grad_norm={m['grad_norm']:.1f} post_ratio={_f(m.get('post_step_mean_ratio'))} "
                      f"post_kl={_f(m.get('post_step_mean_kl'),5)} post_clip={_f(m.get('post_step_clip_fraction'))} "
                      f"adapter_dL2={_f(m.get('adapter_delta_l2'),5)} "
                      f"n_succ_hold={m.get('n_success_hold')} groups={m.get('groups_collected', 1)} "
                      f"chunks={m.get('trainable_chunks_collected', m.get('n_chunks'))} "
                      f"collect_s={m.get('collect_seconds')} opt_s={m.get('optimizer_seconds')} "
                      f"mem={m.get('peak_mem_gb')}", flush=True)
    log["phases"]["train_curve"] = curve
    if adaptive_mode and curriculum is not None:
        log["adaptive_curriculum"]["final_band_stats"] = curriculum.band_stats()
        log["adaptive_curriculum"]["difficulty"] = {
            str(k): v for k, v in curriculum.difficulty.items()}

    # Checkpoint path must be on the TRAINER SERVER's filesystem, which has /train mounted
    # (= /home/v4/rl-train-t_3ed65912 on the host). /out is only visible to the CLIENT.
    ckpt_server_path = f"/train/results/{Path(my.out).name}/{my.ckpt_name}"
    save_res = client.save(
        ckpt_server_path, update_index=my.iters,
        train_meta={"skill": my.train_skill, "updates": my.iters,
                    "reward_variant": my.reward_variant})
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
