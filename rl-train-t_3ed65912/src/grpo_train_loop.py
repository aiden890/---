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
from reward import RewardConfig, RewardManager, official_success  # noqa: E402
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

    def infer(self, states, images, instruction, *, eta, traj_id=None, seed=None, skill=None):
        inputs = self._build_inputs(states, images, instruction)
        req = {"op": "sample", "inputs": inputs, "eta": eta, "traj_id": traj_id, "seed": seed, "skill": skill}
        resp = self._rpc(req)
        actions = resp["actions"]
        decoded = self.processor.decode_action(actions, robot_type=self.robot_type)
        decoded = decoded[0, :, : self.ACTION_DIM]
        decoded = decoded.float().cpu().numpy() if hasattr(decoded, "float") else np.asarray(decoded)
        return np.asarray(decoded, dtype=np.float32), resp.get("logprob")

    def update(self, advantages, clip=0.1, kl_coef=0.005, ratio_max=10.0, adv_clip=3.0):
        return self._rpc({"op": "update", "advantages": advantages, "clip": clip,
                          "kl_coef": kl_coef, "ratio_max": ratio_max, "adv_clip": adv_clip})

    def save(self, path):
        return self._rpc({"op": "save", "path": path})

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
                   frames=None, save_video=None, approach_coef=0.0, timeout_penalty=0.0):
    """Run ONE skill's VLA loop; return (obs, outcome, reward, steps, predicates).

    Training-only dense shaping (not used in eval; both default to 0.0):
      approach_coef: reward += approach_coef * (prev_dist - curr_dist) per step toward lid.
        Creates return variance even when all members fail to grasp, so GRPO has signal.
      timeout_penalty: flat penalty added when outcome==TIMEOUT (skill horizon exceeded).
        Ensures failed trajectories are numerically distinguished from successful ones.
    Neither changes the official eval reward which uses these at their defaults (0.0).
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
    p = sim.predicates()
    prev_eef_lid_dist = float(p.get("eef_lid_dist", 0.0)) if approach_coef > 0.0 else None
    while steps < horizon:
        if not action_plan:
            states = rollout.sample_history(state_queue, args.obs_history, args.obs_interval)
            images = {k: rollout.sample_history(q, args.obs_history, args.obs_interval)
                      for k, q in image_queues.items()}
            chunk, _ = client.infer(states, images, instruction, eta=eta, traj_id=traj_id,
                                    seed=seed, skill=skill_key)
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
        # Approach shaping: reward getting closer to the lid (training-only)
        if approach_coef > 0.0 and prev_eef_lid_dist is not None:
            curr_dist = float(p.get("eef_lid_dist", prev_eef_lid_dist))
            reward_sum += approach_coef * (prev_eef_lid_dist - curr_dist)
            prev_eef_lid_dist = curr_dist
        outcome = monitor.update(p, steps, horizon)
        if frames is not None and (steps % args.video_stride == 0 or outcome or done or trunc):
            frames.append(rollout.make_video_frame(obs))
        if outcome or done or trunc:
            break
    if outcome is None:
        outcome = SkillOutcome.TIMEOUT
        reward_sum += (-timeout_penalty)   # flat cost for skill timeout (training-only)
    if save_video is not None and frames is not None:
        import imageio.v2 as imageio
        imageio.mimsave(save_video, frames, fps=args.video_fps)
    return obs, outcome, reward_sum, steps, p


def eval_episode(sim, client, args, reward_cfg, seed, out_dir=None, tag=""):
    """Deterministic full-plan rollout; returns official + grasp-stage success."""
    obs, _ = rollout.reset_env(sim.genv, seed)
    sim.rest_lid_pos = sim.lid_pos()
    planner = OraclePlanner(max_skill_calls=args.max_skill_calls)
    reward_mgr = RewardManager(reward_cfg)
    call = planner.initial()
    guard = 0
    grasp_ok = False
    total_reward = 0.0
    save_dir = None
    if out_dir is not None:
        save_dir = Path(out_dir); save_dir.mkdir(parents=True, exist_ok=True)
    while call is not None and guard < args.max_skill_calls:
        guard += 1
        frames = [] if (save_dir is not None and guard <= 3) else None
        vid = (save_dir / f"{tag}seed{seed}_call{guard}_{SKILL_KEY[call.skill]}.mp4") if frames is not None else None
        obs, outcome, r, steps, p = _run_one_skill(
            sim, client, obs, args, call.skill, reward_mgr, eta=0.0, traj_id=None,
            seed=None, frames=frames, save_video=str(vid) if vid else None)
        total_reward += r
        if call.skill is Skill.GRASP and outcome is SkillOutcome.SUCCESS:
            grasp_ok = True
        call = planner.propose(call, outcome, p)
    success = bool(sim.predicates().get("official_check_success"))
    return {"seed": seed, "official_success": success, "grasp_success": grasp_ok,
            "total_reward": round(total_reward, 4)}


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
    returns = []
    traj_ids = []
    for m in range(args.group):
        genv, sim = _make_env(args.split, seed)
        try:
            obs, _ = rollout.reset_env(genv, seed)   # identical start for all members
            sim.rest_lid_pos = sim.lid_pos()
            reward_mgr = RewardManager(reward_cfg)
            traj_id = f"iter{it}_m{m}"
            _, outcome, r, steps, p = _run_one_skill(
                sim, client, obs, args, skill, reward_mgr, eta=args.eta,
                traj_id=traj_id, seed=args.seed_base + it * 100 + m,
                approach_coef=train_approach_coef,
                timeout_penalty=train_timeout_penalty)
            # terminal shaping for the skill unit: bonus if the skill's predicate is met
            if outcome is SkillOutcome.SUCCESS:
                r += 1.0
            returns.append(r)
            traj_ids.append(traj_id)
        finally:
            genv.close()
    arr = np.asarray(returns, dtype=np.float64)
    adv = (arr - arr.mean()) / (arr.std() + 1e-8)
    advantages = {tid: float(a) for tid, a in zip(traj_ids, adv)}
    metrics = client.update(advantages, clip=args.clip, kl_coef=args.kl_coef,
                            ratio_max=args.ratio_max, adv_clip=args.adv_clip)
    metrics["mean_return"] = float(arr.mean())
    metrics["max_return"] = float(arr.max())
    metrics["returns"] = [round(x, 4) for x in returns]
    metrics["reward_std"] = float(arr.std())
    return metrics


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
    off = sum(r["official_success"] for r in results) / len(results)
    grasp = sum(r["grasp_success"] for r in results) / len(results)
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
    ap.add_argument("--horizon-grasp", type=int, default=120)
    ap.add_argument("--horizon-move", type=int, default=120)
    ap.add_argument("--horizon-place", type=int, default=150)
    ap.add_argument("--max-skill-calls", type=int, default=3)
    ap.add_argument("--save-videos", type=int, default=6)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--ckpt-name", default="grpo_trained.pt")
    ap.add_argument("--load-ckpt", default=None,
                    help="Load a saved LoRA checkpoint before AFTER-eval phase. "
                         "For standalone post-hoc eval of a saved ARM checkpoint.")
    return ap


def main():
    my, rest = build_parser().parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    for k in ("group", "eta", "clip", "kl_coef", "ratio_max", "adv_clip", "train_skill",
              "horizon_grasp", "horizon_move", "horizon_place",
              "max_skill_calls", "seed_base", "save_videos", "split"):
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

    log = {"config": {**vars(my), "reward_variant": my.reward_variant}, "phases": {}}
    (out / "train_log.jsonl").write_text("")

    eval_seeds = [my.eval_seed_base + i for i in range(my.eval_n)]
    heldout_seeds = [my.heldout_seed_base + i for i in range(my.heldout_n)]

    # ---- EVAL(before) ----
    if not my.skip_eval:
        t0 = time.time()
        before = eval_pool(eval_factory, client, args, reward_cfg, eval_seeds, out / "eval_before", "before_")
        before["seconds"] = round(time.time() - t0, 1)
        log["phases"]["eval_before"] = before
        (out / "eval_before.json").write_text(json.dumps(before, indent=2))
        print("EVAL(before):", before["official_success_rate"], before["grasp_success_rate"], flush=True)

    # ---- TRAIN ----
    curve = []
    with open(out / "train_log.jsonl", "a") as fh:
        for it in range(my.iters):
            seed = my.seed_base + it
            m = train_iteration(client, args, reward_cfg, seed, it)
            m["iter"] = it
            m["seed"] = seed
            curve.append({"iter": it, "loss": m["loss"], "mean_return": m["mean_return"],
                          "grad_norm": m["grad_norm"], "mean_ratio": m["mean_ratio"]})
            fh.write(json.dumps(m) + "\n"); fh.flush()
            print(f"[train] it={it} loss={m['loss']:.4f} mean_return={m['mean_return']:.3f} "
                  f"grad_norm={m['grad_norm']:.3f} ratio={m['mean_ratio']:.3f} mem={m.get('peak_mem_gb')}", flush=True)
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
        after = eval_pool(eval_factory, client, args, reward_cfg, eval_seeds, out / "eval_after", "after_")
        after["seconds"] = round(time.time() - t0, 1)
        log["phases"]["eval_after"] = after
        (out / "eval_after.json").write_text(json.dumps(after, indent=2))
        print("EVAL(after):", after["official_success_rate"], after["grasp_success_rate"], flush=True)

        # ---- HELDOUT (disjoint geometry/seed) ----
        t0 = time.time()
        heldout = eval_pool(heldout_factory, client, args, reward_cfg, heldout_seeds, out / "eval_heldout", "heldout_")
        heldout["seconds"] = round(time.time() - t0, 1)
        log["phases"]["eval_heldout"] = heldout
        (out / "eval_heldout.json").write_text(json.dumps(heldout, indent=2))

    (out / "run_summary.json").write_text(json.dumps(log, indent=2, default=str))
    client.close()
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
