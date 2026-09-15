"""Skill-conditioned rollout driver: baseline (deterministic) and RL (flow-SDE) rollouts.

This is the environment's top-level entry. It runs INSIDE the xiaomi-client container
(RoboCasa + gym + the reusable /work/rollout.py harness) and talks to an IN-PROCESS
model when RL log-probs are needed, or to the socket server for pure baseline rollouts.

Two modes:
  * baseline : reuse the socket EvalClient (unchanged /work/rollout.py) exactly as
               t_4a072806 did -- deterministic Euler sampler, fixed seed. This is the
               apples-to-apples comparison point and needs no model in-process.
  * rl       : load the model in-process, sample action chunks with the flow-SDE
               (eta>0), and STORE per-transition log-probs, the branch state (MuJoCo
               qpos/qvel + controller + RNG + obs history + current skill), and the
               executed-action mask for each replan. No GRPO update is performed here
               (that is the successor training card); this card proves the data needed
               for GRPO can be produced and is correct.

Both modes drive the 3-skill plan through the skill manager (oracle planner for
end-to-end validation; VLM planner is selectable and scored separately) and aggregate
per-skill success/failure-reason/chunk-count plus reward breakdowns.

Reuse policy: geometry/predicate/snapshot logic is imported from the parent's
skill_eval.py (single source of truth). This driver adds only the RL sampling seam,
the skill-manager loop, reward accounting, and per-branch state capture.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

# reusable harness + predicates (single source of truth, imported not copied)
sys.path.insert(0, "/work")                       # xiaomi-cu121 rollout.py
sys.path.insert(0, "/skill_eval_tools")           # parent's skill_eval.py (mounted separately)
import rollout  # noqa: E402
import skill_eval  # noqa: E402  (Sim, run_episode, hold_action, Cond, SKILLS, thresholds)

# this package
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from reward import RewardConfig, RewardManager  # noqa: E402
from skill_manager import (  # noqa: E402
    MonitorConfig,
    OraclePlanner,
    Skill,
    SkillEpisodeResult,
    SkillMonitor,
    SKILL_INSTRUCTION,
    VLMPlannerStub,
)

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
from robocasa.utils.env_utils import convert_action  # noqa: E402

SKILL_ENUM = {"grasp": Skill.GRASP, "move_holding": Skill.MOVE_HOLDING, "place": Skill.PLACE}
SKILL_KEY = {v: k for k, v in SKILL_ENUM.items()}


def capture_branch_state(sim, obs_queues, state_queue, current_skill, rng_state=None):
    """Snapshot everything needed to *branch* a shared prefix for group rollouts.

    Captures MuJoCo qpos/qvel + sim time, controller/gripper state (via skill_eval's
    snapshot), the observation history queues, the current skill, and (optionally) the
    torch RNG state. This is what the successor GRPO card restores to launch `group`
    parallel continuations from the same start.
    """
    snap = sim.snapshot(f"branch_{SKILL_KEY.get(current_skill, current_skill)}", 0)
    return {
        "mujoco": {"qpos": snap["qpos"], "qvel": snap["qvel"], "time": snap["time"],
                   "gripper_current_action": snap["gripper_current_action"], "blender": snap["blender"]},
        "obs_history": {k: [f.copy() for f in q] for k, q in obs_queues.items()},
        "state_history": [s.copy() for s in state_queue],
        "current_skill": SKILL_KEY.get(current_skill, str(current_skill)),
        "rng_state": rng_state,
        "predicates": snap["predicates"],
    }


def run_skill_episode(sim, client, args, planner, reward_cfg, out_dir, seed, *, capture_branches=False):
    """Drive the full 3-skill plan with recovery; aggregate rewards + per-skill stats.

    Uses the reusable skill_eval.run_episode for the inner VLA loop of each skill, so
    the chunk/replan/history handling is identical to the validated rollout code.
    """
    ep = SkillEpisodeResult(planner=planner.name)
    reward_mgr = RewardManager(reward_cfg)
    branch_states = []

    obs, _ = rollout.reset_env(sim.genv, seed)
    sim.rest_lid_pos = sim.lid_pos()
    monitor_cfg = MonitorConfig(grasp_hold_steps=skill_eval.GRASP_HOLD_STEPS,
                                move_hold_steps=skill_eval.MOVE_HOLD_STEPS)
    horizons = {Skill.GRASP: args.horizon_grasp, Skill.MOVE_HOLDING: args.horizon_move, Skill.PLACE: args.horizon_place}

    call = planner.initial()
    total_reward = 0.0
    guard = 0
    while call is not None and guard < args.max_skill_calls:
        guard += 1
        skill_key = SKILL_KEY[call.skill]
        instruction = SKILL_INSTRUCTION[call.skill]
        monitor = SkillMonitor(call.skill, monitor_cfg)
        horizon = horizons[call.skill]

        # --- inner VLA loop for this skill (reuses validated harness) ---
        queue_length = (args.obs_history - 1) * args.obs_interval + 1
        image_queues = {k: collections.deque(maxlen=queue_length) for k in rollout.CAMERA_KEYS}
        state_queue = collections.deque(maxlen=queue_length)
        for k, im in rollout.collect_images(obs).items():
            image_queues[k].append(im)
        state_queue.append(rollout.observation_to_state(obs))
        action_plan = collections.deque()
        frames = [rollout.make_video_frame(obs)]
        steps = chunks = 0
        outcome = None
        log_path = out_dir / f"{skill_key}_call{guard}_steps.jsonl"
        with open(log_path, "w") as log:
            log.write(json.dumps({"type": "skill_start", "skill": skill_key, "call": guard,
                                  "instruction": instruction, "eta": args.eta, "mode": args.mode}) + "\n")
            if capture_branches:
                branch_states.append(capture_branch_state(sim, image_queues, state_queue, call.skill))
            while steps < horizon:
                if not action_plan:
                    states = rollout.sample_history(state_queue, args.obs_history, args.obs_interval)
                    images = {k: rollout.sample_history(q, args.obs_history, args.obs_interval)
                              for k, q in image_queues.items()}
                    chunk, lp = client.infer_with_logprob(states, images, instruction)
                    chunks += 1
                    if len(chunk) < args.replan_steps:
                        raise RuntimeError(f"chunk {len(chunk)} < replan {args.replan_steps}")
                    action_plan.extend(chunk[: args.replan_steps])
                    log.write(json.dumps({"type": "chunk", "step": steps, "chunk_len": int(len(chunk)),
                                          "executed": int(args.replan_steps),
                                          "executed_logprob": (float(lp) if lp is not None else None)}) + "\n")
                a = np.asarray(action_plan.popleft(), dtype=np.float32)
                obs, _, done, trunc, info = sim.genv.step(convert_action(a))
                steps += 1
                for k, im in rollout.collect_images(obs).items():
                    image_queues[k].append(im)
                state_queue.append(rollout.observation_to_state(obs))
                p = sim.predicates()
                rb = reward_mgr.step_reward(steps, p, done=bool(done), truncated=bool(trunc))
                total_reward += rb.total(reward_cfg.vlm_weight)
                outcome = monitor.update(p, steps, horizon)
                if steps % args.video_stride == 0 or outcome or done or trunc:
                    frames.append(rollout.make_video_frame(obs))
                log.write(json.dumps({"type": "step", "step": steps, "predicates": p,
                                      "reward": rb.__dict__, "outcome": (outcome.value if outcome else None)}) + "\n")
                if outcome or done or trunc:
                    break
        if outcome is None:
            from skill_manager import SkillOutcome
            outcome = SkillOutcome.TIMEOUT
        # save skill video
        import imageio.v2 as imageio
        imageio.mimsave(out_dir / f"{skill_key}_call{guard}.mp4", frames, fps=args.video_fps)
        ep.add(call, outcome, steps, chunks)
        call = planner.propose(call, outcome, sim.predicates())

    ep.success = bool(sim.predicates().get("official_check_success"))
    ep.total_steps = ep.total_steps
    result = {
        "planner": planner.name, "mode": args.mode, "eta": args.eta, "seed": seed,
        "success": ep.success, "total_steps": ep.total_steps, "total_reward": total_reward,
        "skill_calls": ep.calls,
        "per_skill": _aggregate_per_skill(ep.calls),
    }
    if capture_branches:
        np.savez_compressed(out_dir / "branch_states.npz",
                            **{f"branch_{i}_qpos": b["mujoco"]["qpos"] for i, b in enumerate(branch_states)})
        result["num_branch_states"] = len(branch_states)
    return result


def _aggregate_per_skill(calls):
    agg = {}
    for c in calls:
        s = c["skill"]
        a = agg.setdefault(s, {"attempts": 0, "successes": 0, "chunks": 0, "steps": 0, "failure_reasons": {}})
        a["attempts"] += 1
        a["chunks"] += c["chunks"]
        a["steps"] += c["steps"]
        if c["outcome"] == "success":
            a["successes"] += 1
        else:
            a["failure_reasons"][c["outcome"]] = a["failure_reasons"].get(c["outcome"], 0) + 1
    for a in agg.values():
        a["success_rate"] = a["successes"] / a["attempts"] if a["attempts"] else None
    return agg


# --------------------------------------------------------------------------- #
# Clients: baseline (socket, no log-prob) and RL (in-process model, log-prob).
# --------------------------------------------------------------------------- #
class BaselineClient:
    """Socket EvalClient wrapper; deterministic sampler, no log-prob (returns None)."""

    def __init__(self, model_path, host, port, robot_type, crop):
        self.c = rollout.EvalClient(model_path, host, port, robot_type, crop)

    def infer_with_logprob(self, states, images, instruction):
        return self.c.infer(states, images, instruction), None

    def close(self):
        self.c.close()


class RLFlowClient:
    """Socket client to the RL flow-SDE server (src/rl_server.py).

    The RoboCasa client image has only CPU torch, so the model runs on the GPU-side RL
    server. This client builds the same processor request the baseline EvalClient
    builds, appends an "rl" block (eta, num_steps, replan, real_action_dim), sends it
    over the length-prefixed socket, and receives {actions, executed_logprob, ...}. It
    decodes the raw actions with the processor exactly like the baseline path, so the
    action semantics on the simulator are unchanged; only the sampler (deterministic
    Euler -> flow-SDE) and the returned log-prob differ.
    """

    def __init__(self, model_path, host, port, robot_type, crop, num_steps, eta,
                 replan_steps, real_action_dim, seed=None):
        import torch
        self.torch = torch
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.robot_type = robot_type
        self.crop = crop
        self.num_steps = num_steps
        self.eta = eta
        self.replan_steps = replan_steps
        self.real_action_dim = real_action_dim
        self.seed = seed
        self.STATE_DIM = rollout.STATE_DIM
        self.ACTION_DIM = rollout.ACTION_DIM
        self.host, self.port = host, port
        self._connect()

    def _connect(self):
        import socket
        import time
        for _ in range(600):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                return
            except OSError:
                time.sleep(1)
        raise ConnectionError(f"cannot reach RL server {self.host}:{self.port}")

    def _rpc(self, request):
        import pickle
        import struct
        blob = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(blob)) + blob)
        ln = self.sock.recv(4)
        n = struct.unpack(">I", ln)[0]
        data = b""
        while len(data) < n:
            data += self.sock.recv(n - len(data))
        return pickle.loads(data)

    def infer_with_logprob(self, states, images, instruction):
        import numpy as np
        torch = self.torch
        state_history = np.asarray(states, dtype=np.float32)
        state = np.zeros((1, state_history.shape[0], self.STATE_DIM), dtype=np.float32)
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
        request = dict(inputs)
        request["task_id"] = self.robot_type
        request["rl"] = {"eta": self.eta, "num_steps": self.num_steps,
                         "replan_steps": self.replan_steps, "real_action_dim": self.real_action_dim,
                         "seed": self.seed}
        resp = self._rpc(request)
        actions = resp["actions"]
        decoded = self.processor.decode_action(actions, robot_type=self.robot_type)
        decoded = decoded[0, :, : self.ACTION_DIM]
        decoded = decoded.float().cpu().numpy() if hasattr(decoded, "float") else np.asarray(decoded)
        lp = resp["executed_logprob"]
        logprob = float(lp[0]) if lp is not None else None
        return np.asarray(decoded, dtype=np.float32), logprob

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("baseline", "rl"), default="baseline")
    ap.add_argument("--planner", choices=("oracle", "vlm"), default="oracle")
    ap.add_argument("--reward-mode", choices=("simulator", "simulator+vlm"), default="simulator")
    ap.add_argument("--vlm-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--horizon-grasp", type=int, default=300)
    ap.add_argument("--horizon-move", type=int, default=300)
    ap.add_argument("--horizon-place", type=int, default=400)
    ap.add_argument("--max-skill-calls", type=int, default=8)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=10086)
    ap.add_argument("--rl-server-port", type=int, default=10087)
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--capture-branches", action="store_true")
    return ap


def main():
    my, rest = build_parser().parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    for k in ("mode", "planner", "seed", "eta", "num_steps", "horizon_grasp", "horizon_move",
              "horizon_place", "max_skill_calls", "model_path"):
        setattr(args, k, getattr(my, k))
    out = Path(my.out)
    out.mkdir(parents=True, exist_ok=True)

    reward_cfg = RewardConfig(mode=my.reward_mode, vlm_weight=my.vlm_weight,
                              horizon=my.horizon_place)
    planner = OraclePlanner(max_skill_calls=my.max_skill_calls) if my.planner == "oracle" \
        else VLMPlannerStub(max_skill_calls=my.max_skill_calls)

    genv = gym.make("robocasa/CloseBlenderLid", split=args.split, seed=my.seed)
    sim = skill_eval.Sim(genv)
    if my.mode == "baseline":
        client = BaselineClient(my.model_path, my.server_addr, my.server_port, args.robot_type, args.crop_ratio)
    else:
        client = RLFlowClient(my.model_path, my.server_addr, my.rl_server_port, args.robot_type,
                              args.crop_ratio, my.num_steps, my.eta, args.replan_steps,
                              rollout.ACTION_DIM, seed=my.seed)
    try:
        result = run_skill_episode(sim, client, args, planner, reward_cfg, out, my.seed,
                                   capture_branches=my.capture_branches)
        (out / "episode_result.json").write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps({"success": result["success"], "mode": my.mode, "eta": my.eta,
                          "per_skill": result["per_skill"]}, indent=2))
    finally:
        genv.close()
        client.close()


if __name__ == "__main__":
    main()
