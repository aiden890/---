"""Instruction-attendance probe for the Xiaomi-Robotics-1 RoboCasa365 checkpoint (t_8cd8d8c0).

Question: does the policy actually ATTEND to the language instruction, or does it
condition only on the observation and ignore the instruction?

Method (from robocasa-vla-rl-experiments skill):
  Hold the observation FIXED (one identical initial MuJoCo state) and vary ONLY the
  instruction text, then sample actions for each instruction. Because the Xiaomi
  checkpoint draws its flow x0 from the GLOBAL unseeded RNG, identical inputs give
  slightly different action chunks -> there is a non-zero SAMPLING-NOISE FLOOR. So we
  measure, per start state:
    within(I)  = mean pairwise L2 between N samples of the SAME instruction  (noise floor)
    between(A,B) = L2 between the mean action chunk of instruction A vs B
  Instruction-conditioned  <=>  between(correct, other) >> within  (ratio >> 1).
  Instruction-ignored (obs-only) <=> between ~ within (ratio ~ 1) for ALL instructions.

Start states (fixed obs), reusing skill_eval.Sim snapshot/restore verbatim:
  reset : official CloseBlenderLid reset(seed) -- lid on counter, gripper far (grasp start).
  move  : generation snapshot -- lid grasped + lifted, far from blender (move_holding start).
  place : generation snapshot -- grasped lid in pre-place region above blender (place start).

No long rollouts are needed for the headline test: we call EvalClient.infer() repeatedly
on the SAME fixed obs. An optional short open-loop rollout-divergence pass (--rollout-steps)
additionally shows how far the arm drifts apart under different instructions.

Reuses /work/rollout.py (EvalClient, queues, convert_action) and /skilltools/skill_eval.py
(Sim, FULL_INSTRUCTION, SKILLS, generation snapshotting) by import -- no duplicated policy code.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, "/skilltools")
import rollout  # noqa: E402  (unchanged /work/rollout.py)
import skill_eval  # noqa: E402  (unchanged t_4a072806 tools/skill_eval.py)
from skill_eval import Sim, FULL_INSTRUCTION, SKILLS, hold_action, Cond  # noqa: E402

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
from robocasa.utils.env_utils import convert_action  # noqa: E402


def f(x):
    return [float(v) for v in np.asarray(x).reshape(-1)]


# The instruction battery. category: correct | skill | opposite | irrelevant | degenerate
def build_battery():
    return [
        ("correct_full", FULL_INSTRUCTION, "correct"),
        ("skill_grasp", SKILLS["grasp"], "skill"),
        ("skill_move", SKILLS["move_holding"], "skill"),
        ("skill_place", SKILLS["place"], "skill"),
        ("opposite_open_drawer", "Open the drawer.", "opposite"),
        ("irrelevant_pick_cup", "Pick up the cup.", "irrelevant"),
        ("irrelevant_turn_on_stove", "Turn on the stove.", "irrelevant"),
        ("empty", "", "degenerate"),
        ("nonsense", "asdf qwer zxcv lorem ipsum.", "degenerate"),
    ]


def make_queues(obs, obs_history, obs_interval):
    queue_length = (obs_history - 1) * obs_interval + 1
    image_queues = {key: collections.deque(maxlen=queue_length) for key in rollout.CAMERA_KEYS}
    state_queue = collections.deque(maxlen=queue_length)
    # fill the whole history window with the single fixed obs (static history)
    for _ in range(queue_length):
        for key, image in rollout.collect_images(obs).items():
            image_queues[key].append(image)
        state_queue.append(rollout.observation_to_state(obs))
    return image_queues, state_queue


def infer_once(client, state_queue, image_queues, args, instruction):
    states = rollout.sample_history(state_queue, args.obs_history, args.obs_interval)
    images = {key: rollout.sample_history(q, args.obs_history, args.obs_interval)
              for key, q in image_queues.items()}
    return np.asarray(client.infer(states, images, instruction), dtype=np.float32)  # [chunk, 12]


def chunk_dist(a, b, k):
    """L2 over the first k action steps (flattened) between two chunks."""
    a = np.asarray(a)[:k].reshape(-1)
    b = np.asarray(b)[:k].reshape(-1)
    return float(np.linalg.norm(a - b))


def cosine(a, b, k):
    a = np.asarray(a)[:k].reshape(-1)
    b = np.asarray(b)[:k].reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def probe_state(client, sim, args, obs, state_tag, out):
    """Fixed-obs action probe: for each instruction, N samples; compute noise floor + between-instr."""
    battery = build_battery()
    obs_history, obs_interval = args.obs_history, args.obs_interval
    # samples[label] = list of action chunks [N][chunk,12]
    samples = {}
    t0 = time.time()
    for label, instr, cat in battery:
        chunks = []
        for _ in range(args.samples):
            # rebuild queues from the SAME fixed obs every time so history is identical
            image_queues, state_queue = make_queues(obs, obs_history, obs_interval)
            chunks.append(infer_once(client, state_queue, image_queues, args, instr))
        samples[label] = np.stack(chunks, axis=0)  # [N, chunk, 12]
        print(f"[{state_tag}] {label:22s} sampled N={args.samples} chunk={samples[label].shape}", flush=True)
    dt = time.time() - t0

    k = args.compare_steps  # compare over first k action steps
    # within-instruction noise floor: mean pairwise L2 across the N samples
    within = {}
    mean_chunk = {}
    for label, _, _ in battery:
        S = samples[label]
        pw = [chunk_dist(S[i], S[j], k) for i, j in itertools.combinations(range(S.shape[0]), 2)]
        within[label] = float(np.mean(pw)) if pw else 0.0
        mean_chunk[label] = S.mean(axis=0)

    ref = "correct_full"
    noise_floor = float(np.mean([within[l] for l, _, _ in battery]))
    rows = []
    for label, instr, cat in battery:
        d = chunk_dist(mean_chunk[label], mean_chunk[ref], k)
        cos = cosine(mean_chunk[label], mean_chunk[ref], k)
        # ratio of between-instruction mean-shift to the sampling noise floor
        ratio = d / noise_floor if noise_floor > 1e-9 else float("inf")
        rows.append({
            "label": label, "category": cat, "instruction": instr,
            "within_instr_L2": within[label],
            "L2_meanchunk_vs_correct": d,
            "cosine_meanchunk_vs_correct": cos,
            "ratio_between_over_noisefloor": ratio,
        })

    result = {
        "state_tag": state_tag,
        "seed": args.seed,
        "samples_per_instruction": args.samples,
        "compare_steps": k,
        "chunk_len": int(next(iter(samples.values())).shape[1]),
        "action_dim": int(next(iter(samples.values())).shape[2]),
        "sampling_noise_floor_L2": noise_floor,
        "reference_instruction": FULL_INSTRUCTION,
        "init_predicates": sim.predicates(),
        "rows": rows,
        "infer_seconds": dt,
    }
    (out / f"probe_{state_tag}.json").write_text(json.dumps(result, indent=2, default=str))
    # also dump raw per-sample first-step actions for auditing
    raw = {label: samples[label][:, 0, :].tolist() for label, _, _ in battery}
    (out / f"probe_{state_tag}_raw_firststep.json").write_text(json.dumps(raw, indent=2))
    # full per-sample chunks [N, chunk, 12] for a definitive full-chunk permutation test offline
    full = {label: samples[label].tolist() for label, _, _ in battery}
    (out / f"probe_{state_tag}_raw_full.json").write_text(json.dumps(full))
    print(f"[{state_tag}] noise_floor_L2={noise_floor:.4f}  "
          + "  ".join(f"{r['label']}={r['ratio_between_over_noisefloor']:.2f}x" for r in rows), flush=True)
    return result


def build_snapshots(client, sim, args, out):
    """Run one VLA generation episode (full instruction) to grab move/place fixed-obs snapshots.

    Mirrors skill_eval.main()'s generation loop but only to obtain snapshots.
    """
    genv = sim.genv
    snaps = {}
    for attempt in range(args.gen_attempts):
        obs, _ = rollout.reset_env(genv, args.seed)
        found = {}
        cnt_move = cnt_place = 0
        move_far_xy = min(0.12, 0.5 * sim.rest_xy_to_closed)

        def gen_cond(p):
            nonlocal cnt_move, cnt_place
            cnt_move = cnt_move + 1 if (p["lid_grasped"] and p["lid_xy_to_closed_pos"] > move_far_xy) else 0
            cnt_place = cnt_place + 1 if (p["lid_grasped"] and p["in_preplace_region"] and not p["lid_on_blender"]) else 0
            if "move" not in found and cnt_move >= skill_eval.SNAP_MOVE_STEPS:
                found["move"] = None
            if "place" not in found and cnt_place >= skill_eval.SNAP_PLACE_STEPS:
                found["place"] = None
            return "place" in found

        orig_step = genv.step
        gen_step = [0]

        def snap_step(action, _orig=orig_step):
            r = _orig(action)
            gen_step[0] += 1
            p = sim.predicates()
            gen_cond(p)
            for key in list(found):
                if found[key] is None:
                    found[key] = sim.snapshot(key, gen_step[0])
            return r
        genv.step = snap_step
        frames = []
        skill_eval.run_episode(sim, client, args, FULL_INSTRUCTION, obs, args.gen_horizon,
                               Cond(lambda p: "place" in found and found["place"] is not None, 1),
                               out / f"generation_attempt{attempt}.jsonl", frames,
                               {"kind": "generation", "attempt": attempt, "seed": args.seed})
        genv.step = orig_step
        for key in list(found):
            if found[key] is not None and key not in snaps:
                snaps[key] = found[key]
        imageio.mimsave(out / f"generation_attempt{attempt}.mp4", frames, fps=args.video_fps)
        print(f"[gen] attempt{attempt}: found={ {k: (v['gen_step'] if v else None) for k,v in found.items()} }", flush=True)
        if "move" in snaps and "place" in snaps:
            break
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--samples", type=int, default=8, help="action samples per instruction")
    ap.add_argument("--compare-steps", type=int, default=16, help="action steps used in distance")
    ap.add_argument("--states", default="reset,move,place")
    ap.add_argument("--gen-attempts", type=int, default=6)
    ap.add_argument("--gen-horizon", type=int, default=900)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=10086)
    ap.add_argument("--model-path", default="/checkpoint")
    my, rest = ap.parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    # expose the extra fields probe_state/build_snapshots read off `args`
    args.seed = my.seed
    args.samples = my.samples
    args.compare_steps = my.compare_steps
    args.gen_attempts = my.gen_attempts
    args.gen_horizon = my.gen_horizon
    out = Path(my.out)
    out.mkdir(parents=True, exist_ok=True)

    client = rollout.EvalClient(my.model_path, my.server_addr, my.server_port, args.robot_type, args.crop_ratio)
    genv = gym.make("robocasa/CloseBlenderLid", split=args.split, seed=my.seed)
    sim = Sim(genv)
    wanted = [s.strip() for s in my.states.split(",") if s.strip()]
    results = {}
    try:
        obs, _ = rollout.reset_env(genv, my.seed)
        assert obs["annotation.human.task_description"] == FULL_INSTRUCTION
        sim.rest_lid_pos = sim.lid_pos()
        sim.rest_xy_to_closed = float(np.linalg.norm(sim.rest_lid_pos[:2] - sim.closed_pos()[:2]))

        snaps = {}
        if "move" in wanted or "place" in wanted:
            snaps = build_snapshots(client, sim, args, out)

        for state_tag in wanted:
            if state_tag == "reset":
                obs, _ = rollout.reset_env(genv, my.seed)
            else:
                if state_tag not in snaps:
                    print(f"[{state_tag}] SKIPPED: generation never produced this start state", flush=True)
                    results[state_tag] = {"status": "skipped", "reason": "no snapshot"}
                    continue
                sim.restore(snaps[state_tag])
                obs, _, _, _, _ = genv.step(convert_action(hold_action(True)))  # settle
            results[state_tag] = probe_state(client, sim, args, obs, state_tag, out)
        (out / "summary.json").write_text(json.dumps({
            "seed": my.seed, "samples": my.samples, "compare_steps": my.compare_steps,
            "states": wanted, "results": {k: (v if isinstance(v, dict) and "rows" not in v else
                                              {"noise_floor": v.get("sampling_noise_floor_L2"),
                                               "rows": v.get("rows")}) for k, v in results.items()},
        }, indent=2, default=str))
    finally:
        genv.close()
        client.close()


if __name__ == "__main__":
    main()
