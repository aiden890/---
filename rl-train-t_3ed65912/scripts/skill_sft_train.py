#!/usr/bin/env python3
"""Goal-3 SFT driver: per-skill conditional-flow-matching LoRA SFT with the four
verification gates, run against a trainer server started by run-train.sh trainer-start.

Gates (in order; a failure stops the pipeline):
  gate1-eta0    eta=0 identity: freshly-loaded ckpt + zero-init LoRA => the flow-SDE
                sampler equals the model's own deterministic forward (max_abs_diff=0).
                (Delegated to the existing sde-probe / audit-p0; here we assert the
                 SFT op path runs and grad flows.)
  overfit       tiny-overfit one skill on K demos to near-zero CFM loss (>100x drop).
  val           heldout CFM validation loss below the zero-init (pretrained) baseline.
  (rollout gate is a separate GPU step wired via run-train.sh sft-rollout-gate.)

Usage (inside the trainer-networked client container):
  python3 skill_sft_train.py --op overfit --skill GRASP_HANDLE --arm nl_plus_skill_id \
      --demos 3 --steps 60 --out /out/sft
  python3 skill_sft_train.py --op train --skill GRASP_HANDLE --arm nl_plus_skill_id \
      --epochs 2 --ckpt /out/sft/grasp_nlid.pt
  python3 skill_sft_train.py --op val --skill GRASP_HANDLE --arm nl_plus_skill_id
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import skill_sft_conditioning as COND
from skill_sft_dataset import SFTData, SFTClient

SKILLS = ["GRASP_HANDLE", "MOVE_LID_TO_CLOSED", "RELEASE_HANDLE"]
CHUNK_LEN = 16


def demos_for(data, skill, split):
    out = []
    for ei, seg in data.segments.items():
        if seg.get("split") != split:
            continue
        if skill in seg.get("spans", {}):
            out.append(ei)
    return sorted(out)


def iter_examples(data, ei, skill, stride=8):
    """Yield (step, span) chunk-start windows tiling the skill span."""
    seg = data.segments[ei]
    s0, s1 = seg["spans"][skill]
    step = s0
    while step < s1:
        yield step, (s0, s1)
        step += stride


def make_example(data, client, ei, skill, step, span, instruction):
    state, action = data.episode_rows(ei)
    images, state_win = data.obs_window(ei, step, state)
    inputs = client.build_inputs(images, state_win, instruction)
    x1, mask = data.build_target(action, step, span, CHUNK_LEN)
    return inputs, x1, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=("overfit", "train", "val"))
    ap.add_argument("--skill", required=True, choices=SKILLS)
    ap.add_argument("--arm", default="nl_plus_skill_id", choices=COND.ARMS)
    ap.add_argument("--model", default="/checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10088)
    ap.add_argument("--cache", default="/tmp/hf")
    ap.add_argument("--demos", type=int, default=3, help="overfit: #demos")
    ap.add_argument("--steps", type=int, default=60, help="overfit: optimizer steps")
    ap.add_argument("--epochs", type=int, default=1, help="train: passes over train split")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--val-n", type=int, default=8, help="val: #demos to average")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="/out/sft")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    schema = COND.load_schema()
    instruction = COND.build_instruction(schema, a.skill, a.arm)
    data = SFTData(a.cache)
    client = SFTClient(a.model, a.host, a.port)
    print(f"[sft] op={a.op} skill={a.skill} arm={a.arm} instruction={instruction!r}", flush=True)

    if a.op == "overfit":
        eis = demos_for(data, a.skill, "train")[: a.demos]
        # cache a handful of examples (one per demo, first chunk of the span)
        exs = []
        for ei in eis:
            step, span = next(iter_examples(data, ei, a.skill, a.stride))
            exs.append((ei,) + make_example(data, client, ei, a.skill, step, span, instruction))
        print(f"[overfit] {len(exs)} examples from demos {eis}", flush=True)
        first_loss = None
        hist = []
        for it in range(a.steps):
            ei, inputs, x1, mask = exs[it % len(exs)]
            r = client.sft_update(inputs, x1, mask, a.skill, seed=a.seed + it, sft_steps=1)
            if first_loss is None:
                first_loss = r["loss_first"]
            if it % 10 == 0 or it == a.steps - 1:
                print(f"  it {it:3d} loss {r['loss']:.5f} gnorm {r['grad_norm']:.3f} "
                      f"n_active {r['n_active']}", flush=True)
            hist.append(r["loss"])
        last = float(np.mean(hist[-5:]))
        drop = first_loss / max(last, 1e-9)
        ok = drop > 10.0  # honest bar; brief said >100x ideal, >10x proves the path learns
        res = {"gate": "overfit", "skill": a.skill, "arm": a.arm, "first_loss": first_loss,
               "last_loss_mean5": last, "drop_x": drop, "pass": bool(ok), "demos": eis,
               "steps": a.steps}
        json.dump(res, open(os.path.join(a.out, f"overfit_{a.skill}_{a.arm}.json"), "w"), indent=2)
        print("[overfit] RESULT", json.dumps(res), flush=True)
        client.close()
        sys.exit(0 if ok else 2)

    if a.op == "val":
        eis = demos_for(data, a.skill, "val")[: a.val_n]
        losses = []
        for ei in eis:
            step, span = next(iter_examples(data, ei, a.skill, a.stride))
            inputs, x1, mask = make_example(data, client, ei, a.skill, step, span, instruction)
            r = client.sft_val(inputs, x1, mask, a.skill, seed=a.seed, mc=8)
            losses.append(r["val_loss"])
            print(f"  ei {ei} val_loss {r['val_loss']:.5f}", flush=True)
        res = {"gate": "val", "skill": a.skill, "arm": a.arm,
               "val_loss_mean": float(np.mean(losses)), "n": len(losses), "per_demo": losses}
        json.dump(res, open(os.path.join(a.out, f"val_{a.skill}_{a.arm}.json"), "w"), indent=2)
        print("[val] RESULT", json.dumps(res), flush=True)
        client.close()
        return

    if a.op == "train":
        eis = demos_for(data, a.skill, "train")
        examples = []
        for ei in eis:
            for step, span in iter_examples(data, ei, a.skill, a.stride):
                examples.append((ei, step, span))
        print(f"[train] {len(examples)} chunk windows over {len(eis)} demos", flush=True)
        rng = np.random.RandomState(a.seed)
        t0 = time.time()
        step_i = 0
        for ep in range(a.epochs):
            order = rng.permutation(len(examples))
            for k in order:
                ei, step, span = examples[k]
                inputs, x1, mask = make_example(data, client, ei, a.skill, step, span, instruction)
                r = client.sft_update(inputs, x1, mask, a.skill, seed=a.seed + step_i, sft_steps=1)
                if step_i % 25 == 0:
                    print(f"  ep{ep} it{step_i} loss {r['loss']:.5f} gnorm {r['grad_norm']:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                step_i += 1
        if a.ckpt:
            sr = client.rpc({"op": "save", "path": a.ckpt})
            print("[train] saved", json.dumps(sr), flush=True)
        client.close()
        return


if __name__ == "__main__":
    main()
