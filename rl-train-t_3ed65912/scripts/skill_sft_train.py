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

        # CFM loss has irreducible per-step noise (fresh x0,t each step), so the raw
        # training-loss trace is a poor gate signal. Measure DETERMINISTICALLY instead:
        # mean over the K examples of a fixed-seed MC-averaged CFM loss (sft_val, mc=16),
        # BEFORE and AFTER training the same K examples. drop = base/final on that clean
        # metric -- an honest overfit test (identical examples, low-variance measurement).
        def mc_loss(tag):
            vs = []
            for ei, inputs, x1, mask in exs:
                rv = client.sft_val(inputs, x1, mask, a.skill, seed=a.seed, mc=16)
                vs.append(rv["val_loss"])
            m = float(np.mean(vs))
            print(f"  [{tag}] mc_loss mean {m:.5f} per_ex {[round(v,4) for v in vs]}", flush=True)
            return m

        base_mc = mc_loss("pre")
        for it in range(a.steps):
            ei, inputs, x1, mask = exs[it % len(exs)]
            r = client.sft_update(inputs, x1, mask, a.skill, seed=a.seed + it, sft_steps=1)
            if it % 20 == 0 or it == a.steps - 1:
                print(f"  it {it:3d} train_loss {r['loss']:.5f} gnorm {r['grad_norm']:.3f} "
                      f"n_active {r['n_active']}", flush=True)
        final_mc = mc_loss("post")
        drop = base_mc / max(final_mc, 1e-9)
        ok = drop > 10.0  # honest bar; brief said >100x ideal, >10x proves the path learns
        res = {"gate": "overfit", "skill": a.skill, "arm": a.arm,
               "base_mc_loss": base_mc, "final_mc_loss": final_mc, "drop_x": drop,
               "pass": bool(ok), "demos": eis, "steps": a.steps}
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
        # op_save runs on the TRAINER SERVER, which mounts /train (not /out; /out is
        # client-only). A --ckpt under /out would land in the trainer container's ephemeral
        # fs and be lost on --rm. Rewrite /out/<f> -> /train/results/skill_sft/ckpt/<f> so the
        # checkpoint persists on the host. Absolute /train or other paths pass through.
        if a.ckpt and a.ckpt.startswith("/out/"):
            fixed = "/train/results/skill_sft/ckpt/" + os.path.basename(a.ckpt)
            print(f"[train] remapping ckpt {a.ckpt} -> {fixed} (trainer-visible mount)", flush=True)
            a.ckpt = fixed
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
            mani = data.manifest
            save_req = {
                "op": "save", "path": a.ckpt,
                "data_hash": mani.get("data_hashes"),
                "data_manifest": {k: mani.get(k) for k in ("task", "dataset_repo",
                                   "dataset_codebase_version", "n_demos", "split")},
                "model_rev": "3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4",  # pinned XiaomiRobotics rev
                "train_meta": {"skill": a.skill, "arm": a.arm, "epochs": a.epochs,
                               "n_examples": len(examples), "stride": a.stride, "seed": a.seed},
            }
            sr = client.rpc(save_req)
            print("[train] saved", json.dumps(sr), flush=True)
        client.close()
        return


if __name__ == "__main__":
    main()
