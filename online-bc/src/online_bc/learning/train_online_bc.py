"""Native flow-matching BC updates over a live success-only replay window."""

import argparse
import importlib
import json
import math
import time
from pathlib import Path
import numpy as np
from online_bc.data.replay import Replay
from online_bc.data.data_control import (
    read_controls,
    control_ready,
    atomic_json,
    select_candidates,
    episode_id,
)


def arrays(params):
    return {k: np.asarray(v.numpy() if hasattr(v, "numpy") else v) for k, v in params.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xiaomi", "pi05", "groot"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument(
        "--skills",
        nargs="+",
        choices=["grasp", "cup_placement", "button_press"],
        default=["cup_placement", "button_press"],
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--defer-inference", action="store_true")
    ap.add_argument("--resume")
    ap.add_argument("--wait-data-seconds", type=int, default=60)
    ap.add_argument("--controls")
    ap.add_argument("--control-health")
    args = ap.parse_args()
    assert args.batch_size >= 1
    assert args.model == "pi05" or args.batch_size == 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        (out / "verification.json").write_text(
            json.dumps(dict(model=args.model, passed=False, status="running"))
        )
    replay = Replay(args.data, controls=args.controls)
    deadline = time.monotonic() + args.wait_data_seconds
    while not replay.ingest():
        if time.monotonic() >= deadline:
            raise RuntimeError("No successful demonstrations available. Supply teacher BC shards.")
        time.sleep(2)
    backend = importlib.import_module("online_bc.models." + args.model + "_backend").Backend(
        args.checkpoint, args.lr
    )
    if args.resume:
        backend.load(args.resume)
    initial = arrays(backend.parameters()) if args.smoke else None
    frozen_before = backend.frozen_fingerprint() if args.smoke and args.model == "pi05" else None
    log = []
    steps = 2 if args.smoke else args.steps
    used = {}

    def usage(status):
        c = read_controls(args.controls)
        candidates = [
            item
            for item in select_candidates(replay.episodes, c)
            if any(m["skill"] in args.skills for m in item[2]["samples"])
        ]
        atomic_json(
            out / "data-usage.json",
            dict(
                status=status,
                completed_updates=len(log),
                planned_updates=steps,
                batch_size=args.batch_size,
                skills=args.skills,
                skill_sampling="balanced_batch"
                if "grasp" in args.skills
                else "alternating_updates",
                expert_pool_episodes=len(replay.expert_cache),
                control_revision=c["revision"],
                eligible_episodes=[episode_id(m) for _, _, m, _ in candidates],
                excluded=list(c["excluded"]),
                used=used,
                updated_at=time.time(),
            ),
        )

    for step in range(1, steps + 1):
        replay.ingest()
        skill = args.skills[(step - 1) % len(args.skills)]
        while True:
            control = read_controls(args.controls)
            ready, reason = control_ready(control, args.control_health)
            if ready:
                try:
                    batch_skills = (
                        [
                            args.skills[(step - 1 + index) % len(args.skills)]
                            for index in range(args.batch_size)
                        ]
                        if "grasp" in args.skills
                        else [skill] * args.batch_size
                    )
                    samples = [replay.sample(item) for item in batch_skills]
                    sample = samples[0]
                    break
                except RuntimeError:
                    reason = "no_eligible_success_data"
            usage(reason)
            time.sleep(2)
        usage("training")
        t = time.monotonic()
        result = (
            backend.update_batch(samples, seed=991000 + step)
            if args.batch_size > 1
            else backend.update(sample, seed=991000 + step)
        )
        for item in samples:
            used[item["episode_id"]] = used.get(item["episode_id"], 0) + 1
        assert math.isfinite(result["loss"]) and math.isfinite(result["grad_norm"])
        record = dict(
            step=step,
            model=args.model,
            skill=skill,
            source_model=sample["source_model"],
            source_seed=sample["seed"],
            episode_id=sample["episode_id"],
            control_revision=sample["control_revision"],
            seconds=time.monotonic() - t,
            batch_size=args.batch_size,
            skill_sample_counts={name: batch_skills.count(name) for name in set(batch_skills)},
            learning_rate=args.lr,
            **result,
        )
        log.append(record)
        print(json.dumps(record), flush=True)
        atomic_json(out / "updates.json", log)
        usage("training")
        if step % args.save_every == 0 or step == steps:
            dest = out / f"update-{step:06d}"
            dest.mkdir(exist_ok=True)
            backend.save(dest, step)
            metadata = dict(
                model=args.model,
                base_checkpoint=args.checkpoint,
                step=step,
                algorithm="success-filtered online behavior cloning",
                objective="native flow-matching supervised loss",
                data_roots=args.data,
                skills=args.skills,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                heldout_eval_seed_start=992001,
            )
            (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))
            tmp = out / "latest.tmp"
            tmp.write_text(str(dest.resolve()))
            tmp.replace(out / "latest")
    usage("completed")
    if args.smoke:
        changed = arrays(backend.parameters())
        assert any(np.any(changed[k] != initial[k]) for k in changed), (
            "Optimizer did not change adapters"
        )
        backend.load(dest)
        loaded = arrays(backend.parameters())
        assert all(np.array_equal(loaded[k], changed[k]) for k in loaded), (
            "Checkpoint reload changed adapters"
        )
        frozen_ok = None
        resume_ok = None
        if args.model == "pi05":
            frozen_ok = backend.frozen_fingerprint() == frozen_before
            assert frozen_ok, "Frozen backbone changed"
            sample = replay.sample(args.skills[0])
            backend.update(sample, seed=4242)
            expected = arrays(backend.parameters())
            backend.load(dest)
            backend.update(sample, seed=4242)
            actual = arrays(backend.parameters())
            resume_ok = all(
                np.allclose(actual[k], expected[k], rtol=1e-6, atol=1e-8) for k in expected
            )
            assert resume_ok, "Optimizer resume diverged"
            backend.load(dest)
        prediction = None
        if not args.defer_inference:
            prediction = np.asarray(backend.infer(replay.sample("cup_placement"), seed=42))
            assert (
                prediction.ndim == 2
                and prediction.shape[1] == 12
                and len(prediction) >= 16
                and np.isfinite(prediction).all()
            )
        report = dict(
            model=args.model,
            passed=not args.defer_inference,
            status="inference_pending" if args.defer_inference else "passed",
            native_optimizer_updates=steps,
            skills_tested=[x["skill"] for x in log],
            finite_loss_and_grad=True,
            adapter_changed=True,
            checkpoint_reload_equal=True,
            inference_after_reload=not args.defer_inference,
            inference_action_shape=list(prediction.shape) if prediction is not None else None,
            trainable_parameters=sum(v.size for v in loaded.values()),
            frozen_backbone_unchanged=frozen_ok,
            optimizer_resume_equal=resume_ok,
            updates=log,
        )
        (out / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
