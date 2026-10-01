"""Lab coordinator; dataset archives upload directly from each rollout host."""

import argparse
import json
import subprocess
import time
import sys
from pathlib import Path
from online_bc.orchestration.collection import batch_plan, eligible_successes, ready_to_train
from online_bc.data.data_control import atomic_json, read_controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    c = json.loads(Path(args.config).read_text())
    run = c["run"]
    root = Path(c["root"])
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    def sync(direction, folder, prefix):
        return subprocess.run(
            [
                c["transport_python"],
                "-m",
                "online_bc.transport.hf_transfer",
                direction,
                str(folder),
                prefix,
                "--token-file",
                c["token_file"],
            ],
            capture_output=True,
            text=True,
        )

    status = root / "status.json"
    start = json.loads(status.read_text())["next_round"] if status.exists() else 1
    if start == 1 and c.get("bootstrap_weights") and not args.dry_run:
        for model, w in c["workers"].items():
            adapter = root / f"adapters/{model}/round-0000"
            while True:
                result = sync("download", adapter, f"{run}/weights/{model}/round-0000")
                if result.returncode == 0 and (adapter / "metadata.json").exists():
                    break
                time.sleep(10)
            subprocess.run(
                [
                    x.format(round=0, model=model, run=run, adapter=str(adapter))
                    for x in w["reload_argv"]
                ],
                check=True,
            )
    for round_index in range(start, args.rounds + 1):
        if args.dry_run:
            print(
                json.dumps(
                    dict(
                        round=round_index,
                        workers=list(c["workers"]),
                        steps=c["steps_per_round"],
                        plan=batch_plan(c, round_index, 0),
                        target_new_successes=c.get("target_new_successes"),
                        run=run,
                    )
                )
            )
            break
        if (
            round_index == 1
            and c.get("evaluation_argv")
            and not (root / "evaluation-version-0000.done").exists()
        ):
            subprocess.run(
                [x.format(round=0, run=run, model="pi05") for x in c["evaluation_argv"]], check=True
            )
            (root / "evaluation-version-0000.done").write_text("passed")
        collection_file = root / f"collection-round-{round_index:04d}.json"
        sources = (
            json.loads(collection_file.read_text())["sources"] if collection_file.exists() else []
        )
        attempted = sum(len(source["seeds"]) for source in sources)
        successes = eligible_successes(sources, c.get("controls_file"))
        batch_index = max((source["batch"] for source in sources), default=0)
        while not ready_to_train(c, round_index, attempted, successes):
            if read_controls(c.get("controls_file"))["paused"]:
                time.sleep(2)
                continue
            plan = batch_plan(c, round_index, attempted)
            if not plan:
                atomic_json(
                    status,
                    dict(
                        next_round=round_index,
                        status="insufficient_new_successes",
                        attempts=attempted,
                        successes=successes,
                    ),
                )
                raise RuntimeError(
                    f"Collected {attempted} attempts, {successes} valid new successes. No learner job queued; collected data is retained."
                )
            batch_index += 1
            processes = []
            for assignment in plan:
                node = assignment["node"]
                w = c["workers"][node]
                cmd = [
                    x.format(
                        round=round_index,
                        run=run,
                        model=w.get("model", node),
                        batch=batch_index,
                        episodes=assignment["episodes"],
                        seed_offset=assignment["seed_offset"],
                    )
                    for x in w["collect_argv"]
                ]
                log_path = root / f"{node}-round-{round_index:04d}-batch-{batch_index:02d}.log"
                log = log_path.open("w")
                processes.append(
                    (
                        node,
                        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT),
                        log,
                        log_path,
                    )
                )
            for node, process, log, log_path in processes:
                code = process.wait()
                log.close()
                if code:
                    raise RuntimeError(
                        f"{node} collection/upload failed: {log_path}. No learner job queued."
                    )
                summary = json.loads(log_path.read_text().strip().splitlines()[-1])
                assert summary["node"] == node and summary["round"] == round_index
                sources.append(summary)
                atomic_json(collection_file, dict(sources=sources))
                if c.get("tracking_root"):
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "online_bc.review.review_catalog",
                            "--config",
                            args.config,
                            "--round",
                            str(round_index),
                            "--source-json",
                            json.dumps(summary),
                        ],
                        check=True,
                    )
            attempted = sum(len(source["seeds"]) for source in sources)
            successes = eligible_successes(sources, c.get("controls_file"))
        prefixes = sources
        for model in c.get("learner_models", c["workers"]):
            folder = root / f"jobs/{model}/round-{round_index:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "job.json").write_text(
                json.dumps(
                    dict(
                        model=model,
                        attempts=attempted,
                        new_successes=successes,
                        round=round_index,
                        data_prefixes=prefixes,
                        steps=c["steps_per_round"],
                        skills=c.get("skills", ["cup_placement", "button_press"]),
                    ),
                    indent=2,
                )
            )
            r = sync("upload", folder, f"{run}/jobs/{model}/round-{round_index:04d}")
            if r.returncode:
                raise RuntimeError(r.stderr[-1000:])
        for node, w in c["workers"].items():
            model = w.get("model", node)
            adapter = root / f"adapters/{node}/round-{round_index:04d}"
            while True:
                r = sync("download", adapter, f"{run}/weights/{model}/round-{round_index:04d}")
                if r.returncode == 0 and (adapter / "metadata.json").exists():
                    break
                time.sleep(10)
            subprocess.run(
                [
                    x.format(adapter=str(adapter), round=round_index, model=model, run=run)
                    for x in w["reload_argv"]
                ],
                check=True,
            )
        if round_index in c.get("evaluation_versions", []) and c.get("evaluation_argv"):
            subprocess.run(
                [x.format(round=round_index, run=run, model="pi05") for x in c["evaluation_argv"]],
                check=True,
            )
        (root / "status.json").write_text(
            json.dumps(dict(completed_round=round_index, next_round=round_index + 1), indent=2)
        )


if __name__ == "__main__":
    main()
