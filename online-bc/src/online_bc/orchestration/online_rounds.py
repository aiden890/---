"""Lab coordinator; dataset archives upload directly from each rollout host."""

import argparse
import json
import subprocess
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from online_bc.orchestration.collection import batch_plan, eligible_successes, ready_to_train
from online_bc.data.data_control import atomic_json, read_controls
from online_bc.orchestration.adaptation import assess


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
    collection_policy = root / "collection-policy.json"
    if collection_policy.exists():
        c.update(json.loads(collection_policy.read_text()))
    start = json.loads(status.read_text())["next_round"] if status.exists() else 1
    evaluation_pool = ThreadPoolExecutor(max_workers=1)
    publication_pool = ThreadPoolExecutor(max_workers=1)
    evaluation = None
    learning_rate = c.get("learning_rate", 1e-4)
    resume_best = None
    prior_decisions = sorted(root.glob("adaptation-version-*.json"))
    if prior_decisions:
        previous = json.loads(prior_decisions[-1].read_text())
        learning_rate = previous["learning_rate"]
        if previous.get("version") == start - 1:
            resume_best = previous.get("resume_round")

    def evaluate(version):
        began = time.monotonic()
        folder = root / f"evaluation/version-{version:04d}"
        done = folder / "done.json"
        if done.exists():
            return json.loads(done.read_text())
        folder.mkdir(parents=True, exist_ok=True)
        count = (
            30
            if version == 0 or version % c.get("full_eval_every", 2) == 0 or version == args.rounds
            else c.get("quick_eval_episodes", 10)
        )
        external = c.get("baseline_external_pid") if version == 0 else None
        if external:
            # Adopt the already-running baseline SSH worker when replacing the
            # coordinator. Never launch a duplicate evaluator over its files.
            while True:
                result = sync("download", folder, f"{run}/evaluation/pi05/version-{version:04d}")
                if result.returncode == 0 and (folder / "metrics.json").exists():
                    break
                try:
                    os.kill(external, 0)
                except ProcessLookupError:
                    raise RuntimeError(
                        "Adopted baseline ended without published evaluation metrics"
                    )
                time.sleep(15)
        else:
            command = [x.format(round=version, run=run, model="pi05") for x in c["evaluation_argv"]]
            if c.get("async_evaluation"):
                command[-1] += f" --eval-episodes {count}"
            with (folder / "worker.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            result = sync("download", folder, f"{run}/evaluation/pi05/version-{version:04d}")
            if result.returncode:
                raise RuntimeError(result.stderr[-1000:])
        report = json.loads((folder / "metrics.json").read_text())
        assert report["policy_version"] == version and report["training_data"] is False
        assert report["attempts"] == count
        if "outcomes" not in report:
            worker = c["workers"][c.get("evaluation_node", "v4")]
            remote_config = worker["collect_argv"][-1].split("--config ")[1].split()[0]
            script = (
                "import pathlib,json; c=json.loads(pathlib.Path("
                + repr(remote_config)
                + ").read_text()); "
                "root=pathlib.Path(c['output_root']).parent/'evaluation'/"
                + repr(f"version-{version:04d}")
                + "; "
                "rows=[json.loads(p.read_text()) for p in root.glob('pi05/*/result.json')]; "
                "print(json.dumps([dict(seed=r['seed'],cup_placed=r['cup_placed'],grasped='mug_grasped' in r['milestones']) for r in rows]))"
            )
            result = subprocess.run(
                ["ssh", worker["host"], "python3 -"],
                input=script,
                text=True,
                capture_output=True,
                check=True,
            )
            report["outcomes"] = json.loads(result.stdout)
        report["controller_wall_seconds"] = time.monotonic() - began
        report.setdefault("wall_seconds", None if external else report["controller_wall_seconds"])
        report["adopted_external"] = bool(external)
        atomic_json(done, report)
        print(json.dumps(dict(event="evaluation_complete", **report)), flush=True)
        return report

    def finish_evaluation():
        nonlocal evaluation, learning_rate, resume_best
        if evaluation is not None:
            evaluation.result()
            evaluation = None
            reports = [
                json.loads(path.read_text()) for path in root.glob("evaluation/version-*/done.json")
            ]
            decision = assess(reports, learning_rate)
            version = max(row["policy_version"] for row in reports)
            adjustments = [
                json.loads(path.read_text()) for path in root.glob("adaptation-version-*.json")
            ]
            last_change = max(
                (
                    row.get("version", -2)
                    for row in adjustments
                    if row["action"] in ["reduce_learning_rate", "resume_best_checkpoint"]
                ),
                default=-2,
            )
            if decision["action"] == "reduce_learning_rate" and version - last_change < 2:
                decision.update(
                    action="continue", learning_rate=learning_rate, reason="adjustment_cooldown"
                )
            decision["version"] = version
            learning_rate = decision["learning_rate"]
            resume_best = decision.get("resume_round")
            atomic_json(root / f"adaptation-version-{version:04d}.json", decision)

    def start_evaluation(version):
        nonlocal evaluation
        finish_evaluation()
        evaluation = evaluation_pool.submit(evaluate, version)

    def publish(command, version, source):
        log_path = root / f"publish-{version:04d}-{source['node']}-{source['batch']:02d}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(
                root / f"publish-failure-{version:04d}-{source['node']}.json",
                dict(returncode=result.returncode, log=str(log_path), source=source),
            )

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
            start_evaluation(0)
            if not c.get("async_evaluation"):
                finish_evaluation()
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
            pending_path = root / f"pending-round-{round_index:04d}.json"
            pending = json.loads(pending_path.read_text()) if pending_path.exists() else None
            available = c
            if round_index > 1 and evaluation is not None and not evaluation.done():
                available = dict(
                    c,
                    workers={
                        node: worker
                        for node, worker in c["workers"].items()
                        if node != c.get("evaluation_node", "v4")
                    },
                )
            plan = pending["plan"] if pending else batch_plan(available, round_index, attempted)
            if not plan:
                cap = c.get("max_attempts_per_round", 32)
                if c.get("extend_collection_on_shortfall", True) and cap < 96:
                    c["max_attempts_per_round"] = min(96, cap * 2)
                    atomic_json(
                        collection_policy, dict(max_attempts_per_round=c["max_attempts_per_round"])
                    )
                    print(
                        json.dumps(
                            dict(
                                event="extend_collection",
                                round=round_index,
                                attempts=attempted,
                                successes=successes,
                                new_cap=c["max_attempts_per_round"],
                            )
                        ),
                        flush=True,
                    )
                    continue
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
            batch_index = pending["batch"] if pending else batch_index + 1
            atomic_json(
                pending_path,
                dict(
                    batch=batch_index,
                    plan=plan,
                    external_pids=pending.get("external_pids", {}) if pending else {},
                ),
            )
            batch_began = time.monotonic()
            atomic_json(
                status,
                dict(
                    next_round=round_index,
                    status="collecting",
                    attempts=attempted,
                    successes=successes,
                    batch=batch_index,
                ),
            )
            processes = []
            for assignment in plan:
                node = assignment["node"]
                if any(s["node"] == node and s["batch"] == batch_index for s in sources):
                    continue
                if node == c.get("evaluation_node", "v4"):
                    finish_evaluation()
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
                external = pending.get("external_pids", {}).get(node) if pending else None
                if external:
                    # Joining an orphaned SSH collector is safe: it keeps its
                    # original log and output files and is never relaunched.
                    class AdoptedProcess:
                        def wait(self, pid=external, path=log_path):
                            while True:
                                try:
                                    summary = json.loads(path.read_text().strip().splitlines()[-1])
                                    if (
                                        summary.get("prefix")
                                        and summary.get("accepted") is not None
                                    ):
                                        return 0
                                except (OSError, ValueError, IndexError):
                                    pass
                                try:
                                    os.kill(pid, 0)
                                except ProcessLookupError:
                                    return 1
                                time.sleep(5)

                    processes.append((node, AdoptedProcess(), log_path.open("a"), log_path))
                    continue
                log = log_path.open("w")
                processes.append(
                    (
                        node,
                        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT),
                        log,
                        log_path,
                    )
                )
            failures = []
            for node, process, log, log_path in processes:
                code = process.wait()
                log.close()
                if code:
                    failures.append(f"{node}: {log_path}")
                    continue
                summary = json.loads(log_path.read_text().strip().splitlines()[-1])
                assert summary["node"] == node and summary["round"] == round_index
                sources.append(summary)
                atomic_json(collection_file, dict(sources=sources))
                if c.get("tracking_root"):
                    publication_pool.submit(
                        publish,
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
                        round_index,
                        summary,
                    )
            if failures:
                raise RuntimeError(
                    "Collection/upload failed; all workers joined, pending plan retained: "
                    + "; ".join(failures)
                )
            pending_path.unlink()
            atomic_json(
                root / f"timing-round-{round_index:04d}-batch-{batch_index:02d}.json",
                dict(
                    round=round_index,
                    batch=batch_index,
                    collection_upload_wall_seconds=time.monotonic() - batch_began,
                ),
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
                        lr=learning_rate,
                        batch_size=c.get("batch_size", 1),
                        resume_round=resume_best,
                    ),
                    indent=2,
                )
            )
            r = sync("upload", folder, f"{run}/jobs/{model}/round-{round_index:04d}")
            if r.returncode:
                raise RuntimeError(r.stderr[-1000:])
        atomic_json(
            status,
            dict(
                next_round=round_index,
                status="waiting_for_learner",
                attempts=attempted,
                successes=successes,
            ),
        )
        resume_best = None
        for node, w in c["workers"].items():
            if node == c.get("evaluation_node", "v4") and evaluation is not None:
                atomic_json(
                    status,
                    dict(next_round=round_index, status="waiting_for_evaluation"),
                )
                finish_evaluation()
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
        if (c.get("eval_every_round") or round_index in c.get("evaluation_versions", [])) and c.get(
            "evaluation_argv"
        ):
            start_evaluation(round_index)
            if not c.get("async_evaluation"):
                finish_evaluation()
        (root / "status.json").write_text(
            json.dumps(dict(completed_round=round_index, next_round=round_index + 1), indent=2)
        )
    finish_evaluation()
    evaluation_pool.shutdown()
    publication_pool.shutdown()


if __name__ == "__main__":
    main()
