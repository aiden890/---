"""Keep a policy version stable for separately recorded evaluation variants."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from online_bc.data.data_control import atomic_json
from online_bc.rollout.docker_worker import run_args
from online_bc.rollout.worker import active_policy_readers, conditional_cup_rate, profile_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--containers", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--eval-episodes", type=int, choices=[10, 30], default=10)
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument(
        "--variant", choices=["base_prefix", "standard_control"], default="base_prefix"
    )
    args = parser.parse_args()
    exclusive = args.variant == "base_prefix" and not args.reuse_server
    config = json.loads(Path(args.config).read_text())
    assert config["model"] == "pi05"
    runtime = Path(config["output_root"]).parent
    root = runtime / "shadow-evaluation" / args.variant / f"version-{args.version:04d}"
    root.mkdir(parents=True, exist_ok=True)
    singleton = (root / "runner.lock").open("a")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "report.json").exists() or (root / "started.json").exists():
        raise RuntimeError("Shadow evaluation already exists; inspect it before retrying")
    atomic_json(root / "status.json", dict(status="waiting_for_idle_policy", version=args.version))
    (root / "pid").write_text(str(os.getpid()))
    lock_path = Path(config["adapter_root"]).parent / "policy-pi05.lock"
    lock = lock_path.open("a")
    deadline = time.monotonic() + args.wait_seconds
    reference_path = (
        runtime / "evaluation" / f"version-{args.version:04d}" / "evaluation-upload/metrics.json"
    )
    while time.monotonic() < deadline:
        if not reference_path.exists():
            time.sleep(2)
            continue

        def wait_expired(signum, frame):
            raise TimeoutError("No idle policy window before deadline")

        signal.signal(signal.SIGALRM, wait_expired)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        try:
            # Queue the reservation rather than polling between fast batch handoffs.
            fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except TimeoutError:
            atomic_json(root / "status.json", dict(status="skipped_no_idle_window"))
            return
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        while exclusive and active_policy_readers(args.config, lock_path=lock_path):
            if time.monotonic() >= deadline:
                atomic_json(root / "status.json", dict(status="skipped_no_idle_window"))
                return
            time.sleep(2)
        with urllib.request.urlopen(config["policy_url"], timeout=3) as response:
            health = json.load(response)
        if health["version"] != args.version:
            atomic_json(
                root / "status.json",
                dict(status="skipped_policy_window_missed", observed=health["version"]),
            )
            return
        if (args.variant == "standard_control" or args.reuse_server) and not health.get(
            "base_prefix_enabled"
        ):
            atomic_json(root / "status.json", dict(status="skipped_routing_server_unavailable"))
            return
        break
    else:
        atomic_json(root / "status.json", dict(status="skipped_no_idle_window"))
        return
    containers = json.loads(Path(args.containers).read_text())
    name = "coffee-online-bc-policy-pi05"
    python = containers[0]["Config"]["Cmd"][0]
    checkpoint = containers[0]["Config"]["Cmd"]
    checkpoint = checkpoint[checkpoint.index("--checkpoint") + 1]
    adapter = config["adapter_container_root"] + f"/round-{args.version:04d}"
    command = run_args(containers[0]) + [
        "-d",
        "--name",
        name,
        containers[0]["Config"]["Image"],
        python,
        "-m",
        "online_bc.models.serve_bc_policy",
        "--model",
        "pi05",
        "--checkpoint",
        checkpoint,
        "--adapter",
        adapter,
    ]
    children = []
    server_modified = False
    logs = []
    output = f"/results/coffee-online-bc/shadow-evaluation/{args.variant}/version-{args.version:04d}/pi05"

    def replace_server(enable_phase):
        nonlocal server_modified
        state = subprocess.run(["docker", "inspect", name], capture_output=True)
        if state.returncode == 0:
            with (
                root / ("policy-before-shadow.log" if enable_phase else "policy-failed-shadow.log")
            ).open("wb") as log:
                subprocess.run(["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT)
            server_modified = True
            subprocess.run(["docker", "stop", "-t", "15", name], check=True, capture_output=True)
            subprocess.run(["docker", "rm", name], check=True, capture_output=True)
        subprocess.run(command + (["--enable-base-prefix"] if enable_phase else []), check=True)
        for _ in range(120):
            try:
                with urllib.request.urlopen(config["policy_url"], timeout=2) as response:
                    ready = json.load(response)
                assert ready["version"] == args.version
                assert not enable_phase or ready.get("base_prefix_enabled")
                return
            except (OSError, AssertionError):
                time.sleep(2)
        raise RuntimeError("Shadow policy did not become ready")

    try:
        reference = json.loads(reference_path.read_text())
        assert reference["policy_version"] == args.version
        atomic_json(
            root / "started.json", dict(version=args.version, at=time.time(), variant=args.variant)
        )
        if args.variant == "base_prefix" and not args.reuse_server:
            replace_server(True)
        shadow = json.loads(Path(args.containers).read_text())
        shadow[1]["Config"]["Env"] = [
            x for x in shadow[1]["Config"]["Env"] if not x.startswith("COFFEE_BC_VARIANT=")
        ]
        shadow[1]["Config"]["Env"].append(
            "COFFEE_BC_VARIANT=" + ("base_prefix" if args.variant == "base_prefix" else "standard")
        )
        container_id = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.Id}}", name], text=True
        ).strip()
        shadow_config = root / "containers.json"
        atomic_json(shadow_config, shadow)
        seeds = list(range(992001, 992001 + args.eval_episodes))
        started = time.monotonic()
        atomic_json(
            root / "status.json",
            dict(status="evaluating", version=args.version, episodes=args.eval_episodes),
        )
        for index in range(2):
            log = (root / f"worker-{index}.log").open("w")
            logs.append(log)
            child = subprocess.Popen(
                [
                    config["host_python"],
                    "-m",
                    "online_bc.rollout.docker_worker",
                    "--model",
                    "pi05",
                    "collect",
                    "--seeds",
                    ",".join(map(str, seeds[index::2])),
                    "--out",
                    output,
                    "--containers",
                    str(shadow_config),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(child)
        while any(child.poll() is None for child in children):
            if time.monotonic() - started > (1800 if args.eval_episodes == 10 else 2700):
                raise TimeoutError("Shadow simulation exceeded its evaluation deadline")
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("Shadow simulator failed")
            time.sleep(2)
        assert all(child.returncode == 0 for child in children), "Shadow simulator failed"
        rows = [json.loads(p.read_text()) for p in sorted((root / "pi05").glob("*/result.json"))]
        assert {row["seed"] for row in rows} == set(seeds) and len(rows) == args.eval_episodes
        for row in rows:
            assert row["policy_version"] == args.version
            assert row["policy_variant"] == (
                "base_prefix" if args.variant == "base_prefix" else "standard"
            )
            if args.variant == "standard_control":
                assert not row["policy_phases"]
                continue
            assert row["policy_phases"]
            start = row["cup_skill_start"]
            for phase in row["policy_phases"]:
                assert phase["step"] % 16 == 0
                prefix = start is None or phase["step"] < start
                assert phase["phase"] == ("prefix" if prefix else "cup_placement")
                assert phase["effective_policy_version"] == (0 if prefix else args.version)
        outcomes = [
            dict(
                seed=x["seed"], cup_placed=x["cup_placed"], grasped="mug_grasped" in x["milestones"]
            )
            for x in rows
        ]
        standard = [x for x in reference["outcomes"] if x["seed"] in seeds]
        assert len(standard) == args.eval_episodes
        report = dict(
            policy_version=args.version,
            variant=args.variant,
            server_container_id=container_id,
            attempts=args.eval_episodes,
            cup_successes=sum(x["cup_placed"] for x in outcomes),
            grasp_successes=sum(x["grasped"] for x in outcomes),
            cup_given_grasp=conditional_cup_rate(outcomes),
            outcomes=outcomes,
            standard_same10=standard[:10],
            standard_same_seeds=standard,
            server_reused=args.reuse_server or args.variant == "standard_control",
            policy_lock_mode="exclusive" if exclusive else "shared",
            wall_seconds=time.monotonic() - started,
            profiling=profile_summary(rows),
            phase_alignment_verified=args.variant == "base_prefix",
            training_data=False,
            deployed_for_training=False,
        )
        atomic_json(root / "report.json", report)
        upload = root / "evaluation-upload"
        upload.mkdir(exist_ok=True)
        atomic_json(upload / "metrics.json", report)
        prefix = f"{args.run}/shadow-evaluation/pi05/{args.variant}/version-{args.version:04d}"
        fields = dict(
            round=args.version,
            batch=0,
            model="pi05",
            run=args.run,
            url=config["policy_url"],
            direction="upload",
            directory=str(upload),
            prefix=prefix,
        )
        subprocess.run([x.format(**fields) for x in config["transport_argv"]], check=True)
        atomic_json(root / "status.json", dict(status="completed", version=args.version))
        print(json.dumps(report), flush=True)
    except Exception as error:
        # Stop only containers owned by this shadow output before restoring service.
        for container_id in subprocess.check_output(["docker", "ps", "-q"], text=True).split():
            info = json.loads(
                subprocess.check_output(["docker", "inspect", container_id], text=True)
            )[0]
            if output in (info["Config"].get("Cmd") or []):
                subprocess.run(["docker", "stop", "-t", "10", container_id], capture_output=True)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
        if server_modified:
            replace_server(False)
        atomic_json(
            root / "status.json",
            dict(status="failed_restored_standard", error_type=type(error).__name__),
        )
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
