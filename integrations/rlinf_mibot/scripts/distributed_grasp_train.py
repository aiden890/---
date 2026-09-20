#!/usr/bin/env python3
"""Long-running Spark rollout -> RTX 3090 learner coordinator.

The two resident trainer processes hold identical policy snapshots.  Spark runs the
RoboCasa group and exports rollout-store payloads; the 3090 imports a mixed-outcome
group, updates LoRA, saves a full resumable LoRA/optimizer checkpoint, and publishes
the next actor snapshot.  Homogeneous groups are retained as evidence but never sent
to the optimizer.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPARK = "spark1"
LEARNER = "amp_csi"
ACTOR_CONTAINER = "rlinf-dist-current-actor"
LEARNER_CONTAINER = "rlinf-merged-verify-trainer"
SPARK_ROOT = "/home/csi-agent-dgx_spark1/workspace/rlinf-mibot-prep"
LEARNER_ROOT = "/home/guest/rlinf_merge_verify"
RUNNER = f"{SPARK_ROOT}/source-current/integrations/rlinf_mibot/run.sh"
LOCAL_STAGE = Path("/tmp/rlinf-mibot-long-training")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def source_commit() -> str:
    """Return the exact coordinator revision recorded in rollout provenance."""
    return subprocess.check_output(
        ["git", "-C", str(HERE.parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()


def run(argv, *, stdout=None):
    return subprocess.run(argv, check=True, text=stdout is None, stdout=stdout)


def ssh(host: str, command: str, *, capture=False):
    return subprocess.run(["ssh", host, command], check=True, text=True,
                          capture_output=capture)


def rpc(host: str, container: str, request: dict) -> dict:
    encoded = base64.b64encode(json.dumps(request).encode()).decode()
    code = (
        "import base64,json,pickle,socket,struct;"
        f"q=json.loads(base64.b64decode('{encoded}'));"
        "s=socket.create_connection(('127.0.0.1',10088),60);s.settimeout(None);"
        "b=pickle.dumps(q);s.sendall(struct.pack('>I',len(b))+b);"
        "n=struct.unpack('>I',s.recv(4))[0];x=b'';"
        "exec(\"while len(x)<n:\\n x+=s.recv(n-len(x))\");"
        "print(json.dumps(pickle.loads(x),default=str))"
    )
    result = ssh(host, f"docker exec {container} python3 -c {json.dumps(code)}",
                 capture=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def collect(run_id: str, seed: int, *, groups: int = 8,
            group_size: int = 8, workers: int = 4) -> list[dict]:
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    node_ranks = ",".join("0" for _ in range(workers))
    env = (
        f"RLINF_CHECKPOINT={SPARK_ROOT}/checkpoint "
        f"RLINF_ASSETS={SPARK_ROOT}/assets "
        f"RLINF_RESULTS={SPARK_ROOT}/results "
        f"RLINF_CACHE={SPARK_ROOT}/cache "
        f"RLINF_IMAGE=rlinf-mibot:spark-a4ee4562 RLINF_WORKERS={workers} "
        f"RLINF_NODE_RANKS={node_ranks}"
    )
    state = ssh(
        SPARK,
        f"if test -f {SPARK_ROOT}/results/{run_id}/DONE; then echo done; "
        f"elif test -d {SPARK_ROOT}/results/{run_id}; then echo partial; else echo new; fi",
        capture=True,
    ).stdout.strip()
    if state == "partial":
        summary_text = ssh(
            SPARK,
            f"test -f {SPARK_ROOT}/results/{run_id}/summary.json && "
            f"cat {SPARK_ROOT}/results/{run_id}/summary.json || true",
            capture=True,
        ).stdout.strip()
        if summary_text:
            summary = json.loads(summary_text)
            checks = summary.get("audit", {}).get("checks", {})
            required = (
                "episodes_present", "exact_expected_job_coverage",
                "no_duplicate_job_ids", "no_duplicate_seeds", "no_config_mixing",
                "no_optimizer_updates", "planned_configs_present",
                "production_artifacts_present", "rows_match_planned_jobs",
                "trainer_payload_hashes_valid",
            )
            expected = int(groups) * int(group_size)
            if (int(summary.get("planned_episodes", -1)) == expected
                    and int(summary.get("completed_episodes", -1)) == expected
                    and not summary.get("fatal_failures")
                    and int(summary.get("optimizer_update_requests", -1)) == 0
                    and all(checks.get(key) is True for key in required)):
                # A coordinator interruption can remove DONE after every signed payload
                # was already written. Reuse that immutable wave instead of exporting
                # over existing payload files; caller revalidates policy version/hash.
                state = "salvaged"
    if state not in ("done", "salvaged"):
        provenance = source_commit()
        if state == "partial":
            # A code-only coordinator restart must not invalidate already collected,
            # policy-bound trajectories. Resume with the provenance embedded when the
            # immutable plan was first created; payload policy hash checks still gate use.
            manifest_commit = ssh(
                SPARK,
                f"python3 -c \"import json; print(json.load(open('{SPARK_ROOT}/results/"
                f"{run_id}/manifest.json'))['run']['source_commit'])\"",
                capture=True,
            ).stdout.strip()
            if not re.fullmatch(r"[0-9a-f]{40}", manifest_commit):
                raise RuntimeError(f"invalid source commit in partial manifest: {manifest_commit!r}")
            provenance = manifest_commit
        mode = "grid-resume" if state == "partial" else "grid-parallel"
        command = (
            f"{env} bash {RUNNER} {mode} {run_id} "
            f"sampler.noise_level=0.1 rollout.groups={int(groups)} "
            f"rollout.group_size={int(group_size)} "
            "env.horizon=208 reward.variant=grasp_binary "
            f"seed.base={seed} runtime.max_in_flight={workers} "
            f"run.source_commit={provenance}"
        )
        ssh(SPARK, command)
    text = ssh(
        SPARK,
        f"docker exec {ACTOR_CONTAINER} cat /results/{run_id}/episodes.jsonl",
        capture=True,
    ).stdout
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def transfer_run(run_id: str) -> dict:
    if not SAFE_NAME.fullmatch(run_id):
        raise ValueError(f"unsafe run ID: {run_id!r}")
    stage = f"/results/.transfer-{run_id}"
    remote = f"{LEARNER_ROOT}/results/{run_id}"
    source_command = (
        "set -o pipefail; "
        f"docker exec {ACTOR_CONTAINER} python3 /integration/src/prepare_streaming_transfer.py "
        f"/results/{run_id} {stage} >&2; "
        f"docker exec {ACTOR_CONTAINER} tar -C {stage} -cf - . | zstd -1 -T0"
    )
    destination_command = (
        f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)} && "
        f"zstd -d -T0 | tar -C {shlex.quote(remote)} -xf -"
    )
    started = time.time()
    source = subprocess.Popen(
        ["ssh", SPARK, "bash", "-lc", shlex.quote(source_command)],
        stdout=subprocess.PIPE)
    try:
        destination = subprocess.run(
            ["ssh", LEARNER, "bash", "-lc", shlex.quote(destination_command)],
            stdin=source.stdout, check=False)
        if source.stdout is not None:
            source.stdout.close()
        source_status = source.wait()
        if source_status or destination.returncode:
            raise subprocess.CalledProcessError(
                source_status or destination.returncode, "compressed rollout transfer")
    finally:
        if source.poll() is None:
            source.terminate()
            source.wait()
        ssh(SPARK, f"docker exec {ACTOR_CONTAINER} rm -rf {stage}")
    manifest = json.loads(ssh(
        LEARNER, f"cat {remote}/transfer_manifest.json", capture=True).stdout)
    return {"elapsed_seconds": time.time() - started, **manifest}


def consume(run_id: str) -> dict:
    command = (
        "docker run --rm --network host -e PYTHONPATH=/integration/src "
        f"-v {LEARNER_ROOT}/integration:/integration:ro "
        f"-v {LEARNER_ROOT}/results:/results xiaomi-cu121:t_9f03a613 "
        "python3 /integration/src/consume_collector.py "
        f"/results/{run_id} --host 127.0.0.1 --port 10088 "
        "--config /integration/configs/consume_production.yaml"
    )
    ssh(LEARNER, command)
    report = ssh(
        LEARNER,
        f"cat {LEARNER_ROOT}/results/{run_id}/consume_smoke/consume_smoke.json",
        capture=True,
    )
    return json.loads(report.stdout)


def sync_adapter(run_id: str, expected_version: int) -> dict:
    learner_path = f"/results/{run_id}/consume_smoke/adapter-v{expected_version}.pt"
    published = rpc(LEARNER, LEARNER_CONTAINER,
                    {"op": "publish_adapter", "path": learner_path})
    if int(published["policy_version"]) != expected_version:
        raise RuntimeError(f"learner version mismatch: {published}")
    local_adapter = LOCAL_STAGE / f"adapter-v{expected_version}.pt"
    with local_adapter.open("wb") as stream:
        run(["ssh", LEARNER,
             f"docker exec {LEARNER_CONTAINER} cat {learner_path}"], stdout=stream)
    actor_dir = f"/results/{run_id}/learner"
    actor_path = f"{actor_dir}/adapter-v{expected_version}.pt"
    actor_temp = f"{actor_path}.partial"
    ssh(SPARK, f"docker exec {ACTOR_CONTAINER} mkdir -p {actor_dir}")
    with local_adapter.open("rb") as stream:
        subprocess.run(
            ["ssh", SPARK,
             f"docker exec -i {ACTOR_CONTAINER} sh -c 'cat > {actor_temp}'"],
            stdin=stream, check=True,
        )
    expected_sha256 = hashlib.sha256(local_adapter.read_bytes()).hexdigest()
    verify_code = (
        "import hashlib,os;"
        f"p={actor_temp!r};dst={actor_path!r};expected={expected_sha256!r};"
        "actual=hashlib.sha256(open(p,'rb').read()).hexdigest();"
        "assert actual==expected,(actual,expected);"
        "os.replace(p,dst)"
    )
    ssh(SPARK, f"docker exec {ACTOR_CONTAINER} python3 -c {json.dumps(verify_code)}")
    reset = rpc(SPARK, ACTOR_CONTAINER, {"op": "reset"})
    if "error" in reset:
        raise RuntimeError(f"actor rollout-store reset failed: {reset['error']}")
    loaded = rpc(SPARK, ACTOR_CONTAINER, {"op": "load_adapter", "path": actor_path})
    if "error" in loaded:
        raise RuntimeError(f"actor adapter load failed: {loaded['error']}")
    if (int(loaded["policy_version"]) != expected_version or
            loaded["policy_hash"] != published["policy_hash"]):
        raise RuntimeError(f"actor synchronization mismatch: {loaded} != {published}")
    return published


def prune_learner_payloads(run_id: str, rows: list[dict]) -> None:
    config_ids = sorted({str(row["config_id"]) for row in rows})
    if any(not SAFE_NAME.fullmatch(item) for item in config_ids):
        raise RuntimeError(f"unsafe config ID: {config_ids}")
    targets = " ".join(f"/results/{run_id}/{item}/payloads" for item in config_ids)
    ssh(LEARNER,
        f"docker run --rm -v {LEARNER_ROOT}/results:/results alpine rm -rf {targets}")


def prune_actor_payloads(run_id: str, rows: list[dict]) -> None:
    """Retain rollout metadata/audits while removing multi-GB tensor payloads."""
    config_ids = sorted({str(row["config_id"]) for row in rows})
    if any(not SAFE_NAME.fullmatch(item) for item in config_ids):
        raise RuntimeError(f"unsafe config ID: {config_ids}")
    targets = " ".join(f"/results/{run_id}/{item}/payloads" for item in config_ids)
    ssh(SPARK, f"docker exec {ACTOR_CONTAINER} rm -rf {targets}")


def append_progress(item: dict) -> None:
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    with (LOCAL_STAGE / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(item, sort_keys=True) + "\n")


def cleanup_local(run_id: str, version: int) -> None:
    stage = LOCAL_STAGE / run_id
    if stage.exists():
        shutil.rmtree(stage)
    archive = LOCAL_STAGE / f"{run_id}.tar"
    adapter = LOCAL_STAGE / f"adapter-v{version}.pt"
    if archive.exists():
        archive.unlink()
    if adapter.exists():
        adapter.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=400)
    args = parser.parse_args()
    actor = rpc(SPARK, ACTOR_CONTAINER, {"op": "metrics"})
    learner = rpc(LEARNER, LEARNER_CONTAINER, {"op": "metrics"})
    if (actor["policy_version"], actor["policy_hash"]) != (
            learner["policy_version"], learner["policy_hash"]):
        raise RuntimeError(f"initial actor/learner mismatch: {actor} != {learner}")
    learner_config = rpc(LEARNER, LEARNER_CONTAINER,
                         {"op": "config", "code_rev": "z1-aligned"})
    cfg = learner_config["config"]
    expected = {"lr": 5e-6, "weight_decay": 0.01, "grad_clip": 1.0,
                "clip": 0.2, "kl_coef": 0.0, "update_epochs": 4}
    mismatches = {key: (cfg.get(key), value) for key, value in expected.items()
                  if cfg.get(key) != value}
    if mismatches:
        raise RuntimeError(f"learner is not Z-1 aligned: {mismatches}")
    progress_path = LOCAL_STAGE / "progress.jsonl"
    previous = ([json.loads(line) for line in progress_path.read_text().splitlines()
                 if line.strip()] if progress_path.is_file() else [])
    accepted = sum(item.get("status") == "updated" for item in previous)
    prior_attempts = [int(match.group(1)) for item in previous
                      for match in [re.match(r"grasp-long-a(\d+)-v\d+", item.get("run", ""))]
                      if match]
    first_attempt = max(prior_attempts, default=0) + 1
    for attempt in range(first_attempt, args.max_attempts + 1):
        if accepted >= args.updates:
            break
        version = int(actor["policy_version"])
        run_id = f"grasp-long-a{attempt:04d}-v{version:04d}"
        if not SAFE_NAME.fullmatch(run_id):
            raise RuntimeError(run_id)
        rows = collect(run_id, 720000 + attempt * 100)
        group_outcomes = {}
        for row in rows:
            group_outcomes.setdefault(row["trainer_payload"]["group_id"], []).append(
                bool(row.get("success")))
        informative = [values for values in group_outcomes.values()
                       if any(values)]
        if not informative:
            append_progress({"run": run_id, "status": "homogeneous_skipped",
                             "successes": sum(bool(row.get("success")) for row in rows)})
            prune_actor_payloads(run_id, rows)
            continue
        transfer_run(run_id)
        report = consume(run_id)
        if report.get("status") != "PASS":
            raise RuntimeError(f"learner gate failed: {report}")
        next_version = int(report["update"]["policy_version_after"])
        published = sync_adapter(run_id, next_version)
        prune_learner_payloads(run_id, rows)
        prune_actor_payloads(run_id, rows)
        actor = rpc(SPARK, ACTOR_CONTAINER, {"op": "metrics"})
        learner = rpc(LEARNER, LEARNER_CONTAINER, {"op": "metrics"})
        if (actor["policy_version"], actor["policy_hash"]) != (
                learner["policy_version"], learner["policy_hash"]):
            raise RuntimeError("post-update actor/learner mismatch")
        accepted += 1
        append_progress({
            "run": run_id, "status": "updated", "accepted_update": accepted,
            "policy_version": next_version, "policy_hash": published["policy_hash"],
            "successes": sum(bool(row.get("success")) for row in rows),
            "loss": report["update"]["loss"],
            "grad_norm": report["update"]["grad_norm"],
            "checkpoint": report["checkpoint"],
        })
        cleanup_local(run_id, next_version)
    if accepted < args.updates:
        raise RuntimeError(f"only {accepted}/{args.updates} updates completed")


if __name__ == "__main__":
    main()
