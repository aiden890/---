"""Install at an idle policy boundary; retain the old container for same-version recovery."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


FILES = [
    "src/online_bc/models/pi05_backend.py",
    "src/online_bc/models/phase_policy.py",
    "src/online_bc/models/serve_bc_policy.py",
    "src/online_bc/rollout/worker.py",
    "src/online_bc/rollout/docker_worker.py",
    "src/online_bc/transport/hf_transfer.py",
]

# Runs inside the existing policy image, using saved native observations only.
CLIENT = r'''
import hashlib,json,pickle,sys,urllib.request
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]); mode=sys.argv[2]; version=int(sys.argv[3]); url=sys.argv[4]
if mode=='before':
    samples=[]
    root=Path('/results/coffee-online-bc')
    for manifest in sorted(root.glob('rollouts/round-*/batch-*/cup-dataset/pi05/*/bc/manifest.json'),reverse=True):
        m=json.loads(manifest.read_text())
        for row in m['samples']:
            obs_path=manifest.parent/row['observation']
            if not obs_path.exists():continue
            with np.load(obs_path,allow_pickle=False) as obs:
                sample=dict(obs={k:obs[k].copy() for k in obs.files},prompt=m['prompt'])
            samples.append(dict(sample=sample,seed=533701+len(samples),source=str(obs_path),step=row['step']))
            break
        if len(samples)==3:break
    assert len(samples)==3
    with (out/'requests.pkl').open('xb') as f:pickle.dump(samples,f,protocol=4)
else:
    with (out/'requests.pkl').open('rb') as f:samples=pickle.load(f)
responses=[]; receipts=[]
for sample in samples:
    for variant,phase in [('standard',None),('base_prefix','prefix'),('base_prefix','cup_placement'),('standard',None)]:
        payload=dict(sample=sample['sample'],seed=sample['seed'])
        if variant!='standard':payload.update(variant=variant,skill_phase=phase)
        req=urllib.request.Request(url,data=pickle.dumps(payload,protocol=4))
        with urllib.request.urlopen(req,timeout=180) as response:result=pickle.loads(response.read())
        action=result.pop('actions')
        assert action.shape==(50,12) and np.isfinite(action).all()
        assert result['version']==version and result['variant']==variant and result['skill_phase']==phase
        assert result['effective_policy_version']==(0 if phase=='prefix' else version)
        responses.append(action);receipts.append(result)
with (out/(mode+'.pkl')).open('xb') as f:pickle.dump(responses,f,protocol=4)
report=dict(requests=len(responses),version=version,receipts_verified=True,
            samples=[{k:v for k,v in s.items() if k!='sample'} for s in samples],
            request_sha256=hashlib.sha256((out/'requests.pkl').read_bytes()).hexdigest())
if mode!='before':
    with (out/'before.pkl').open('rb') as f:reference=pickle.load(f)
    report['max_action_error']=max(float(np.abs(a-b).max()) for a,b in zip(responses,reference))
    report['bitwise_equal']=all(np.array_equal(a,b) for a,b in zip(responses,reference))
(out/(mode+'.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report))
'''


def atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2))
    os.replace(temp, path)


def command(argv, log=None, timeout=300):
    # Never include argv/env in an exception or user-visible report.
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Subprocess timed out; inspect the preserved operation state") from None
    if log is not None:
        log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Subprocess failed; inspect the preserved operation log")
    return result.stdout


def health(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def ready(url, version=None, inference=None):
    for _ in range(150):
        try:
            row = health(url)
            if row.get("ready") and row.get("model") == "pi05":
                if version is not None:
                    assert row["version"] == version
                if inference is not None:
                    assert row.get("inference_only", False) is inference
                assert row.get("base_prefix_enabled") is True
                return row
        except OSError:
            time.sleep(2)
    raise RuntimeError("Policy readiness timed out")


def active_sims(root):
    pids = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = path.read_bytes().decode().split("\0")
            if ("online_bc.rollout.run_coffee" in argv
                    and any("/results/coffee-online-bc/" in x or str(root) in x for x in argv)):
                pids.append(int(path.parent.name))
        except (OSError, UnicodeError):
            pass
    return pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    root, stage, out = map(Path, [args.root, args.stage, args.out])
    out.mkdir(parents=True, exist_ok=True)
    singleton = (out / "installer.lock").open("a")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "started.json").exists() or (out / "report.json").exists():
        raise SystemExit("Preserved installation attempt exists; refusing duplicate")
    (out / "pid").write_text(str(os.getpid()))
    sys.path.insert(0, str(root / "src"))
    from online_bc.rollout.worker import active_policy_readers
    from online_bc.rollout.docker_worker import run_args

    config_path = root / "configs/pi05-v4-worker.json"
    template_path = root / "configs/pi05-v4-containers.json"
    lock_path = root / "policy-pi05.lock"
    lock = lock_path.open("a")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        acquired = False
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        readers = active_policy_readers(config_path, lock_path=lock_path) if acquired else []
        sims = active_sims(root) if acquired else []
        if acquired and not readers and not sims:
            break
        if acquired:
            fcntl.flock(lock, fcntl.LOCK_UN)
        atomic(out / "status.json", dict(status="waiting_for_idle_boundary", at=time.time(),
                                        exclusive_acquired=acquired, legacy_readers=readers,
                                        simulator_pids=sims, production_changed=False))
        if time.monotonic() >= deadline:
            atomic(out / "status.json", dict(status="deferred_readers_active", at=time.time(),
                                            production_changed=False))
            return
        time.sleep(5)

    name = "coffee-online-bc-policy-pi05"
    original_name = "coffee-online-bc-policy-pi05-before-inference"
    config = json.loads(config_path.read_text())
    url = config["policy_url"]
    old_health = health(url)
    assert old_health.get("inference_only", False) is False
    assert old_health["model"] == "pi05" and old_health["ready"]
    assert old_health.get("base_prefix_enabled") is True
    current = json.loads((root / "current-adapter.json").read_text())
    version = current["version"]
    assert version == old_health["version"]
    adapter = Path(config["adapter_root"]) / f"round-{version:04d}"
    assert (adapter / "pi05-optimizer.pkl").stat().st_size > 0
    assert json.loads((adapter / "metadata.json").read_text())["step"] == version
    assert json.loads((adapter / "metadata.json").read_text())["model"] == "pi05"
    inspect = json.loads(command(["docker", "inspect", name]))[0]
    assert inspect["State"]["Running"]
    assert "online_bc.models.serve_bc_policy" in inspect["Config"]["Cmd"]
    hc = inspect["HostConfig"]
    assert hc["NetworkMode"] == hc["IpcMode"] == "host"
    assert hc["RestartPolicy"]["Name"] == "no"
    assert not any(hc.get(k) for k in ["Memory", "MemorySwap", "NanoCpus", "CpuQuota",
                                     "CpuPeriod", "CpusetCpus", "CpusetMems", "PidsLimit"])
    assert hc["DeviceRequests"][0]["Count"] == -1
    assert not hc.get("Privileged")
    original_exists = subprocess.run(["docker", "inspect", original_name],
                                     capture_output=True).returncode == 0
    assert not original_exists
    manifest = json.loads((stage / "manifest.json").read_text())
    assert set(manifest) == set(FILES)
    for rel in FILES:
        assert hashlib.sha256((stage / rel).read_bytes()).hexdigest() == manifest[rel]

    backup = out / "backup"
    backup.mkdir(mode=0o700)
    # Full inspect may contain private environment values; keep it private and outside code.
    private = backup / "container-inspect.json"
    private.write_text(json.dumps(inspect))
    private.chmod(0o600)
    for rel in [*FILES, str(config_path.relative_to(root)), str(template_path.relative_to(root))]:
        target = backup / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
    command(["docker", "logs", name], out / "previous-server.log")
    client = out / "client.py"
    client.write_text(CLIENT)
    container_out = "/results/coffee-online-bc/" + str(out.relative_to(root))
    python = inspect["Config"]["Cmd"][0]

    def capture(mode):
        command(["docker", "exec", name, python, container_out + "/client.py",
                 container_out, mode, str(version), url], out / (mode + "-client.log"))
        return json.loads((out / (mode + ".json")).read_text())

    atomic(out / "started.json", dict(at=time.time(), pid=os.getpid(), version=version,
                                      old_container_id=inspect["Id"]))
    stopped = False
    new_created = False
    source_changed = False
    try:
        capture("before")
        atomic(out / "status.json", dict(status="installing_under_exclusive_lock", version=version))
        for rel in FILES:
            temp = (root / rel).with_suffix(".install-tmp")
            shutil.copy2(stage / rel, temp)
            os.replace(temp, root / rel)
        source_changed = True
        template = json.loads(template_path.read_text())
        template[0]["OnlineBCPolicy"] = dict(inference_only=True, enable_base_prefix=True)
        atomic(template_path, template)
        command(["docker", "stop", "--time", "30", name], out / "stop-old.log")
        stopped = True
        command(["docker", "rename", name, original_name], out / "retain-old.log")
        cmd = inspect["Config"]["Cmd"].copy()
        cmd[cmd.index("--adapter") + 1] = current["container_path"]
        if "--inference-only" not in cmd:
            cmd.append("--inference-only")
        command(run_args(inspect) + ["-d", "--name", name, "-w",
                                    inspect["Config"]["WorkingDir"],
                                    inspect["Config"]["Image"], *cmd], out / "start-new.log")
        new_created = True
        candidate_inspect = backup / "candidate-container-inspect.json"
        candidate_inspect.write_text(command(["docker", "inspect", name]))
        candidate_inspect.chmod(0o600)
        new_health = ready(url, version, True)
        after = capture("after")
        assert after["bitwise_equal"] and after["max_action_error"] == 0
        config["inference_only_download"] = True
        atomic(config_path, config)
        command(["docker", "logs", name], out / "new-server.log")
        report = dict(status="installed", at=time.time(), policy_version=version,
                      old_container_id=inspect["Id"], new_container_id=command(
                          ["docker", "inspect", "-f", "{{.Id}}", name]).strip(),
                      health=new_health, before_after=after, source_sha256=manifest,
                      future_filtered_downloads_enabled=True,
                      first_filtered_reload_verified=False, production_gradient_updates=0,
                      same_checkpoint=True, workers=config["workers"],
                      collection_workers=config["collection_workers"],
                      full_optimizer_retained=True)
        atomic(out / "report.json", report)
        atomic(out / "status.json", report)
        print(json.dumps(report))
    except BaseException as error:
        atomic(out / "failure.json", dict(at=time.time(), type=type(error).__name__,
                                          reason=str(error), policy_version=version))
        if stopped and subprocess.run(["docker", "inspect", original_name],
                                      capture_output=True).returncode == 0:
            new_created = subprocess.run(["docker", "inspect", name],
                                         capture_output=True).returncode == 0
        if new_created:
            command(["docker", "logs", name], out / "failed-new-server.log")
            command(["docker", "stop", "--time", "30", name], out / "stop-failed.log")
            command(["docker", "rm", name], out / "remove-failed.log")
        if source_changed:
            for rel in [*FILES, str(config_path.relative_to(root)),
                        str(template_path.relative_to(root))]:
                shutil.copy2(backup / rel, root / rel)
        if stopped:
            if subprocess.run(["docker", "inspect", original_name],
                              capture_output=True).returncode == 0:
                command(["docker", "rename", original_name, name], out / "restore-name.log")
            command(["docker", "start", name], out / "restore-old.log")
            ready(url, inference=False)
            # The retained container may have originally started at an older adapter.
            import pickle
            payload = pickle.dumps(dict(op="load", path=current["container_path"]), protocol=4)
            with urllib.request.urlopen(urllib.request.Request(url, data=payload), timeout=180) as r:
                receipt = pickle.loads(r.read())
            assert receipt["version"] == version
            restored = ready(url, version, False)
            capture("restored")
        else:
            restored = health(url)
        assert restored["version"] == version and restored["ready"]
        assert all((root / rel).read_bytes() == (backup / rel).read_bytes()
                   for rel in [*FILES, str(config_path.relative_to(root)),
                               str(template_path.relative_to(root))])
        report = dict(status="failed_restored", at=time.time(), policy_version=version,
                      health=restored, future_filtered_downloads_enabled=False,
                      full_optimizer_retained=True, failure_preserved=True,
                      source_and_config_restored=True)
        atomic(out / "report.json", report)
        atomic(out / "status.json", report)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
