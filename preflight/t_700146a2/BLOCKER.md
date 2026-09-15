# t_700146a2: blocked preflight, not execution-ready

## Verified locally in this run
Read parent INDEX.md, summary.json version/attempt metadata, run-t_5af7225b.sh, verify_videos.py, xiaomi-cu121/REPORT.md and remote.sh. Workspace is /home/aiden/Desktop/lab/robot. SSH_AUTH_SOCK environment string is /run/user/1000/keyring/ssh; existence, contents and signing are NOT verified. HERMES_KANBAN_TASK was unset, so all board calls use explicit task IDs.

Historical reference only (not live remote verification):
- Model XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365
- Checkpoint revision 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4
- Xiaomi code 0dd7aef8dc87296246aae812a1f59ccb708e5546
- RoboCasa 4f8a2980def75a55dff96b990745b83540425f09
- Inference image xiaomi-cu121:t_9f03a613, sha256:aac5e8e597636dc80df55a3150da68f2fb5897e28bfb02836319578fba477c87
- Client image xiaomi-client:t_9f03a613, sha256:c52e01c1048c46fe0fef0ed535fad3b5067f97b3d5fc397e92a8e96f9702cd19
- CUDA toolkit 12.1.1, torch 2.5.1+cu121, Python 3.10.12, transformers 4.57.1, flash-attn 2.8.3; driver 535.183.01; RTX3090.
- Remote parent /home/v4/robocasa-docker-t_9f03a613; checkpoint subdirectory and checkpoint.sha256 must be verified before reuse.
- Existing experiment outputs /home/v4/rollouts-xiaomi-t_460aea68 must be inventoried before any new rollout.

## Exact blocked command and result
The following terminal request was rejected BEFORE execution; no SSH authentication result was produced:

```sh
python3 -c 'import os,glob,stat,subprocess; paths=[os.environ.get("SSH_AUTH_SOCK","")]+glob.glob("/tmp/ssh-*/agent.*"); [(print("SOCKET",p,flush=True),subprocess.run(["ssh-add","-l"],env={**os.environ,"SSH_AUTH_SOCK":p})) for p in paths if p and os.path.exists(p) and os.stat(p).st_uid==os.getuid() and stat.S_ISSOCK(os.stat(p).st_mode)]' && bash robocasa-docker/remote.sh 'hostname; whoami'
```

Result: exit_code=-1, status=blocked, output empty.
Error: BLOCKED: Command flagged as dangerous (script execution via -e/-c flag) but single-query mode (-q) runs without a user present to approve it.

This is an approval-layer blocker, NOT evidence that SSH keys, v4, GPU, or model are broken. No alternate encoding, wrapper, approval configuration change or transfer was used to bypass it.

## Unverified / do not claim ready
SSH socket ownership/liveness/key fingerprint/signing; live hostname/user; GPU utilization, graphics/compute processes and other users; running/stopped containers; current image IDs/package versions; checkpoint manifest verification; free bytes/inodes; remote output directory collision checks/creation; local and remote copy/encoder/decoder tool availability. No episode, container operation, remote write, upload or download was performed by this task.

Operator previously reported xiaomi-server-t_460aea68 plus xiaomi-client-t_460aea68-lettuce-01, -steam-01, -lunches-01 running despite launcher PID55518 disappearing. That report is NOT current measurement. Do not infer idle GPU or terminated containers. Do not stop them under this inspection-only task.

## Required resolution
Resume in an operator-approved session capable of approving the read-only agent discovery and remote inspection. Do not change security settings as an automated workaround. Rediscover a current-user-owned agent, match the intended public-key fingerprint, and use bounded BatchMode/IdentitiesOnly/StrictHostKeyChecking probes. The existing remote.sh pins a temporary socket and must not be treated as permanent access configuration. Never request a private key or passphrase in chat.

Then record dated raw outputs for hostname/whoami, nvidia-smi including processes, docker ps -a, targeted docker inspect/image inspect, package manifests, checkpoint.sha256 verification, df bytes/inodes, encoder/decoder and transfer-tool checks. Inspect existing experiment stats/logs first. Do not launch clients if any unrelated GPU job is running. Transfer remains deferred by user instruction even if scp is installed.

## Proposed output/record convention (not created remotely)
Local future rollout root: /home/aiden/Desktop/lab/robot/robocasa-docker/rollouts/xiaomi-robotics-1-t_f22ea8b3
Remote future rollout root: /home/v4/rollouts-xiaomi-t_f22ea8b3
These are proposals, NOT collision-checked reservations. Create with fail-if-exists semantics after inventory; suffix a fresh attempt ID rather than overwrite. This report directory is local documentation only.

Keep category/unique-attempt/config.json and rollout.log; evaluator output rollout/TASK/stats.json and episode_000_seed_ACTUAL_RESULT.mp4; append-only commands.jsonl and heartbeat.jsonl; root versions.json and validation.json. Record command argv, UTC start/end, return code, container IDs, image IDs, source/checkpoint revisions, base seed AND stats actual seed, task instruction, original horizon, termination reason, success, steps, video path/bytes/hash/full decoded frame count. Preserve reset errors separately from policy episodes. Fixed simulator seeds do not seed upstream server sampling or guarantee bitwise replay.

The parent shell runner hardcodes its output root/server/client names. DO NOT run it unchanged for new experiments. After approved remote inventory, reuse remote source in a newly scoped runner with only root/container naming changes; retain checkpoint/model, original horizon, evaluator and action parameters. Its invocation shape is:

bash NEW_SCOPED_RUNNER episode CATEGORY TASK BASE_SEED UNIQUE_ATTEMPT

This is a template, not an installed/tested runner. Do not restart the parent server automatically. Parent defaults: split target; task-set category with hyphens replaced by underscores; num-trials 1; replan16/history4/interval2/crop0.95/video-stride2/fps20. Actual seed is base seed plus task index for num-trials=1; use stats as authority. Assets need writable task-owned storage for upstream temporary XML; never modify the parent volume or clone concurrently from actively mutating assets without coordination.

Heartbeat: kanban_heartbeat with explicit executing task ID at task start/end and every few minutes; append UTC/task/attempt/container/status/last-observed-step to heartbeat.jsonl. A heartbeat or process exit is not an episode completion result. Require stats, logs and full MP4 decode.

Decoder reference: parent verify_videos.py fully iterates imageio frames inside the verified client image, checks nonempty output, stores count/dimensions/fps/hash and writes contact sheets. It hardcodes /output and overwrites video-verification.json; only use in a fresh scoped validation output after approval, not over preserved parent evidence. Historical parent decoding is not a current tool check. After transfer is reauthorized, verify remote/local size and SHA256 equality and full local decoding if available; report unavailable local decoding honestly.
