# t_700146a2 preflight: v4 environment / GPU / SSH readiness (measured 2026-09-14 12:25-12:33 UTC)

Supersedes BLOCKER.md (attempt 1; approval-gate blocker, now cleared with approvals.mode=off by the user).
Raw outputs: remote/01_probe.txt .. remote/06_source_sha256_remapped.txt. No episode, container change, remote write, upload or download was performed.

## 1. SSH (verified)
- Key: /home/aiden/.ssh/id_ed25519_macbook_aiden, fingerprint SHA256:DaKOzHT793wjBbcsCdkFK1j3jjtkRfUlp//lgHFMMbY ("min", ED25519).
- Agents holding it: /run/user/1000/keyring/ssh (gnome keyring, stable) and /tmp/ssh-XXXXXX0rvlUn/agent.2039712 (temporary; pinned in robocasa-docker/remote.sh).
- Probe `hostname; whoami` -> v4 / v4, exit 0. Host key already trusted (StrictHostKeyChecking=yes passed).
- Exact command (remote/ssh_cmd.txt):
  ssh -o IdentityAgent=/tmp/ssh-XXXXXX0rvlUn/agent.2039712 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=10 -i /home/aiden/.ssh/id_ed25519_macbook_aiden -p 22 v4@115.145.175.197 'CMD'
  If the /tmp agent is gone, use -o IdentityAgent=/run/user/1000/keyring/ssh (verified to hold the same key). Rediscover with: SSH_AUTH_SOCK=<sock> ssh-add -l

## 2. GPU (verified, NOT idle)
- RTX 3090 24576MiB, driver 535.183.01 (CUDA 12.2 max), util 0%, 10488MiB used.
- Only compute process: PID 55248 python3 /work/upstream/deploy/server.py --model /checkpoint --port 10086 = container xiaomi-server-t_460aea68 (Up ~42 min, started 11:44:00Z). Xorg/gnome-shell 15MiB G.
- No other user jobs (who: 0 users; load 0.00). Free VRAM ~14GB: enough for one EGL client alongside the running server (parent runs did exactly this).
- No xiaomi-client-* containers exist any more (operator report of lettuce/steam/lunches clients is stale; they finished). The 5 t_460aea68 episodes are complete.

## 3. Containers / images (verified)
- Running: xiaomi-server-t_460aea68 (a5f3649ffb7f), image sha256:aac5e8e5... = xiaomi-cu121:t_9f03a613, --network none, mounts /home/v4/robocasa-docker-t_9f03a613/checkpoint->/checkpoint, /home/v4/robocasa-docker-t_9f03a613->/work. Model loaded, serving; log shows 5 completed request bursts.
- Exited (do not touch): xiaomi-server-t_5af7225b, xiaomi-server-t_9f03a613, verify/test containers, unrelated vls/isaaclab, mujoco AMC containers.
- Images: xiaomi-cu121:t_9f03a613 aac5e8e59763 15.3GB; xiaomi-client:t_9f03a613 c52e01c1048c 3.89GB; robocasa-sim:t_04bf4fb7. IDs match inference-image-id.txt / client-image-id.txt.
- Asset volumes: robocasa-assets-t_04bf4fb7, -t_5af7225b, -t_9f03a613. t_460aea68 runner reused robocasa-assets-t_5af7225b (writable; upstream writes temp XML). Reuse the same volume; do not run two clients on it concurrently.

## 4. Checkpoint / source identity (verified)
- Model XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365, revision 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4; Xiaomi code 0dd7aef8dc87296246aae812a1f59ccb708e5546; RoboCasa 4f8a2980def75a55dff96b990745b83540425f09 (REPORT.md).
- `sha256sum -c checkpoint.sha256` in /home/v4/robocasa-docker-t_9f03a613: all 20 files OK (3 safetensors shards, 4.99G+5.00G+0.12G).
- source.sha256 (7 files) OK on remote (paths remapped to basenames) and OK locally in robocasa-docker/xiaomi-cu121/. rollout.py sha cd83a86a... identical in parent and /home/v4/rollouts-xiaomi-t_460aea68/rollout.py.
- Runtime stack (image, from REPORT.md): CUDA 12.1.1, torch 2.5.1+cu121, Python 3.10.12, transformers 4.57.1, flash-attn 2.8.3.

## 5. Storage (verified)
- v4 /: 916G, 160G free, inodes 3% used. Local /home/aiden: 146G free. One episode = ~MB-scale MP4 + json; no constraint.

## 6. Tools (verified)
- v4 host: ffmpeg/ffprobe MISSING; rsync, scp present; nvidia runtime registered in docker.
- Decoder: inside xiaomi-client:t_9f03a613 -> imageio 2.37.0 + imageio_ffmpeg bundled ffmpeg 7.0.2 (verified by running the image). Full-decode must run in the client image (parent verify_videos.py pattern), not on the host.
- Local: ffmpeg/ffprobe/imageio/cv2 MISSING; rsync/scp present. Local full decode is NOT available -> report honestly; do remote decode in client image, then copy MP4 + verification json.

## 7. Existing results to reuse (verified, all horizon failures, do not rerun as duplicates)
/home/v4/rollouts-xiaomi-t_460aea68/
  composite-seen/lettuce-01   WashLettuce           seed 22 (base 7+15) horizon 1650 steps 1650 fail
  composite-seen/lunches-01   PackIdenticalLunches  seed 11 (base 7+4)  horizon 3900 steps 3900 fail
  composite-seen/steam-01     SteamInMicrowave      seed 19 (base 7+12) horizon 2100 steps 2100 fail
  composite-unseen/bread-01   BreadSelection        seed 9  (base 7+2)  horizon 1950 steps 1950 fail
  composite-unseen/condiments-01 CategorizeCondiments seed 10 (base 7+3) horizon 1650 steps 1650 fail
Each has config.json, rollout.log, rollout/TASK/stats.json, rollout/summary.json, episode_000_seed_N_failure.mp4. Local copy: robocasa-docker/rollouts-xiaomi-t_460aea68/ (hash-verified per worker comment).
Note: stats seed = base_seed + global_episode_index (task index within the task set), so "seed 7" in config != actual seed; stats.json is authority.

## 8. Dedicated output paths for t_f22ea8b3 (collision-checked: neither exists)
- Remote: /home/v4/rollouts-xiaomi-t_f22ea8b3            (ls -> No such file, 12:31 UTC)
- Local:  /home/aiden/Desktop/lab/robot/robocasa-docker/rollouts/xiaomi-robotics-1-t_f22ea8b3
Create with `mkdir` only after re-checking `test ! -e`; per-attempt dirs use `test ! -e "$dir"` (fail-if-exists). Never write into t_460aea68 / t_5af7225b / parent dirs.

## 9. Runner template (NOT installed on v4; copy = remote write, requires the executing task to do it)
File: run-t_f22ea8b3.sh.template in this directory. Diff vs run-t_460aea68.sh: only root/server/client names and the attempt-dir guard. Same image, checkpoint mount, asset volume, evaluator flags (replan16/history4/interval2/crop0.95/video-stride2/fps20, num-trials 1, split target, original horizon).
Server choice: prefer reusing the running xiaomi-server-t_460aea68 (same image+checkpoint; --network container:xiaomi-server-t_460aea68) to avoid a second 10GB model on the GPU. Do NOT stop or restart it. Only if it is gone, `start` a xiaomi-server-t_f22ea8b3.
Invocation: bash run-t_f22ea8b3.sh episode CATEGORY TASK BASE_SEED UNIQUE_ATTEMPT   (e.g. composite-seen PreSoakPan 7 presoak-01)
Run one client at a time (shared asset volume, GPU headroom ~14GB).

## 10. Record conventions (for t_f22ea8b3)
- root/versions.json: model id+revision, code commit, robocasa commit, image IDs, driver, torch/CUDA, runner sha256.
- root/commands.jsonl (append-only): {utc_start, utc_end, argv, rc, container_name, container_id, task, category, base_seed, attempt}.
- root/heartbeat.jsonl (append-only): {utc, task_id, attempt, container, status, last_step}; plus kanban_heartbeat(task_id=t_f22ea8b3) at start/end and every <=10 min; poll with `docker logs --tail 3 <client>` / `docker ps --filter name=xiaomi-client-t_f22ea8b3`.
- per attempt: config.json, rollout.log (tee), rollout/TASK/stats.json, rollout/summary.json, episode_000_seed_N_{success,failure}.mp4.
- validation.json: per MP4 bytes, sha256, decoded frame count/dims/fps (decoded inside client image into the new root, never overwriting parent evidence).
- Reset errors kept separately (rollout.log + reset_errors.jsonl), not counted as policy episodes.
- Success authority: stats.json success flag; do not infer from video.
- Transfer: deferred until user authorizes; when authorized, rsync -av --ignore-existing then sha256 compare remote/local.

## 11. Verdict
execution_ready: true for one client at a time against the existing server, subject to a pre-launch recheck of nvidia-smi and docker ps (state measured at 12:26-12:33 UTC; server was serving, no client running). Unverified: nothing material remains; local full-decode unavailable (report, not a blocker).
