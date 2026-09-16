#!/usr/bin/env bash
# pull_stage1.sh — central-side puller for Stage-1 GRASP GRPO screening (task t_c896eb17).
# Pulls STAGE1_AGG.json + STAGE1_FINAL.md from v4 to the tracking page so the live HTML
# (experiments/exp-20260916-stage1-grasp-optlr.html fetches ../stage1_agg.json) updates.
# Cron-safe: uses the persistent v4 ssh-agent socket + passphrase-unlocked key explicitly.
# Also refreshes a small progress line into stage1_agg.json even before the finisher runs,
# by pulling per-arm train logs' latest iter.
set -uo pipefail
AGENT=/tmp/ssh-XXXXXX0rvlUn/agent.2039712
KEY=/home/aiden/.ssh/id_ed25519_macbook_aiden
V4="ssh -o IdentityAgent=$AGENT -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15 -i $KEY -p 22 v4@115.145.175.197"
RSH="ssh -o IdentityAgent=$AGENT -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15 -i $KEY -p 22"
TRACK=/home/aiden/Desktop/lab/robot/robocasa-docker/tracking
RBASE=/home/v4/rl-train-t_3ed65912/results/stage1
RREPORT=/home/v4/rl-train-t_3ed65912/REPORT/STAGE1_FINAL.md
LOG=$TRACK/pull_stage1.log
ts(){ date +%H:%M:%S; }

# 1) If the finisher produced the aggregate, pull it verbatim (authoritative).
if $V4 "test -f $RBASE/STAGE1_AGG.json" 2>>"$LOG"; then
  rsync -a -e "$RSH" "v4@115.145.175.197:$RBASE/STAGE1_AGG.json" "$TRACK/stage1_agg.json" >>"$LOG" 2>&1 \
    && echo "[$(ts)] pulled STAGE1_AGG.json" >>"$LOG"
  $V4 "test -f $RREPORT" 2>/dev/null && rsync -a -e "$RSH" "v4@115.145.175.197:$RREPORT" "$TRACK/STAGE1_FINAL.md" >>"$LOG" 2>&1
  # finisher done -> no more live-progress needed
  exit 0
fi

# 2) Otherwise build a lightweight live-progress stage1_agg.json from arm train logs.
python3 - "$TRACK" <<PY
import json, subprocess, pathlib, re, sys
track = pathlib.Path(sys.argv[1])
AGENT="$AGENT"; KEY="$KEY"
V4=["ssh","-o",f"IdentityAgent={AGENT}","-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=yes",
    "-o","BatchMode=yes","-o","ConnectTimeout=15","-i",KEY,"-p","22","v4@115.145.175.197"]
KCSI="/home/v4/.ssh/csi_gyuri"
# arm -> (server_cmd_prefix, remote train.log path)
arms = {
  "s1_sgd2e3":  (V4, "/home/v4/rl-train-t_3ed65912/results/s1_sgd2e3/train.log"),
  "s1_adamw3e6":(V4, "/home/v4/rl-train-t_3ed65912/results/s1_adamw3e6/train.log"),
  # amp_csi arms: hop v4 -> amp_csi via v4's csi key
  "s1_adamw1e5":(V4+["--"], None),
  "s1_adamw3e5":(V4+["--"], None),
}
def tail_remote(cmd, path):
    try:
        out = subprocess.run(cmd+[f"tail -n 400 {path} 2>/dev/null"], capture_output=True, text=True, timeout=30).stdout
        return out
    except Exception:
        return ""
def amp_tail(path):
    # v4 -> amp_csi hop
    inner = f"tail -n 400 {path} 2>/dev/null"
    hop = (f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "
           f"-i /home/v4/.ssh/csi_gyuri -p 10001 guest@115.145.173.248 \"{inner}\"")
    try:
        out = subprocess.run(V4+[hop], capture_output=True, text=True, timeout=40).stdout
        return out
    except Exception:
        return ""
def parse(log):
    its = re.findall(r"^\[train\] it=(\d+).*?n_succ=(\d+)/(\d+).*?post_ratio=([\d.]+).*?post_kl=([\d.]+).*?post_clip=([\d.]+)", log, re.M)
    befores = re.findall(r"^\[before_\]\s+(\d+)/(\d+)", log, re.M)
    d = {"last_it": None, "n_succ": None, "post_ratio": None, "post_kl": None, "post_clip": None, "before_prog": None}
    if befores:
        d["before_prog"] = f"{befores[-1][0]}/{befores[-1][1]}"
    if its:
        it = its[-1]
        d.update(last_it=int(it[0]), n_succ=f"{it[1]}/{it[2]}",
                 post_ratio=float(it[3]), post_kl=float(it[4]), post_clip=float(it[5]))
    return d
rows=[]
amp_map={
 "s1_adamw1e5":"/home/guest/rl-train-t_3ed65912/results/s1_adamw1e5/train.log",
 "s1_adamw3e5":"/home/guest/rl-train-t_3ed65912/results/s1_adamw3e5/train.log",
}
for arm,(cmd,path) in arms.items():
    if arm in amp_map:
        log = amp_tail(amp_map[arm])
    else:
        log = tail_remote(cmd, path)
    p = parse(log)
    p["arm"]=arm
    rows.append(p)
(track/"stage1_agg.json").write_text(json.dumps({"live_progress": rows, "finalized": False}, indent=2))
print("live progress rows:", [ (r["arm"], r.get("last_it"), r.get("before_prog")) for r in rows])
PY
echo "[$(ts)] refreshed live progress" >>"$LOG"
