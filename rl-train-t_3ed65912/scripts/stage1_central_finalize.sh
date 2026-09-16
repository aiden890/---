#!/usr/bin/env bash
# Central-repo finalizer for Stage-1 (task t_74a6ed7a). Runs DETACHED on the lab box
# (aiden@ the central repo host). Polls v4 for the finisher's STAGE1_FINAL.DONE sentinel,
# then pulls REPORT/STAGE1_FINAL.md + results/stage1/STAGE1_AGG.json into the CENTRAL repo,
# folds the verdict into tracking/stage1.json, and git-commits. Makes the pipeline truly
# self-finishing (results land in the central repo even if the Hermes session ends).
set -uo pipefail
REPO=/home/aiden/Desktop/lab/robot/robocasa-docker
CARD=$REPO/rl-train-t_3ed65912
LOG=$CARD/results/stage1_central_finalize.log
mkdir -p "$CARD/results" "$CARD/REPORT" "$CARD/results/stage1"
: > "$LOG"
log(){ echo "[$(date +%F\ %H:%M:%S)] $*" | tee -a "$LOG"; }

# v4 ssh (same agent-socket trick this task uses)
V4SOCK=$(for s in /tmp/ssh-*/agent.*; do SSH_AUTH_SOCK=$s ssh-add -l 2>/dev/null | grep -qi DaKO && echo "$s" && break; done)
export SSH_AUTH_SOCK="$V4SOCK"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes -i /home/aiden/.ssh/id_ed25519_macbook_aiden v4@115.145.175.197"
SCP="scp -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes -i /home/aiden/.ssh/id_ed25519_macbook_aiden"

log "waiting for v4 STAGE1_FINAL.DONE ..."
ok=0
for i in $(seq 1 5760); do   # up to 16h
  if $SSH 'test -f /home/v4/rl-train-t_3ed65912/results/stage1/STAGE1_FINAL.DONE'; then ok=1; break; fi
  sleep 10
done
[ "$ok" = 1 ] || { log "TIMED OUT waiting for finisher"; exit 1; }
log "finisher done on v4 -> pulling artifacts into central repo"

$SCP v4@115.145.175.197:/home/v4/rl-train-t_3ed65912/REPORT/STAGE1_FINAL.md "$CARD/REPORT/STAGE1_FINAL.md" >>"$LOG" 2>&1 || log "warn: STAGE1_FINAL.md pull failed"
$SCP v4@115.145.175.197:/home/v4/rl-train-t_3ed65912/results/stage1/STAGE1_AGG.json "$CARD/results/stage1/STAGE1_AGG.json" >>"$LOG" 2>&1 || log "warn: STAGE1_AGG.json pull failed"
# per-arm run summaries (small json) for the record
for arm in s1_sgd2e3 s1_adamw3e6 s1_adamw1e5 s1_adamw3e5; do
  mkdir -p "$CARD/results/stage1/$arm"
  $SCP v4@115.145.175.197:/home/v4/rl-train-t_3ed65912/results/stage1/$arm/run_summary.json "$CARD/results/stage1/$arm/run_summary.json" >>"$LOG" 2>&1 || true
  $SCP v4@115.145.175.197:/home/v4/rl-train-t_3ed65912/results/stage1/$arm/eval_before.json "$CARD/results/stage1/$arm/eval_before.json" >>"$LOG" 2>&1 || true
  $SCP v4@115.145.175.197:/home/v4/rl-train-t_3ed65912/results/stage1/$arm/eval_after.json  "$CARD/results/stage1/$arm/eval_after.json"  >>"$LOG" 2>&1 || true
  $SCP v4@115.145.175.197:/home/v4/rl-train-t_3ed65912/results/stage1/$arm/eval_heldout.json "$CARD/results/stage1/$arm/eval_heldout.json" >>"$LOG" 2>&1 || true
done

# fold verdict + per-arm deltas into tracking/stage1.json
python3 - "$CARD/results/stage1/STAGE1_AGG.json" "$REPO/tracking/stage1.json" >>"$LOG" 2>&1 <<'PY'
import json, sys, pathlib
agg_p, track_p = sys.argv[1], sys.argv[2]
track = json.loads(pathlib.Path(track_p).read_text())
try:
    agg = json.loads(pathlib.Path(agg_p).read_text())
except Exception as e:
    print("no agg json:", e); track["verdict"]="finisher produced no aggregate (see REPORT/STAGE1_FINAL.md)"; pathlib.Path(track_p).write_text(json.dumps(track,ensure_ascii=False,indent=2)+"\n"); sys.exit(0)
rows = {r["arm"]: r for r in agg.get("rows", [])}
for a in track["arms"]:
    r = rows.get(a["id"])
    if r:
        a["status"]="done"
        a["grasp_before"]=r.get("grasp_before"); a["grasp_after"]=r.get("grasp_after")
        a["delta_grasp"]=r.get("delta_grasp"); a["grasp_heldout"]=r.get("grasp_heldout")
        a["post_ratio"]=r.get("post_ratio"); a["post_kl"]=r.get("post_kl"); a["post_clip"]=r.get("post_clip")
        a["n_gated"]=f"{r.get('n_gated')}/{r.get('n_iters')}"
    else:
        a["status"]="no-result"
sel = agg.get("selected")
track["verdict"] = (f"SELECT {sel['arm']} (opt={sel['optimizer']} lr={sel['lr']}) paired GRASP Δ={sel['delta_grasp']} post_ratio={sel['post_ratio']}"
                    if sel else "no comparable result")
pathlib.Path(track_p).write_text(json.dumps(track,ensure_ascii=False,indent=2)+"\n")
print("tracking updated; verdict:", track["verdict"])
PY

cd "$REPO"
git add REPORT/STAGE1_FINAL.md rl-train-t_3ed65912/REPORT/STAGE1_FINAL.md rl-train-t_3ed65912/results/stage1 tracking/stage1.json 2>/dev/null
git commit -q -m "Stage-1 GRASP optimizer/LR 스크리닝 최종 결과 반영 (finisher aggregate + tracking verdict) (t_74a6ed7a)" >>"$LOG" 2>&1 && log "committed central results" || log "nothing to commit / commit failed"
log "== central finalize DONE =="
