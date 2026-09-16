#!/usr/bin/env bash
# collect_and_register.sh (task t_fc5e73d5) -- runs on the LAB machine (aiden).
#
# Polls amp_csi for the INFERENCE.DONE sentinel written by wait-then-run.sh.
# When the obs-only VLM-verifier inference has finished, it:
#   1. rsyncs the result dir (videos + traces + summaries) from amp_csi into the
#      central repo under architecture_smoke/out_vlm_verify/,
#   2. picks a representative seed whose episode has a VLM verifier overlay,
#   3. copies that overlay mp4 into tracking/media/inference/ and git-adds it,
#   4. registers a version entry in tracking/inference_versions.json,
#   5. commits centrally.
# Idempotent: exits early if already registered (marker file). Safe to run from
# cron every ~10 min. Does NOT touch training or the shared server.
set -uo pipefail

REPO=/home/aiden/Desktop/lab/robot/robocasa-docker
PKG_REMOTE=/home/guest/architecture-smoke-t_fc5e73d5
SENTINEL="$PKG_REMOTE/INFERENCE.DONE"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
HOST=amp_csi
MARK="$REPO/architecture_smoke/.registered_vlm_verify"
LOG="$REPO/architecture_smoke/collect_and_register.log"

say(){ echo "[$(date '+%F %H:%M:%S')] $*" >> "$LOG"; }

[ -f "$MARK" ] && exit 0

# sentinel present?
if ! $SSH "$HOST" "test -f $SENTINEL"; then
  exit 0
fi
say "sentinel found on $HOST -> collecting"

# 1. pull results
dest="$REPO/architecture_smoke/out_vlm_verify"
mkdir -p "$dest"
rsync -az -e "$SSH" "$HOST:$PKG_REMOTE/architecture_smoke/out/" "$dest/" \
  || { say "rsync failed"; exit 1; }

# 2. pick a seed with an overlay video (prefer one with >=2 skills executed)
sel_seed=""
for sd in "$dest"/seed*; do
  [ -f "$sd/episode_vlm_verify.mp4" ] || continue
  sel_seed=$(basename "$sd" | sed 's/seed//')
  # prefer the first that has a summary with >=2 skills
  n=$(python3 -c "import json,sys;print(len(json.load(open('$sd/summary.json')).get('skills',[])))" 2>/dev/null || echo 0)
  [ "${n:-0}" -ge 2 ] && break
done
if [ -z "$sel_seed" ]; then
  say "no overlay video found in $dest -- inference may have produced no frames"
  exit 1
fi
say "selected seed $sel_seed"

# 3. copy overlay into tracking/media and force-add (mp4 is gitignored)
mediadir="$REPO/tracking/media/inference"
mkdir -p "$mediadir"
vid="base_vlm_verify_seed${sel_seed}.mp4"
cp "$dest/seed${sel_seed}/episode_vlm_verify.mp4" "$mediadir/$vid"

# 4. register version in inference_versions.json (via python, dedup by id)
python3 - "$REPO/tracking/inference_versions.json" "$sel_seed" "media/inference/$vid" <<'PY'
import json, sys, datetime
path, seed, vidrel = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
vers = d.setdefault("versions", [])
vid_id = "base-vlm-verify"
vers = [v for v in vers if v.get("id") != vid_id]
today = datetime.date.today().isoformat()
vers.append({
    "id": vid_id,
    "label": "학습 전 base · VLM(Qwen3-VL) verify",
    "stage": "pre-training",
    "date": today,
    "desc": ("학습 전 base 정책으로 CloseBlenderLid 인퍼런스. 스킬 종료 판정은 "
             "정책 자신의 frozen Qwen3-VL VQA(P(yes), obs-only: 3-cam+14D proprio, "
             "특권 predicate 미사용) — proprio 이벤트 게이트 + hysteresis latch. "
             "오버레이: skill(GRASP/MOVE/PLACE)·지시문·VLM P(yes)·ADVANCE/CONTINUE/timeout. "
             "PLACE는 단일프레임 VQA 약함(false negative 가능) 정직 표기. seed %s." % seed),
    "video": vidrel,
})
d["versions"] = vers
json.dump(d, open(path, "w"), ensure_ascii=False, indent=2)
print("registered", vid_id)
PY

# 5. commit
cd "$REPO"
git add -f "tracking/media/inference/$vid"
git add tracking/inference_versions.json
git commit -q -m "inference 탭: 학습 전 base · VLM(Qwen3-VL) verify 인퍼런스 영상 등록 (seed ${sel_seed}, t_fc5e73d5)" \
  && say "committed registration for seed $sel_seed" || say "git commit noop/failed"

touch "$MARK"
say "DONE"
