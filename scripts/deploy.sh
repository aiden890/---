#!/usr/bin/env bash
# deploy.sh — 중앙 저장소의 RL 소스 코드를 GPU 서버(v4, amp_csi)로 배포한다.
#
# 사용법:
#   scripts/deploy.sh              # 모든 서버(v4, amp_csi)에 배포
#   scripts/deploy.sh v4           # v4에만
#   scripts/deploy.sh amp_csi      # amp_csi에만
#   scripts/deploy.sh --dry-run    # 실제 전송 없이 변경분만 표시
#
# 배포 대상: RL 소스와 canonical Skill VLA runtime 소스/스크립트.
# 제외: 체크포인트(*.pt), 영상(*.mp4), results/, vendor/, __pycache__ 등 (아래 EXCLUDES).
# 원칙: 코드는 중앙(이 저장소)에서만 고치고, 서버로는 배포만 한다. 서버에서 직접 코드 수정 금지.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- 서버 정의 --------------------------------------------------------------
# v4: remote.sh가 쓰는 살아있는 IdentityAgent 소켓을 그대로 재사용(세션마다 경로가 바뀌므로 추출).
V4_AGENT="$(grep -oE '/tmp/ssh-[^ ]+/agent\.[0-9]+' "$REPO_ROOT/scripts/remote.sh" 2>/dev/null | head -1 || true)"
if [ -n "$V4_AGENT" ] && [ -S "$V4_AGENT" ]; then
  V4_SSH="ssh -o IdentityAgent=$V4_AGENT -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=10 -p 22"
else
  # 폴백: 현재 SSH_AUTH_SOCK 사용
  V4_SSH="ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=10 -p 22"
fi
V4_HOST="v4@115.145.175.197"
V4_BASE="/home/v4"

# amp_csi: ssh config의 Host amp_csi 사용
AMP_SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
AMP_HOST="amp_csi"
AMP_BASE="/home/guest"

# ---- 배포할 코드 디렉토리 ---------------------------------------------------
DIRS=(rl-train-t_3ed65912 rl-env-t_4f3f2b20 architecture_smoke)
TMP_ROOT="$(mktemp -d)"
MANIFEST_DIR="$TMP_ROOT/manifests"
STAGE_DIR="$TMP_ROOT/stage"
mkdir -p "$MANIFEST_DIR" "$STAGE_DIR"
trap 'rm -rf "$TMP_ROOT"' EXIT
for d in "${DIRS[@]}"; do
  python3 "$REPO_ROOT/scripts/build_source_manifest.py" "$d" "$MANIFEST_DIR/$d.json" --commit HEAD
  git -C "$REPO_ROOT" archive HEAD "$d" | tar -x -C "$STAGE_DIR"
done

# ---- rsync 제외 패턴 (코드만, 대용량/생성물 제외) ---------------------------
EXCLUDES=(
  --exclude='__pycache__/' --exclude='*.pyc'
  --exclude='results/' --exclude='REPORT/'
  --exclude='out*/' --exclude='calib_out/' --exclude='runtime_runs/'
  --exclude='vendor/' --exclude='pylibs/' --exclude='*.pt' --exclude='*.pth'
  --exclude='*.mp4' --exclude='*.tar.gz' --exclude='*.pkl' --exclude='*.bak_*'
  --exclude='.git/' --exclude='data/' --exclude='checkpoint*/'
)

DRY=""
TARGETS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY="--dry-run" ;;
    v4|amp_csi) TARGETS+=("$a") ;;
    *) echo "알 수 없는 인자: $a" >&2; exit 2 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(v4 amp_csi)

deploy_one() {
  local name="$1" sshcmd="$2" host="$3" base="$4"
  echo "==== [$name] $host:$base ===="
  for d in "${DIRS[@]}"; do
    [ -d "$d" ] || { echo "  (로컬에 $d 없음, 건너뜀)"; continue; }
    echo "  -> $d/"
    rsync -az $DRY "${EXCLUDES[@]}" \
      -e "$sshcmd" \
      "$STAGE_DIR/$d/" "$host:$base/$d/"
    rsync -az $DRY -e "$sshcmd" \
      "$MANIFEST_DIR/$d.json" "$host:$base/$d/source_manifest.json"
  done
  echo "  [$name] 배포 완료"
}

for t in "${TARGETS[@]}"; do
  case "$t" in
    v4)      deploy_one v4 "$V4_SSH" "$V4_HOST" "$V4_BASE" ;;
    amp_csi) deploy_one amp_csi "$AMP_SSH" "$AMP_HOST" "$AMP_BASE" ;;
  esac
done

echo "== deploy.sh 완료 (${TARGETS[*]}) =="
