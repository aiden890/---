#!/usr/bin/env bash
# RLinf-MiBoT one-command bootstrap for the NEW GPU server.
#   1. host preflight (fails early on missing driver/docker/toolkit/mounts)
#   2. clone/check RLinf at the pinned commit (into ./vendor/rlinf, gitignored)
#   3. build the pinned Docker image
#   4. validate mounts (checkpoint hash-check optional, assets present)
#   5. run non-GPU-optional smoke checks
# It STOPS before any training. `bash bootstrap.sh train` is the ONLY path that trains.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

# shellcheck disable=SC1091
source ./REVISIONS.lock 2>/dev/null || true   # REVISIONS.lock is KEY=VALUE (and sha lines); grep the pins we need.
RLINF_COMMIT="${RLINF_COMMIT:-$(grep -E '^RLINF_COMMIT=' REVISIONS.lock | cut -d= -f2)}"
RLINF_REPO="${RLINF_REPO:-$(grep -E '^RLINF_REPO=' REVISIONS.lock | cut -d= -f2)}"

step() { echo; echo "==> $*"; }

cmd="${1:-deploy}"

step "[1/5] host preflight"
bash preflight.sh

if [[ "$cmd" == "preflight" ]]; then echo "preflight-only done."; exit 0; fi

step "[2/5] pin RLinf @ ${RLINF_COMMIT} (external dependency, not vendored into git)"
mkdir -p vendor
if [[ ! -d vendor/rlinf/.git ]]; then
  git clone "$RLINF_REPO" vendor/rlinf
fi
git -C vendor/rlinf fetch --all --quiet || true
git -C vendor/rlinf checkout --quiet "$RLINF_COMMIT"
got=$(git -C vendor/rlinf rev-parse HEAD)
[[ "$got" == "$RLINF_COMMIT" ]] || { echo "RLinf HEAD $got != pinned $RLINF_COMMIT"; exit 1; }
echo "  RLinf at $got"

step "[3/5] build image ${RLINF_IMAGE:-rlinf-mibot:latest}"
bash run.sh build

step "[4/5] validate mounts"
if [[ -f checkpoint.sha256 && -d "${RLINF_CHECKPOINT:-/nonexistent}" ]]; then
  echo "  verifying checkpoint file hashes (this can take a minute)..."
  ( cd "$(dirname "${RLINF_CHECKPOINT}")" && \
    sed "s# checkpoint/# $(basename "${RLINF_CHECKPOINT}")/#g" "$OLDPWD/checkpoint.sha256" | sha256sum -c - ) \
    && echo "  checkpoint hashes OK" || echo "  WARN: checkpoint hash mismatch — verify the checkpoint revision"
else
  echo "  (skipping checkpoint hash check; set RLINF_CHECKPOINT and keep checkpoint.sha256)"
fi

step "[5/5] smoke checks (no training)"
bash run.sh smoke

echo
echo "bootstrap complete. Deployment is READY up to the non-training gates."
echo "Next (explicit) commands:"
echo "  bash run.sh ppo-smoke     # tiny adapter-only PPO smoke (one short update)"
echo "  bash run.sh ode-eval      # deterministic ODE before/after eval"
echo "TRAINING never runs from bootstrap; invoke it explicitly."
