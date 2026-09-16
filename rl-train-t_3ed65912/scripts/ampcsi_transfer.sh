#!/usr/bin/env bash
# Detached transfer: v4 -> amp_csi for the 2-GPU GRPO screening (task t_74a6ed7a).
# Moves the checkpoint, rl-train/rl-env code, skill_eval tools, the two docker images,
# and the robocasa assets volume to amp_csi, then verifies and writes a DONE sentinel.
# Runs detached (nohup) so the Hermes session need not stay open. Idempotent-ish:
# re-running re-syncs (rsync) and reloads images (docker load is fine to repeat).
#
# amp_csi target layout mirrors v4 under /home/guest:
#   /home/guest/robocasa-docker-t_9f03a613/checkpoint
#   /home/guest/rl-env-t_4f3f2b20
#   /home/guest/rl-train-t_3ed65912
#   /home/guest/rollouts-xiaomi-t_4a072806/tools
#   docker images xiaomi-cu121:t_9f03a613, xiaomi-client:t_9f03a613
#   docker volume robocasa-assets-t_5af7225b
set -uo pipefail
KEY=/home/v4/.ssh/csi_gyuri
CSI="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i $KEY -p 10001 guest@115.145.173.248"
RSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i $KEY -p 10001"
DST=guest@115.145.173.248
DSTHOME=/home/guest
LOG=/home/v4/rl-train-t_3ed65912/results/ampcsi_transfer.log
SENT=/home/v4/rl-train-t_3ed65912/results/ampcsi_transfer.DONE
mkdir -p /home/v4/rl-train-t_3ed65912/results
: > "$LOG"; rm -f "$SENT"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "FAILED: $*"; echo "FAILED $*" > "$SENT"; exit 1; }

log "== amp_csi transfer start =="

# 0) target dirs
$CSI "mkdir -p $DSTHOME/robocasa-docker-t_9f03a613 $DSTHOME/rl-env-t_4f3f2b20 $DSTHOME/rl-train-t_3ed65912 $DSTHOME/rollouts-xiaomi-t_4a072806/tools" || fail "mkdir dst"

# 1) checkpoint (9.5G) — rsync
log "rsync checkpoint (9.5G)..."
rsync -a --info=progress2 -e "$RSH" /home/v4/robocasa-docker-t_9f03a613/checkpoint/ \
  "$DST:$DSTHOME/robocasa-docker-t_9f03a613/checkpoint/" >>"$LOG" 2>&1 || fail "rsync checkpoint"

# 2) rl-env card (7.9M) + skill_eval tools (88K)
log "rsync rl-env + skill_eval tools..."
rsync -a -e "$RSH" /home/v4/rl-env-t_4f3f2b20/ "$DST:$DSTHOME/rl-env-t_4f3f2b20/" >>"$LOG" 2>&1 || fail "rsync rl-env"
rsync -a -e "$RSH" /home/v4/rollouts-xiaomi-t_4a072806/tools/ "$DST:$DSTHOME/rollouts-xiaomi-t_4a072806/tools/" >>"$LOG" 2>&1 || fail "rsync skill tools"

# 3) rl-train code (exclude the heavy regenerable results/hf caches)
log "rsync rl-train code (excl results/hf/pycache)..."
rsync -a -e "$RSH" \
  --exclude 'results/' --exclude 'hf_cache/' --exclude 'hf_home/' --exclude 'pylibs/' \
  --exclude '__pycache__/' --exclude '*.pyc' \
  /home/v4/rl-train-t_3ed65912/ "$DST:$DSTHOME/rl-train-t_3ed65912/" >>"$LOG" 2>&1 || fail "rsync rl-train"

# 4) docker images: stream save|load (no intermediate disk)
log "streaming docker image xiaomi-cu121:t_9f03a613 (15.3G)..."
docker save xiaomi-cu121:t_9f03a613 | gzip -1 | $CSI "gunzip | docker load" >>"$LOG" 2>&1 || fail "image cu121"
log "streaming docker image xiaomi-client:t_9f03a613 (3.9G)..."
docker save xiaomi-client:t_9f03a613 | gzip -1 | $CSI "gunzip | docker load" >>"$LOG" 2>&1 || fail "image client"

# 5) assets volume (22.7G): tar from a throwaway container mount, stream, restore into a new volume
log "streaming assets volume robocasa-assets-t_5af7225b (22.7G)..."
$CSI "docker volume create robocasa-assets-t_5af7225b" >>"$LOG" 2>&1 || fail "create assets volume"
docker run --rm -v robocasa-assets-t_5af7225b:/a alpine tar -C /a -cf - . | gzip -1 | \
  $CSI "gunzip | docker run --rm -i -v robocasa-assets-t_5af7225b:/a alpine tar -C /a -xf -" \
  >>"$LOG" 2>&1 || fail "assets stream"

# 6) verify on amp_csi
log "verify on amp_csi..."
$CSI "echo IMAGES:; docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep xiaomi; \
      echo CKPT:; du -sh $DSTHOME/robocasa-docker-t_9f03a613/checkpoint; \
      echo ASSETS:; docker run --rm -v robocasa-assets-t_5af7225b:/a alpine du -sh /a; \
      echo CODE:; ls $DSTHOME/rl-train-t_3ed65912/src/advantage.py $DSTHOME/rl-env-t_4f3f2b20/src/reward.py" \
      >>"$LOG" 2>&1 || fail "verify"

log "== amp_csi transfer DONE =="
echo "OK $(date +%s)" > "$SENT"
