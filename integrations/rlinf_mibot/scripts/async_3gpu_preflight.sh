#!/usr/bin/env bash
set -euo pipefail

learner="${LEARNER_HOST:-amp_csi}"
actor1="${ACTOR1_HOST:-spark1}"
actor2="${ACTOR2_HOST:-spark2}"

check_host() {
  local host="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
    'printf "host=%s arch=%s " "$(hostname)" "$(uname -m)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
}

echo "[learner]"
check_host "$learner"
echo "[actor1]"
check_host "$actor1"
echo "[actor2]"
check_host "$actor2"

echo "[connectx actor1 -> actor2]"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$actor1" \
  'ping -c 2 -W 1 192.168.200.13 >/dev/null && ping -c 2 -W 1 192.168.201.13 >/dev/null && echo PASS'
echo "[connectx actor2 -> actor1]"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$actor2" \
  'ping -c 2 -W 1 192.168.200.12 >/dev/null && ping -c 2 -W 1 192.168.201.12 >/dev/null && echo PASS'

echo "PASS: three GPUs reachable and both ConnectX paths respond"
