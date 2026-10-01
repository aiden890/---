#!/usr/bin/env bash
# Standalone bootstrap. Reads credentials locally; never sends them to the chat.
set -euo pipefail
work_dir="${PI_CUP_ROOT:-$HOME/pi-cup-online-bc}"
kit_dir="${PI_CUP_KIT:-$HOME/online-bc-kit}"
mkdir -p "$work_dir" "$kit_dir"
mode="${1:---connect}"
case "$mode" in
  --connect|--install-tailscale)
    if test "$(id -u)" -eq 0; then root_cmd=(env); else
      command -v sudo >/dev/null || { printf 'sudo is required for Tailscale SSH.\n' >&2; exit 1; }
      sudo -v
      root_cmd=(sudo)
    fi
    if ! command -v tailscale >/dev/null || ! command -v tailscaled >/dev/null; then
      case "$(uname -m)" in
        x86_64) task_ts_arch=amd64 ;;
        aarch64|arm64) task_ts_arch=arm64 ;;
        *) printf 'Unsupported architecture: %s\n' "$(uname -m)" >&2; exit 1 ;;
      esac
      task_ts_name="tailscale_1.102.4_${task_ts_arch}"
      task_ts_dir="$work_dir/tailscale-binaries"
      mkdir -p "$task_ts_dir"
      python3 - "$task_ts_dir" "$task_ts_name" <<'PY_DOWNLOAD'
import sys, time, urllib.request
from pathlib import Path
folder, name = Path(sys.argv[1]), sys.argv[2]
for suffix, target in [('.tgz', 'tailscale.tgz'), ('.tgz.sha256', 'SHA256')]:
    url = 'https://pkgs.tailscale.com/stable/' + name + suffix
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as source, (folder/target).open('wb') as dest:
                while True:
                    block = source.read(1024*1024)
                    if not block: break
                    dest.write(block)
            break
        except OSError:
            if attempt == 2: raise
            time.sleep(2)
PY_DOWNLOAD
      python3 - "$task_ts_dir" <<'PY_CHECKSUM'
import hashlib, sys
from pathlib import Path
folder = Path(sys.argv[1])
assert hashlib.sha256((folder/'tailscale.tgz').read_bytes()).hexdigest() == (folder/'SHA256').read_text().split()[0], 'Tailscale checksum mismatch'
PY_CHECKSUM
      tar -xzf "$task_ts_dir/tailscale.tgz" -C "$task_ts_dir" --strip-components=1 "$task_ts_name/tailscale" "$task_ts_name/tailscaled"
      "${root_cmd[@]}" install -m 0755 "$task_ts_dir/tailscale" /usr/local/bin/tailscale
      "${root_cmd[@]}" install -m 0755 "$task_ts_dir/tailscaled" /usr/local/bin/tailscaled
    fi
    export PATH="/usr/local/bin:$PATH"
    if test "$mode" = --install-tailscale; then tailscale version; exit 0; fi
    ts_dir="$work_dir/tailscale"
    mkdir -p "$ts_dir"
    socket="$ts_dir/tailscaled.sock"
    if ! "${root_cmd[@]}" tailscale --socket="$socket" status --json >/dev/null 2>&1; then
      "${root_cmd[@]}" bash -c 'umask 077; nohup tailscaled --tun=userspace-networking --state="$1/tailscaled.state" --socket="$1/tailscaled.sock" --port=41641 > "$1/tailscaled.log" 2>&1 < /dev/null & echo "$!" > "$1/tailscaled.pid"' _ "$ts_dir"
      for attempt in $(seq 1 30); do test -S "$socket" && break; sleep 1; done
      test -S "$socket" || { "${root_cmd[@]}" tail -20 "$ts_dir/tailscaled.log"; exit 1; }
    fi
    printf 'Open the login link below in your browser and use the same Tailscale account as Lab.\n'
    "${root_cmd[@]}" tailscale --socket="$socket" up --ssh --accept-dns=false --hostname="${PI_TAILSCALE_HOSTNAME:-skku-vla-gpu}" --timeout=10m
    task_ts_ip="$("${root_cmd[@]}" tailscale --socket="$socket" ip -4)"
    printf '\nTAILSCALE_IP=%s\nSSH_COMMAND=ssh %s@%s\n' "$task_ts_ip" "$(id -un)" "$task_ts_ip"
    printf 'Send the IP and SSH_COMMAND to Codex. Existing HF token can then be copied from Lab.\n'
    printf 'Next stage: bash %s --install\n' "${BASH_SOURCE[0]}"
    exit 0
    ;;
  --install) ;;
  *) printf 'Usage: bash setup-skku-online-bc.sh --connect|--install|--install-tailscale\n' >&2; exit 2 ;;
esac
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-$work_dir/secrets/hf-token}"
if ! test -s "$HF_TOKEN_FILE"; then
  mkdir -p "$(dirname "$HF_TOKEN_FILE")"
  read -r -s -p 'HF token (hidden): ' task_hf_token
  printf '\n'
  (umask 077; printf '%s' "$task_hf_token" > "$HF_TOKEN_FILE")
  unset task_hf_token
fi
if ! command -v uv >/dev/null; then python3 -m pip install --user uv; fi
export PATH="$HOME/.local/bin:$PATH"
test -x "$work_dir/transport-venv/bin/python" || uv venv --python 3.11 "$work_dir/transport-venv"
uv pip install --python "$work_dir/transport-venv/bin/python" 'huggingface_hub>=2,<3'
export PI_BOOTSTRAP_DIR="$kit_dir"
"$work_dir/transport-venv/bin/python" - <<'PY'
import hashlib, json, os, tarfile
from pathlib import Path
from huggingface_hub import sync_bucket
kit=Path(os.environ['PI_BOOTSTRAP_DIR'])
incoming=kit/'download'
sync_bucket('hf://buckets/khmin101/vla-rollout-transfer/pi05-cup-online-bc-20261001/code',str(incoming),token=Path(os.environ['HF_TOKEN_FILE']).read_text().strip(),quiet=True)
archive=incoming/'pi05-cup-online-bc-kit.tar.gz'
expected=json.loads((incoming/'SHA256.json').read_text())['sha256']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected, 'Kit checksum mismatch'
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        name=Path(member.name)
        if name.is_absolute() or '..' in name.parts or not member.isfile():raise ValueError('Unsafe kit member')
    tar.extractall(kit,filter='data')
PY
bash "$kit_dir/scripts/start-pi-cup-learner.sh" setup
bash "$kit_dir/scripts/start-pi-cup-learner.sh" check
bash "$kit_dir/scripts/start-pi-cup-learner.sh" download
printf '\nSetup complete. Learner has not started.\nKIT_DIR=%s\nHF_TOKEN_FILE=%s\n' "$kit_dir" "$HF_TOKEN_FILE"
printf 'Next: export HF_TOKEN_FILE="%s"\nbash "%s/scripts/start-pi-cup-learner.sh" validate\n' "$HF_TOKEN_FILE" "$kit_dir"
