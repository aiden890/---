#!/usr/bin/env bash
# SKKU_BC_STATIC_TAILSCALE_V3: no apt installation; existing token stays local.
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
    # Reuse the previously successful SKKU connector verbatim.
    task_sudo=("${root_cmd[@]}")
    task_name="${PI_TAILSCALE_HOSTNAME:-skku-vla-gpu}"
  task_socket=/var/run/tailscale/tailscaled.sock
  "${task_sudo[@]}" mkdir -p /var/run/tailscale
  if ! "${task_sudo[@]}" tailscale --socket="$task_socket" status --json >/dev/null 2>&1; then
    if pgrep -x tailscaled >/dev/null; then
      echo 'A daemon already exists on another socket or is starting; refusing a duplicate.' >&2; exit 1
    fi
    # Every new daemon gets its own state directory; never share state between live sessions.
    task_state=${TS_STATE_DIR:-/var/lib/tailscale}
    [[ $task_state == /* ]] || { echo 'TS_STATE_DIR must be absolute.' >&2; exit 2; }
    "${task_sudo[@]}" install -d -m 700 "$task_state"
    task_daemon=$(command -v tailscaled)
    "${task_sudo[@]}" sh -c 'nohup "$1" --tun=userspace-networking --port=41641 --socket="$2" --state="$3/tailscaled.state" >> "$3/tailscaled.log" 2>&1 < /dev/null &' sh "$task_daemon" "$task_socket" "$task_state"
    task_ready=0
    for _ in {1..20}; do
      if "${task_sudo[@]}" tailscale --socket="$task_socket" status --json >/dev/null 2>&1; then task_ready=1; break; fi
      sleep 1
    done
    [[ $task_ready == 1 ]] || { echo "Daemon not ready: $task_state/tailscaled.log" >&2; exit 1; }
  fi
  echo 'Authenticate the displayed login URL using the Mac/labdesktop tailnet.'
  "${task_sudo[@]}" tailscale --socket="$task_socket" up --ssh --hostname="$task_name" --accept-dns=false
  task_ip=$("${task_sudo[@]}" tailscale --socket="$task_socket" ip -4)
  task_user=${SUDO_USER:-$(id -un)}
  printf '\nTAILSCALE_IP=%s\nSSH_COMMAND=ssh %s@%s\n\n' "$task_ip" "$task_user" "$task_ip"
  "${task_sudo[@]}" tailscale --socket="$task_socket" ping --c 3 --timeout 2s 100.86.183.64 || true
  echo 'Direct connectivity is not guaranteed. Tailscale SSH access follows tailnet policy.'
    printf 'TS_SOCKET=%s\nTS_STATE_DIR=%s\n' "$task_socket" "${TS_STATE_DIR:-/var/lib/tailscale}"
    printf 'Send TAILSCALE_IP and SSH_COMMAND to Codex. No HF token is needed for this stage.\n'
    exit 0
    ;;
  --install) ;;
  *) printf 'Usage: bash setup-skku-online-bc.sh --connect|--install|--install-tailscale\n' >&2; exit 2 ;;
esac
printf "Use this file with --connect.\n" >&2
exit 2
