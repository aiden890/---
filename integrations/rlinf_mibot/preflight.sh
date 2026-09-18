#!/usr/bin/env bash
# RLinf-MiBoT host preflight. READ-ONLY probe of the NEW server. Prints actionable
# PASS/FAIL/WARN per gate and exits non-zero if any hard gate fails. NEVER installs or
# updates host drivers, packages, or toolkits — it only inspects and reports.
#
# Usage: bash preflight.sh            (reads .env for mount paths / port / arch)
#        JSON=/path bash preflight.sh (also write a machine-readable report there)
set -uo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

FAIL=0; WARN=0
declare -a RESULTS
rec() { RESULTS+=("$1|$2|$3"); }          # gate|status|detail
pass() { echo "  [PASS] $1: $2"; rec "$1" PASS "$2"; }
warn() { echo "  [WARN] $1: $2"; rec "$1" WARN "$2"; WARN=$((WARN+1)); }
fail() { echo "  [FAIL] $1: $2  -> $3"; rec "$1" FAIL "$2 :: fix: $3"; FAIL=$((FAIL+1)); }

echo "== RLinf-MiBoT host preflight =="

# 0. Architecture / image profile. The pinned CUDA 12.1 digest is the verified x86_64
# default; DGX Spark is normally aarch64 and needs an operator-validated NGC base whose
# torch/flash-attn already support that machine.
arch=$(uname -m 2>/dev/null || true)
if [[ -n "$arch" ]]; then pass "host_arch" "$arch"; else fail "host_arch" "uname failed" "repair host userland"; fi
if [[ -n "${RLINF_EXPECT_ARCH:-}" && "$arch" != "$RLINF_EXPECT_ARCH" ]]; then
  fail "host_arch_match" "expected=$RLINF_EXPECT_ARCH actual=$arch" "run on the intended DGX Spark host"
else
  pass "host_arch_match" "expected=${RLINF_EXPECT_ARCH:-any} actual=$arch"
fi
if [[ "$arch" == "aarch64" ]]; then
  if [[ -z "${RLINF_BASE_IMAGE:-}" || "${RLINF_INSTALL_TORCH:-1}" != "0" ]]; then
    fail "spark_image_profile" "aarch64 cannot use the default x86 CUDA12.1 torch-wheel path" \
      "set RLINF_BASE_IMAGE to the validated DGX Spark NGC PyTorch image and RLINF_INSTALL_TORCH=0"
  else
    pass "spark_image_profile" "base=${RLINF_BASE_IMAGE}; using base-image torch"
  fi
else
  pass "spark_image_profile" "x86-compatible default profile"
fi

# 1. NVIDIA driver + GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [[ -n "$drv" ]]; then
    pass "nvidia_driver" "driver=$drv gpu=$name"
    # CUDA 12.1 runtime needs driver >= 525; warn (not fail) below that.
    major=${drv%%.*}
    if [[ "$major" =~ ^[0-9]+$ ]] && (( major < 525 )); then
      warn "driver_cuda121" "driver $drv may be < 525.60 required by CUDA 12.1 image"
    else
      pass "driver_cuda121" "driver $drv satisfies CUDA 12.1 (>=525)"
    fi
    # compute capability / VRAM
    if [[ -n "$cc" ]]; then
      pass "gpu_compute_cap" "compute_cap=$cc (set RLINF_CUDA_ARCH=${cc/./} if building)"
    else warn "gpu_compute_cap" "could not read compute_cap"; fi
    if [[ "$vram" =~ ^[0-9]+$ ]]; then
      if (( vram < 20000 )); then warn "gpu_vram" "${vram}MiB — MiBoT 5B + adapter needs ~16-22GB; may OOM";
      else pass "gpu_vram" "${vram}MiB"; fi
    fi
  else
    fail "nvidia_driver" "nvidia-smi present but returned nothing" "check driver install / GPU visibility"
  fi
else
  fail "nvidia_driver" "nvidia-smi not found" "install the NVIDIA driver (do NOT let this script do it)"
fi

# 2. Docker
if command -v docker >/dev/null 2>&1 && docker version >/dev/null 2>&1; then
  pass "docker" "$(docker version --format '{{.Server.Version}}' 2>/dev/null)"
else
  fail "docker" "docker daemon not reachable" "install/start Docker and add the user to the docker group"
fi

# 3. NVIDIA Container Toolkit (GPU visible inside a container)
if command -v docker >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi 'nvidia'; then
  pass "nvidia_container_toolkit" "nvidia runtime registered with docker"
elif command -v nvidia-ctk >/dev/null 2>&1; then
  warn "nvidia_container_toolkit" "nvidia-ctk present but runtime not shown in docker info; verify with a --gpus run"
else
  fail "nvidia_container_toolkit" "not detected" "install nvidia-container-toolkit and restart docker"
fi

# Runtime-level device visibility is the actual toolkit gate.
if command -v docker >/dev/null 2>&1 && docker run --rm --gpus all \
    "${RLINF_GPU_PROBE_IMAGE:-nvidia/cuda:12.1.1-base-ubuntu22.04}" nvidia-smi -L >/dev/null 2>&1; then
  pass "container_gpu" "docker --gpus all sees the NVIDIA device"
else
  fail "container_gpu" "GPU probe container failed" "fix nvidia-container-toolkit/runtime before image build"
fi

# 4. Disk / RAM
avail_gb=$(df -Pk "${RLINF_RESULTS:-.}" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
if [[ "$avail_gb" =~ ^[0-9]+$ ]]; then
  if (( avail_gb < 40 )); then warn "disk" "${avail_gb}GB free at results path — image+checkpoint+results want ~40GB+";
  else pass "disk" "${avail_gb}GB free"; fi
fi
ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
if [[ "$ram_gb" =~ ^[0-9]+$ ]]; then
  if (( ram_gb < 16 )); then warn "ram" "${ram_gb}GB — flash-attn build + sim may need more (lower RLINF_MAX_JOBS)";
  else pass "ram" "${ram_gb}GB"; fi
fi

# 5. Ports (trainer RPC)
port="${RLINF_TRAINER_PORT:-10088}"
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}\$"; then
  warn "port" "TCP ${port} already in use — change RLINF_TRAINER_PORT"
else
  pass "port" "TCP ${port} free (or unknown; container uses its own net namespace)"
fi

# 6. Outbound network (needed once to clone RLinf + pull base image, unless pre-seeded)
if command -v curl >/dev/null 2>&1 && curl -fsS -m 8 https://github.com >/dev/null 2>&1; then
  pass "outbound_network" "github.com reachable (RLinf clone + base image pull possible)"
else
  warn "outbound_network" "no outbound to github.com — pre-seed the RLinf clone + base image offline"
fi

# 7. Required mount contents
if [[ -n "${RLINF_CHECKPOINT:-}" && -d "${RLINF_CHECKPOINT}" ]]; then
  if ls "${RLINF_CHECKPOINT}"/model*.safetensors >/dev/null 2>&1 && [[ -f "${RLINF_CHECKPOINT}/config.json" ]]; then
    pass "checkpoint_mount" "${RLINF_CHECKPOINT} has safetensors + config.json"
  else
    fail "checkpoint_mount" "${RLINF_CHECKPOINT} missing model*.safetensors/config.json" "point RLINF_CHECKPOINT at the full Xiaomi checkpoint dir"
  fi
else
  fail "checkpoint_mount" "RLINF_CHECKPOINT not set or not a dir" "set RLINF_CHECKPOINT in .env to the checkpoint path"
fi
if [[ -n "${RLINF_ASSETS:-}" && -d "${RLINF_ASSETS}" ]]; then
  pass "assets_mount" "host directory ${RLINF_ASSETS} present (mode=${RLINF_ASSETS_MODE:-ro})"
elif [[ -n "${RLINF_ASSETS:-}" ]] && command -v docker >/dev/null 2>&1 &&
     docker volume inspect "$RLINF_ASSETS" >/dev/null 2>&1; then
  pass "assets_mount" "Docker volume ${RLINF_ASSETS} present (mode=${RLINF_ASSETS_MODE:-ro})"
else
  fail "assets_mount" "RLINF_ASSETS is neither a host dir nor a Docker volume" \
    "set RLINF_ASSETS to the RoboCasa assets dir or a pre-populated Docker volume"
fi

echo
echo "== preflight summary: FAIL=$FAIL WARN=$WARN =="
if [[ -n "${JSON:-}" ]]; then
  {
    echo '{"fail":'"$FAIL"',"warn":'"$WARN"',"gates":['
    for i in "${!RESULTS[@]}"; do
      IFS='|' read -r g s d <<< "${RESULTS[$i]}"
      sep=,; [[ $i -eq 0 ]] && sep=
      printf '%s{"gate":"%s","status":"%s","detail":%s}' "$sep" "$g" "$s" "$(printf '%s' "$d" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')"
    done
    echo ']}'
  } > "$JSON"
  echo "wrote $JSON"
fi
[[ $FAIL -eq 0 ]] || { echo "preflight FAILED — resolve the [FAIL] gates before bootstrap."; exit 1; }
echo "preflight OK."
