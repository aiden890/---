# Commands used for the measured GPU QA

# Host preflight (reads task-local .env)
JSON=/home/guest/experiments/rlinf-t_43b4edc9/preflight.json bash preflight.sh

# Image build, based on the already verified Xiaomi CUDA image
RLINF_BASE_IMAGE=xiaomi-cu121:t_9f03a613
RLINF_INSTALL_TORCH=0
RLINF_INSTALL_FLASH_ATTN=0
bash run.sh build

# GPU/CUDA/EGL gate
# Executed in rlinf-mibot:t_43b4edc9 with --gpus all and the task asset volume.
# Assertions: torch CUDA available, RTX3090 selected, RLinf imports, EGL vendor contains NVIDIA.

# Paired serial baseline
bash run.sh model-start t43-serial
bash run.sh grid-serial t43-serial-v2 env.horizon=16 rollout.replan_steps=8
bash run.sh model-stop t43-serial

# Planned bounded failure/retry evidence
bash run.sh model-start t43-parallel
bash run.sh grid-parallel t43-parallel validation.inject_fail_once_seed=710000

# Parallel overlap acceptance
bash run.sh grid-parallel t43-parallel-pass env.horizon=16 rollout.replan_steps=8

# One accepted trainer update, checkpoint save, strict reload
bash run.sh grid-consume t43-parallel-pass

# Serial/parallel measured comparison
bash run.sh grid-compare t43-serial-v2 t43-parallel-pass

# Resume validation
# Start t43-resume-test serial collector; after one durable episodes.jsonl row:
docker stop -t 2 rlinf-mibot-collector-t43-resume-test
# Verify no live collector, preserve the exit-137 log, clear only the stale lock, then:
bash run.sh grid-resume t43-resume-test runtime.workers=1 runtime.node_ranks=[0] runtime.max_in_flight=1 env.horizon=16 rollout.replan_steps=8

# Duplicate model-launch validation (expected exit 1)
bash run.sh model-start t43-parallel

# Cleanup
bash run.sh model-stop t43-parallel
# Remove only task-named collector/model containers and task asset overlay volume.

Note: the original parser did not route `env.horizon=16` into the literal grid key `env.horizon`, so measured manifests retained horizon 250; see REPORT.md. The final source fixes this behavior.
