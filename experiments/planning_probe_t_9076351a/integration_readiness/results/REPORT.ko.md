# Generic planner 단일-GPU integration readiness

## 최종 판정

Enum 판정은 `INCONCLUSIVE`, production integration hard gate는 명시적으로 `FAIL/NOT READY` — production 통합 금지.

한 process 안에서 Xiaomi action policy와 `Qwen/Qwen3-VL-4B-Instruct@ebb281...`를 RTX3090 한 장에 함께 올리고 CUDA 호출을 직렬화하는 후보 A는 메모리와 policy-idle 성능 gate를 통과했다. 하지만 현재 production server/rollout 코드에는 policy forward 또는 control-loop의 명시적 deadline이 없다. 또한 동기식 planner 호출 1회 동안 action loop가 3.300초 멈춘다. 따라서 measured p95가 기존 예산을 충족하는지 판정할 수 없으며, task의 fail-closed 규칙에 따라 production integration은 FAIL/NOT READY다.

## 실측

- 단일 process / 단일 visible GPU: PASS. 실행 직전 compute process 0, 실행 중 nvidia-smi compute row는 동일 host PID 1개뿐이었다. Docker PID namespace 안에서는 Python이 PID 1이었다.
- OOM / NaN / RPC failure: 0.
- peak reserved VRAM: 18,700 MiB.
- reserved 기준 headroom: 5,422.2 MiB (2 GiB gate PASS).
- Xiaomi-only 12회: p50 253.770 ms, p95 256.262 ms, max 256.419 ms, 3.935 Hz.
- dual-loaded/planner-idle Xiaomi 12회: p50 253.266 ms, p95 254.668 ms, max 255.333 ms, 3.946 Hz.
- throughput 변화: +0.259% (report의 degradation은 -0.259%); ≤10% gate PASS.
- planner boundary 1회: 3,299.575 ms.
- planner 직후 policy: 255.337 ms, finite output.
- warmup: baseline 1,140.527/255.055 ms, dual 268.572/254.251 ms.

## Planner 재확인 범위

부모 canonical case 중 seed0/instruction0 한 건만 재실행했다. 모델의 direct semantic JSON은 유효했고 `GRASP_OBJECT → MOVE_OBJECT → PLACE_OBJECT`, name/args validity, deterministic canonicalization, strict 5-field plan이 모두 통과했다. 이는 부모의 12-case accuracy를 확장하는 새 accuracy 주장이 아니다.

## Scheduler 의미

후보 A는 planner와 policy CUDA 호출을 한 thread에서 동기 직렬화한다. planner가 실행되는 3.300초 동안 action step은 처리되지 않고 queue도 생성되지 않으므로 무제한 backpressure는 없다. 그러나 이 pause가 skill boundary에서 허용되는지, 또는 기존 action chunk를 계속 소비해야 하는지 production 계약이 없다. 먼저 explicit deadline과 boundary behavior를 정의한 뒤 동일 probe로 재판정해야 한다.

## 보존 및 정리

`report.json`은 model revision, process inventory, gate, raw semantic output, canonical strict plan, 전체 VRAM timeline을 보존한다. `latency.csv`, `vram_timeline.csv`, `remote_run.log`, `cleanup_and_inventory.txt`, `provenance.json`을 함께 보존했다. 종료 gate는 experiment container 0, ports 10086/10087 listener 0, compute process 0, GPU 38 MiB/0%였다. Production runtime과 v4 Arm A는 변경하지 않았다.
