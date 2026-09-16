# pi-RL Flow-SDE 구현 및 검증 (t_e5cd5736)

목적(왜): RL rollout sampler를 "faithful pi-RL marginal-preserving Flow-SDE"로 교체.
기존 fixed-noise Gaussian sampler(constant sigma=eta*sqrt(dt), uncorrected drift)는
pi-RL / Flow-SDE / marginal-preserving 로 부를 수 없다(operator 지시). 학습 전에
operator가 요구한 sampler/log-prob 게이트를 통과해야 GRPO/PPO로 진행 가능.

## 소스 (memory 재구현 금지, RLinf를 truth로 사용)
- Framework : RLinf/RLinf commit `bde6c918642abf9a4776cb1d5fabcc5087dfe195` (2026-09-16)
- File      : `rlinf/models/embodiment/openpi/openpi_action_model.py`
  - `sample_mean_var_val()` flow_sde 분기 (L1176-1183): corrected drift + sigma schedule
  - `_get_timesteps()` (L1120), `get_logprob_norm()` (L1287), `get_log_prob_value()` (recompute)
- Paper     : pi-RL, arXiv:2510.25889 (Flow-SDE: ODE->SDE, equivalent marginals)

## 이식된 식 (RLinf 그대로, OpenPI 시간축)
- timesteps: `[1, (N-1)/N, ..., 1/N, 0]` (N+1개)
- sigma schedule: `sigma_i = noise_level * sqrt(t_o/(1-t_o))` (t_o=1 분모는 ts[1]로 guard)
- corrected drift: `x1_weight = t_next - sigma_i^2 * delta / (2*t_o)`  ← fixed_noise엔 없는 보정항
- per-step std: `std = sqrt(delta) * sigma_i`  (시간 의존, fixed_noise는 constant)
- transition logprob: diagonal Gaussian `log N(x_{k+1}; mean, std^2)`

## Xiaomi-specific 유일 조정: MiBoT 시간/부호 변환
- OpenPI: t_o 1(noise)->0(data), v_o = noise-ward, denoise가 t_o 감소
- MiBoT : t_m 0(noise)->1(data), v_m=dit_forward = data-ward, Euler `x+v_m*dt`로 t_m 증가
- 관계 : `t_o = 1 - t_m`, `v_o(x,t_o) = -v_m(x, t_m)`
- noise_level=0 degenerate: `x_{k+1} = x_k + v_m*delta` 직접형으로 계산 → 체크포인트 ODE와 bit-exact

## 코드 배치 (중복 없음)
- `rl-env-t_4f3f2b20/src/pirl_flow_sde.py` — 새 faithful sampler (PiRLFlowSDESampler)
- `rl-env-t_4f3f2b20/src/flow_sde.py` — 기존 것은 그대로 두되 "fixed_noise baseline"로 명명(파일 docstring에 이미 명시)
- `rl-env-t_4f3f2b20/tests/test_pirl_flow_sde.py` — CPU 수학 게이트
- `rl-env-t_4f3f2b20/scripts/pirl_sampler_gpu_gates.py` — 실 체크포인트 in-process 게이트
- `rl-train-t_3ed65912/src/grpo_trainer_server.py` — `--sampler {fixed_noise,pirl}` 스위치
  (기본 fixed_noise = 과거 런 byte-identical 재현; op_sample/op_update 양쪽 분기)
- `rl-train-t_3ed65912/scripts/run-train.sh` — `pirl-gates` 서브커맨드

## 검증 결과 (모두 PASS)

### CPU 수학 게이트 (독립 RLinf 참조 구현 대비, tests/test_pirl_flow_sde.py)
| gate | 내용 | 결과 |
|---|---|---|
| G1 | noise=0 == MiBoT Euler | max_abs 2.98e-08 PASS |
| G4 | ported drift/std == 독립 RLinf openpi 참조 (여러 t) | worst_diff 0.0 PASS |
| G4b | sigma schedule 값 (sig0=nl*sqrt(N), std0=nl) | PASS |
| G3 | rollout vs recompute logp, ratio~1 | max_abs 3.81e-06 PASS |
| G6 | logp/std finite 전 noise level | PASS |
| marginal | E[SDE]-ODE 평균 편차 | 0.013 PASS |

### GPU in-process 게이트 (실 pinned 체크포인트, sha256 config=ff3fdf7b…)
| gate | operator 요구항 | 결과 |
|---|---|---|
| G1 | noise=0이 결정론 ODE를 tol내 재현 | **max_abs_diff=0.0 (bit-exact)** PASS |
| G2 | rollout vs recompute old-logp 동일 (동일 latent path) | **max_abs_diff=0.0** PASS |
| G3 | update 전 importance ratio==1 (모든 sample) | **ratio=[1.0,1.0,1.0,1.0]** PASS |
| G4 | corrected drift + g(t) schedule == pi-RL/RLinf 식 | schedule 기록, CPU 단위테스트 diff 0.0 PASS |
| G6 | logp/ratio/KL/grad finite, fail-closed; grad가 DiT 도달 | grad_norm 780.7, finite, reaches_dit PASS |
| G8 | framework commit/식/ckpt hash/precision/steps/noise 기록 | json에 전부 기록 PASS |
- 결과 json: `rl-train-t_3ed65912/results/pirl/pirl_gpu_gates.json`, 재현: `bash scripts/run-train.sh pirl-gates`

### G5 (action 통계, noise level 스윕) — 성공률 스윕은 sim 필요(별도)
noise level 별 ODE 대비 편차/표본 spread (단조 증가 = exploration 정상):
| noise_level | mean_abs_dev_from_ode | sample_spread_std |
|---|---|---|
| 0.1 | 0.0088 | 0.030 |
| 0.3 | 0.0295 | 0.089 |
| 0.5 | 0.0543 | 0.149 |
| 0.7 | 0.0852 | 0.217 |
| 1.0 | 0.1327 | 0.340 |

### 트레이너 end-to-end (sim rollout + op_update, --sampler pirl)
- `pirl_grasp_smoke` (GRASP-from-reset, iters1 group4 update-epochs2):
  `it=0 loss=0.0229 mean_return=0.031 grad_norm=414.0 ratio=1.037 mem=10.49GB`
  → eta>0 pirl rollout + 미분 가능한 recompute + PPO clip/KL 경로 정상 동작 확인.
  epoch0 on-policy ratio≈1, 이후 epoch에서 1에서 벗어나 clip/KL 관여.
- `pirl_move_smoke` (MOVE-from-grasp-entry): eta=0 pirl eval + train loop 정상 실행.
  단, raw seed 5000-5003이 entry-gate(deterministic grasp→grasp-success) 미달로
  ENTRY_FAILED(=sampler 무관, MOVE 본런은 entry-seed prefilter 캐시 사용).

## 재현성 (G8)
- precision bfloat16, denoise_steps 5, RLinf commit bde6c918, ckpt config sha256 ff3fdf7b…
- 재현: `bash scripts/run-train.sh pirl-gates --num-steps 5 --noise-level 0.5 --group 4`
- CPU: `docker run --rm -v <rlenv>:/env -w /env xiaomi-cu121:t_9f03a613 python3 tests/test_pirl_flow_sde.py`

## 남은 것 (다음 단계, operator 요구 게이트 5·7)
- G5(성공률): noise level별 zero-shot task success를 sim에서 측정해 exploration/붕괴 균형 noise 선택.
- G7: 작은 update 후 결정론 ODE sampler로 평가해 개선 전이 확인.
- operator 지시: "official RoboCasa365 PPO path"로 correctness baseline부터, sampler/logprob 게이트 통과 후에만 GRPO 추가.
