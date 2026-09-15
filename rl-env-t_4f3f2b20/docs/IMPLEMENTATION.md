# CloseBlenderLid Skill-Conditioned RL 학습 환경 (task t_4f3f2b20)

Xiaomi-Robotics-1 (MiBoT) 를 RoboCasa365 `CloseBlenderLid` 에서 **skill-conditioned RL**
로 학습할 수 있게 하는 Docker 기반 **환경**입니다. 이 카드는 환경 구축까지이며, 실제 GRPO
학습 loop 수렴은 후속 카드(t_3ed65912 등)에서 진행합니다. 검증된 rollout 코어
(`xiaomi-cu121/rollout.py`, `rollouts-xiaomi-t_4a072806/tools/skill_eval.py`)를 재사용하고,
RL에 필요한 부분(flow-SDE 샘플러 + transition log-prob, skill manager, reward, randomization)만
추가했습니다. **중복 source of truth 없음**: predicate/스냅샷/geometry 는 skill_eval.py 를,
rollout harness 는 rollout.py 를 import 만 합니다.

## 1. 아키텍처 (2-컨테이너 재사용)

로컬에는 GPU 가 없어 모든 GPU 작업은 원격 `v4` (RTX 3090, driver 535.183.01, CUDA 12.1) 에서
실행합니다. 이미지가 역할별로 나뉘어 있습니다:

| 이미지 | 내용 | 역할 |
|---|---|---|
| `xiaomi-cu121:t_9f03a613` | CUDA torch + flash-attn + checkpoint, **robocasa 없음** | 모델 추론 서버(GPU) |
| `xiaomi-client:t_9f03a613` | CPU torch + robocasa + gym, **GPU 모델 불가** | 시뮬레이터 stepping |

따라서 **샘플링(GPU)과 stepping(sim)을 소켓으로 분리**하는 기존 검증된 구조를 그대로 씁니다.
RL 은 `src/rl_server.py`(GPU, :10087)가 flow-SDE 로 action chunk + log-prob 을 만들고,
`src/rl_rollout.py`(sim client)가 소켓으로 받아 env 를 step 합니다. Baseline 은 이미 떠 있는
공유 서버 `xiaomi-server-t_460aea68`(:10086)를 그대로 사용하며 **중지/재시작하지 않습니다**.

## 2. flow-SDE 샘플러 + transition log-prob (RL 핵심)

체크포인트의 action expert 는 배포 시 **결정적 Euler ODE** 를 적분합니다
(`modeling_mibot.py` `ActionExpert.forward`):

```
x_0 ~ N(0,I);  dt = 1/N
for k in range(N):  x_{k+1} = x_k + v_theta(x_k, k/N) * dt
```

RL 을 위해 이를 **Euler–Maruyama SDE** 로 확장합니다 (`src/flow_sde.py`):

```
mu_k    = x_k + v_theta(x_k, t_k) * dt          # drift = 모델의 velocity (그대로)
sigma_k = eta * sqrt(dt)                        # 등방 탐험 노이즈
x_{k+1} = mu_k + sigma_k * eps_k,  eps_k ~ N(0,I)
log p(x_{k+1}|x_k) = sum_dims log N(x_{k+1}; mu_k, sigma_k^2 I)
```

- `eta = 0` 이면 결정적 Euler 샘플러를 **bit-for-bit 재현** (probe 에서 max_abs_diff=0.0 확인).
- drift 는 바꾸지 않으므로 학습된 동역학은 유지, `eta` 만 탐험/entropy 를 조절.
- 모든 transition 이 Gaussian → log-prob 이 정확하고 theta 에 대해 미분 가능(정책 그래디언트 recompute).
- **GRPO masking**: 실제 env 에 실행된 timestep/dimension(첫 `replan_steps` 행 × 실제 `action_dim` 열)
  만 loss 에 포함(`make_executed_mask`, `executed_logprob`). 초기 노이즈 log p(x_0)는 별도 저장(그룹 내
  공유 conditioning 에서 상쇄되지만 감사 가능하게 보존).

`build_velocity_field`(`src/flow_policy.py`)가 체크포인트 forward 의 conditioning 준비(VLM
forward, position/attention, state_embed)를 그대로 재현하고 모델 자신의 `dit_forward` 를
velocity field 로 반환합니다. 체크포인트 파일은 읽기 전용·해시 고정이라 **수정/복사하지 않고**
로드된 모듈의 공개 속성만 사용합니다. upstream forward 가 바뀌면 probe(eta=0 동일성)가 실패해
drift 를 즉시 노출합니다.

## 3. Reward 원칙 (`src/reward.py`)

계층이 코드로 강제됩니다:

- **PRIMARY = 시뮬레이터 ground-truth predicate 만.** 공식 terminal 성공 =
  `lid_on_blender == True` AND gripper–lid 거리 > 0.15 m (그리고 lid upright). VLM 은 primary 에
  **절대 들어가지 않음**.
- **SKILL milestone (episode 당 최대 1회):** stable grasp, lid lifted, collision-free transfer,
  pre-place reached, lid seated, released, retreated. 경계에서 진동해 farming 하는 것을 막기 위해
  최초 True 될 때 한 번만 지급.
- **BASELINE (Z-1 호환):** terminal 1/0 + success-aware decay `gamma = 0.998`
  (`reward = gamma^step`), Z-1 baseline 과 직접 비교 가능하게 유지.
- **VLM = 보조 진단 score 만.** 기본 weight 0. `simulator+vlm` ablation 에서만 작은 shaped bonus 로
  가산 → reward-only ablation 이 순수 **config 스위치**(`RewardConfig.with_vlm`).
- **penalty:** object drop, disallowed collision, timeout 기록.

predicate 키는 `skill_eval.py` `Sim.predicates()` 와 정확히 일치시켜 그 dict 를 그대로 소비합니다.

## 4. Randomization (`src/randomization.py`, `configs/randomization.json`)

`(split, episode_index)` 만으로 씬이 **정확히 재현**됩니다(마스터 seed 의 안정적 해시로 RNG 시드).
드로우된 전체 파라미터를 rollout 마다 직렬화(`to_metadata`), `from_metadata` 로 동일 씬 복원.

| 축 | 범위(pilot) |
|---|---|
| lid 초기 XY/Z/yaw | XY ±0.04 m, Z 0–0.03 m, yaw ±30° |
| blender pose | XY ±0.03 m, yaw ±15° |
| robot base / eef | base XY ±0.05 m·yaw ±10°, eef XYZ ±0.02 m |
| camera | pos ±0.01 m, rot ±1°, fov ±1° (calibration noise) |
| lighting | intensity ×0.8–1.2, ambient ×0.7–1.3 |
| physics | lid mass ×0.85–1.15, friction ×0.8–1.2, contact damping ×0.8–1.25 |
| clutter | distractor 0–2 개 |

**train / validation / test 분리**: lid handle geometry(asset variant)와 scene seed 는
train↔test **disjoint** 풀(코드에서 `splits_disjoint()`가 True 확인) → 학습 미사용 geometry·seed 로
held-out 평가 준비 완료. **validation rule** 로 비현실적 조합 제외 후 재샘플:
lid z < rest, 음수 physics scale, 저질량×고마찰 불안정, blender+base 도달범위 초과(합 > 0.12 m),
음수 lighting. (150 씬 샘플 전부 재샘플 후 valid 확인.)

## 5. Skill manager (`src/skill_manager.py`)

고수준 skill 3종: `GRASP(object)`, `MOVE_HOLDING(object, dest, constraints)`,
`PLACE(object, dest, relation)`. planner 가 이들을 조합, skill monitor 가 predicate 로
success/failure 이유/timeout 판정, 실패→다음 skill(recovery).

- CloseBlenderLid 기본 plan: `GRASP(blender_lid) → MOVE_HOLDING(blender_lid, above_blender,
  collision_free) → PLACE(blender_lid, blender, securely_on_top)`.
- **복구 규칙:** object_dropped→GRASP 재시작; move 중 alignment 상실→MOVE_HOLDING 재시작;
  release 후 lid 목표 밖→GRASP 재시작; lid_on_blender 이거나 gripper 가까움→PLACE 내 retreat 지속.
- **OraclePlanner**: predicate 기반 결정적 → 환경(executor/monitor/reward/recovery) 자체를
  end-to-end 로 검증하는 ground truth.
- **VLMPlannerStub**: 실제 VLM planner 를 위한 pluggable 인터페이스로, oracle 과 **별도 평가**.
  backend 를 주입하면 카메라+goal 을 읽어 구조화된 SkillCall 을 내는 실제 planner 로 교체.

skill transition predicate: stable grasp(`lid_grasped`), lid lifted(`lid_lifted`),
grasp maintained, collision-free transfer(`lid_grasped` ∧ no other contact),
pre-place reached(`in_preplace_region`), lid seated(`lid_on_blender`),
released(¬contact ∧ on_blender), retreated(`gripper_lid_far_0.15` ∧ on_blender).

## 6. 검증 증거 (v4 에서 실제 실행)

| 항목 | 결과 |
|---|---|
| 단위 테스트 flow-SDE 5개 | 전부 pass (`tests/test_flow_sde.py`) |
| 단위 테스트 env components 13개 | 전부 pass (`tests/test_env_components.py`) |
| GPU probe: flow-SDE eta=0 == 체크포인트 | **max_abs_diff 0.0**, group log-prob finite·distinct·recompute 일치 (`results/probe/probe_report.json`) |
| Baseline end-to-end (deterministic, seed 9) | 8 skill call, 오라클 recovery 동작, reward 집계, 영상/predicate/step log 저장 (`results/baseline-run1/`) |
| RL end-to-end (flow-SDE eta=0.6, log-prob 저장) | chunk 마다 `executed_logprob` 저장(≈-59.4), branch state npz(MuJoCo qpos) 저장 (`results/rl-run1/`) |
| Randomization | train5 재현 True, train/test disjoint True (`results/randomization_manifest.json`) |
| 공유 서버 | `xiaomi-server-t_460aea68` 내내 유지, RL 서버만 별도 기동·종료, GPU 10.5→20.9→10.5 GB |

end-to-end 성공 여부(정책 성능)는 이 카드 범위가 아님. 이 seed 에서 결정적 정책의 GRASP 는
대부분 실패(부모 t_4a072806 와 일치) — 환경이 성공/실패/복구를 **정확히 판정**함을 보이는 것이 목적.

## 7. 재현 명령 (v4)

SSH: 키 `id_ed25519_macbook_aiden`, agent 소켓 `/tmp/ssh-XXXXXX0rvlUn/agent.2039712`,
대상 `v4@115.145.175.197`. (아래는 원격 셸 기준.)

```
cd /home/v4/rl-env-t_4f3f2b20

# 단위 테스트 (client 이미지: CPU torch + numpy)
bash scripts/run-rl-env.sh tests

# GPU probe: flow-SDE == 체크포인트(eta=0) + log-prob sanity
bash scripts/run-rl-env.sh probe

# Baseline (결정적, 공유 서버 :10086 사용, 서버 유지)
bash scripts/run-rl-env.sh baseline run1 9

# RL (flow-SDE + log-prob). RL 서버 기동 -> rollout -> 서버 종료
bash scripts/run-rl-env.sh rl-server-start
bash scripts/run-rl-env.sh rl run1 9 0.6 --horizon-grasp 120 --horizon-move 120 --horizon-place 150 --max-skill-calls 3
bash scripts/run-rl-env.sh rl-server-stop
```

config: `configs/pilot_single_gpu.json`(3090 저메모리, group_size 4, VLM frozen,
action expert/skill adapter/termination head 만 학습) / `configs/scale_multi_gpu.json`(다중 GPU
확장, group_size 16) / `configs/randomization.json`(범위·asset 풀·validation rule).

## 8. 후속(GRPO 학습 카드)로 넘기는 것

이 환경이 produce 하는 데이터: 그룹 rollout 의 transition log-prob(정책 ratio 용), branch state
(MuJoCo qpos/qvel + controller + obs history + current skill; `capture_branch_state`),
executed-action mask, reward breakdown. 후속 카드는 각 skill 시작점에서 shared-prefix branch 를
group_size 만큼 복제하고, 실행된 timestep/dim 만 GRPO loss 에 포함해 action expert/skill adapter/
termination head 를 업데이트(VLM backbone frozen)합니다. 실제 loop 수렴은 그 카드 범위입니다.
