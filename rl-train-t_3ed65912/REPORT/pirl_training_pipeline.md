# pi-RL 훈련 코드 구성 (t_e5cd5736)

목적(왜): operator가 나중에 상세 검토할 수 있도록, faithful pi-RL Flow-SDE가 GRPO
훈련 루프에 어떻게 연결돼 있는지 — 파일 역할, 데이터 흐름, fixed_noise vs pirl 차이 —
를 한 곳에 정리한다. sampler 수식 자체는 `pirl_flow_sde.md` 참조(중복 안 함).

---

## 1. 핵심 파일과 역할

| 파일 | 역할 |
|---|---|
| `rl-env-t_4f3f2b20/src/pirl_flow_sde.py` | faithful pi-RL marginal-preserving Flow-SDE **정의**. `openpi_timesteps()`, `openpi_sigmas()`, `pirl_step_mean_std()`(corrected drift + per-step std), `pirl_transition_logprob()`(diagonal Gaussian logprob), `pirl_flow_sde_sample()`(rollout 샘플러). MiBoT 시간/부호 변환(t_o=1−t_m, v_o=−v_m)이 여기 캡슐화됨. |
| `rl-env-t_4f3f2b20/src/flow_sde.py` | 기존 **fixed_noise baseline**(constant σ=eta·√dt, uncorrected drift). 그대로 보존 — 과거 런 byte-identical 재현용. pi-RL로 부르지 않음. |
| `rl-train-t_3ed65912/src/grpo_trainer_server.py` | GPU **트레이너 서버**(:10088, own container, `--network none`). 모델 로드, per-skill LoRA(DiT qkv_proj, rank8×3skill), `--sampler {fixed_noise,pirl}` 스위치. `op_sample`(rollout 액션+latent+logprob 생성)과 `op_update`(logprob 재계산→ratio→PPO/GRPO 업데이트) 양쪽에서 sampler 분기. |
| `rl-train-t_3ed65912/src/grpo_train_loop.py` | CPU **클라이언트 훈련 루프**(sim client image). rollout 하네스/reward/skill FSM은 env card(`/rl_env/src`)를 import로 재사용(single source of truth). EVAL(eta=0)→TRAIN(eta>0)→EVAL(eta=0)→HELDOUT phase. `TrainerClient`가 서버로 op 요청. |
| `rl-env-t_4f3f2b20/src/reward.py` | reward **정의**. operator 결정(2026-09-16): `use_milestones=False` — 모든 성공 조건 충족 시 최종 한 번 +1.0(마일스톤 분해 폐지). GRASP 20스텝 연속 판정 유지. |
| `rl-train-t_3ed65912/scripts/pirl_g5_sweep.py` | **G5(success)** noise-level 성공률 스윕. `grpo_train_loop`를 import로 재사용(포크 금지). |
| `rl-env-t_4f3f2b20/scripts/pirl_sampler_gpu_gates.py` | sampler/logprob **게이트**(G1/G2/G3/G4/G5-stats/G6/G8). 실 체크포인트 in-process. |
| `rl-train-t_3ed65912/scripts/run-train.sh` | v4 실행 런처: `trainer-start [--sampler pirl ...]`, `train`, `g5-sweep`, `pirl-gates`, `audit-branch`, `sft` 서브커맨드. |

---

## 2. 데이터 흐름 (rollout → stored latent → logprob recompute → ratio → PPO update → DiT LoRA)

```
[클라이언트: grpo_train_loop.train_iteration]
  group 멤버마다 동일 초기 state에서 skill rollout (eta>0):
    _run_one_skill → client.infer(op="sample", eta>0, skill, seed, chunk_index)
        │
        ▼  [서버: grpo_trainer_server op_sample, --sampler pirl]
        build_velocity_field(model, state, ...)  = MiBoT DiT velocity field (LoRA 활성)
        for k in num_steps:                        pi-RL Euler-Maruyama step
            t_o=ts[k]; t_next=ts[k+1]; σ=openpi_sigmas
            m,std = pirl_step_mean_std(x_k, v, t_o, t_next, σ)   # corrected drift
            x_{k+1} = m + std·ε   (ε ~ N(0,I), 시드 결정론)
            xs.append(x_k); pirl_stds.append(std)                # ← STORED LATENT PATH
        old_logp = pirl_transition_logprob(xs, means, pirl_stds, exec_mask)
        반환: action_chunk, {xs_cpu, pirl_stds, old_logp, exec_mask, skill, inputs_cpu}
        │
        ▼  [클라이언트] sim.step(action) → predicate → RewardManager.step_reward
    group return r_i = sim reward (+ train-only shaping: approach/timeout/hold, +1 skill success)
  advantage a_i = (r_i − mean)/std   (group-relative, GRPO)
    │
    ▼  client.update(advantages, op="update", update_epochs≥2)
        [서버: op_update, --sampler pirl]
        for epoch in update_epochs:
          for each stored chunk:
            set_active_skill(wrappers, chunk.skill)        # 해당 skill LoRA만 활성
            means_new = [pirl_step_mean_std(x_k, vfield(x_k, t_m=1−t_o), ...)]  # 미분가능 재계산
            new_logp = pirl_transition_logprob(xs, means_new, pirl_stds, exec_mask)  # ← 저장 std 재사용
            ratio = exp(clamp(new_logp − old_logp))         # epoch0 == 1.0 (on-policy probe)
            loss  = −min(ratio·adv, clip(ratio,1±clip)·adv) + kl_coef·KL_to_base
          loss.backward(); grad_clip; opt.step()            # DiT LoRA(+선택 expert/VLM) 갱신
```

핵심 포인트:
- **stored latent path**: rollout 때 `xs`(각 denoise step 입력)와 `pirl_stds`(시간의존 per-step std)를 저장하고, update의 logprob 재계산에서 **그대로 재사용**. 이래야 epoch0에서 ratio==1이 정확히 성립(fixed_noise는 constant std라 저장 불필요했으나 pirl은 필수).
- **미분 경로**: update의 `means_new`만 `vfield`를 다시 통과(grad 흐름), `xs`/`pirl_stds`는 상수 취급 → grad가 DiT LoRA까지 도달(G6 grad_norm finite 확인).
- **skill 격리**: `set_active_skill`로 해당 skill의 LoRA만 활성 → per-skill adapter가 자기 skill span에서만 학습.

---

## 3. fixed_noise_baseline vs pirl 차이 (식·부호·시간변환)

| 항목 | fixed_noise (baseline) | pirl (faithful) |
|---|---|---|
| σ schedule | constant `σ = eta·√dt` | 시간의존 `σ_i = nl·√(t_o/(1−t_o))` |
| drift | uncorrected `x_k + v·dt` | corrected `x1_weight = t_next − σ_i²·Δ/(2·t_o)` |
| per-step std | constant | 시간의존(저장·재사용 필수) |
| marginal 보존 | ✗ (그렇게 부르면 안 됨) | ✓ (pi-RL/RLinf 식) |
| eta=0 | ODE와 동일 | ODE와 bit-exact(degenerate 단락 처리) |
| 명칭 | fixed_noise_baseline | pi-RL / Flow-SDE / marginal-preserving OK |

MiBoT 어댑터 부분(pirl에서만): OpenPI 식은 t_o(1=noise→0=data), MiBoT DiT는 t_m(0=noise→1=data,
v_m=data-ward). 변환 `t_o=1−t_m`, `v_o=−v_m`. 모든 OpenPI 계수는 t_o로 유지해 분모 0 방지.

---

## 4. 실행 (재현 명령)

```
# 트레이너 서버 (pirl sampler)
cd /home/v4/rl-train-t_3ed65912
bash scripts/run-train.sh trainer-start --sampler pirl --optimizer sgd --lr 2e-3 \
     --train-mode adapter_only --num-steps 5 --eta 0.5 --update-epochs 2

# G5(success) noise-level 성공률 스윕
bash scripts/run-train.sh g5-sweep grasp_g5 --skill grasp \
     --noise-sweep 0.0,0.1,0.3,0.5,0.7,1.0 --n 12 --samples-per-seed 2 --seed-base 5000

# G7(post-update ODE transfer): 짧은 pirl 업데이트 후 eta=0 ODE eval before/after
bash scripts/run-train.sh train g7_grasp --train-skill grasp --eta <G5추천> \
     --iters <N> --eval-n 12 --group 4 --update-epochs 2

# sampler/logprob 게이트 (참고)
bash scripts/run-train.sh pirl-gates --num-steps 5 --noise-level 0.5 --group 4
```

## 5. 정직성 주의
- eta=0 eval은 결정론 ODE(체크포인트와 동일) — before==after면 adapter 미변화가 아니라
  실제 policy delta 0을 의미(adapter는 skill 주어지면 항상 활성; audit fix 반영).
- GRASP는 reset에서 base 성공률이 낮아 eta>0 rollout에서 성공이 드묾 → group-reward 분산
  부재 시 GRPO 신호 없음. G5가 이 여부를 먼저 확인(성공 포장 금지).
