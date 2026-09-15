# rl-env-t_4f3f2b20 — CloseBlenderLid skill-conditioned RL 환경

Xiaomi-Robotics-1 로 RoboCasa365 `CloseBlenderLid` 에서 skill-conditioned RL 을 수행하는
Docker 기반 **학습 환경**(환경 구축까지; GRPO 학습 loop 수렴은 후속 카드). 전체 설명·reward 식·
randomization 범위·skill transition predicate·재현 명령은 **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)**.

## 구조 (clean: 코드/설정/결과/문서 분리)

```
rl-env-t_4f3f2b20/
  src/         재사용 코드
    flow_sde.py       flow-SDE 샘플러 + transition log-prob (RL 핵심)
    flow_policy.py    체크포인트 velocity field 어댑터(모델 수정 없음)
    rl_server.py      GPU 측 flow-SDE 추론 서버(:10087, xiaomi-cu121 이미지)
    rl_rollout.py     sim client: 3-skill plan 실행 + reward/log-prob/branch 수집
    reward.py         simulator-primary reward, Z-1 baseline, VLM 보조(=0 기본)
    randomization.py  seed 재현·train/val/test disjoint·validation rule
    skill_manager.py  GRASP/MOVE_HOLDING/PLACE, monitor, oracle & VLM planner
  configs/     pilot_single_gpu.json / scale_multi_gpu.json / randomization.json
  scripts/     run-rl-env.sh (v4 러너), sde_probe.py (GPU 검증)
  tests/       test_flow_sde.py (5), test_env_components.py (13)  — 전부 pass
  docs/        IMPLEMENTATION.md
  results/     probe/ baseline-run1/ rl-run1/ randomization_manifest.json  (검증 증거)
```

## 재사용·중복 금지

- rollout harness 는 `xiaomi-cu121/rollout.py` 를 **import**(복사 X).
- predicate/스냅샷/geometry 는 부모 `rollouts-xiaomi-t_4a072806/tools/skill_eval.py` 를
  **import**(중복 source of truth 없음).
- 체크포인트 modeling 파일은 읽기 전용·해시 고정 → **수정/복사 없이** 로드된 모듈 속성만 사용.
- 공유 서버 `xiaomi-server-t_460aea68` 는 유지, RL 서버만 별도 기동/종료.

## 빠른 확인

```
# v4 원격 셸에서
cd /home/v4/rl-env-t_4f3f2b20
bash scripts/run-rl-env.sh tests     # 단위 테스트 18개
bash scripts/run-rl-env.sh probe     # flow-SDE == 체크포인트(eta=0) 증명
```
