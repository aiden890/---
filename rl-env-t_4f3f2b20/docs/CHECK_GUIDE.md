# RL 환경 직접 점검 가이드

경로: `rl-env-t_4f3f2b20/`  ·  학습/검증: `rl-train-t_3ed65912/`
점검 순서: **① 리워드 → ② 물리·상태 일관성 → ③ 학습 신호(분산) → ④ 랜덤화 → ⑤ 영상 육안**

명령은 모두 복사해서 로컬 터미널에 붙여넣으면 됩니다. 무거운 GPU 실행은 필요 없고,
대부분 numpy만으로 로컬에서 돕니다.

---

## ① 리워드 정합성 (가장 중요)

리워드가 시뮬레이터 공식 성공 판정과 정확히 일치하는지 확인.

```
cd /home/aiden/Desktop/lab/robot/robocasa-docker/rl-train-t_3ed65912
python3 tests/test_reward_verification.py
```

- 확인 포인트: 마지막 줄 `ALL REWARD-VERIFICATION CHECKS PASSED` + `0 FAILED`
- 핵심 케이스: PLACE에서 뚜껑을 올렸다가 release 때 기울어지면(>7°) 성공 리워드가
  **지급되지 않아야** 함 → `[ok] PLACE tilt-on-release: terminal reward NEVER paid`
- 코드로 직접 확인: `src/reward.py`의 `official_success()`, `SKILL_MILESTONES`,
  `DEFAULT_MILESTONE_BONUS` — 성공 조건은 `lid_on_blender AND gripper_far AND upright`.

## ② 리워드 == 시뮬레이터 판정 (in-sim 대조)

실제 시뮬레이터를 돌려 매 스텝 리워드 성공 플래그와 공식 판정이 어긋나는지 비교한 결과.

```
cd /home/aiden/Desktop/lab/robot/robocasa-docker/rl-train-t_3ed65912
cat results/insim_verify.json
```

- 확인 포인트: `success_flag_vs_sim_disagreements: 0`, `overall_pass: true`
- `determinism.same_seed_lidpos_l2: 0.0` → 같은 seed는 완전히 동일 (재현성 OK)
- `diff_seed_lidpos_l2 > 0` → 다른 seed는 실제로 다른 장면 (랜덤화 작동)

## ③ flow-SDE 학습 신호 (log-prob·gradient)

RL이 미분 가능한 확률적 정책으로 확장됐는지, 학습 신호가 유한한지.

```
cd /home/aiden/Desktop/lab/robot/robocasa-docker/rl-train-t_3ed65912
cat results/eta_grad_verify.json
```

- 확인 포인트:
  - `logprob_finite.executed_all_finite: true` (NaN/발산 없음)
  - `gradient.grad_all_finite: true`, `grad_nonzero: true`, `reaches_dit: true`
    (gradient가 실제 학습 대상까지 흐름)
  - `gradient.ratio_onpolicy: [1,1,1,1]` (importance ratio가 초기엔 1.0 = 올바름)
  - `stochastic.distinct_members: true` (eta>0에서 group rollout이 실제로 갈라짐)

## ④ 랜덤화 기록·재현

매 rollout의 seed/asset/물리 파라미터가 기록되고 train/val/test가 분리됐는지.

```
cd /home/aiden/Desktop/lab/robot/robocasa-docker/rl-env-t_4f3f2b20
python3 -c "import json;d=json.load(open('results/randomization_manifest.json'));print(type(d), len(d) if hasattr(d,'__len__') else '');print(json.dumps(d,indent=1)[:1500])"
cat configs/randomization.json
```

- 확인 포인트: lid pose/yaw, handle 두께·크기, blender pose, camera noise,
  mass/friction/damping 범위가 config에 있고, manifest에 rollout별로 실제 값이 남는지.
- train/val/test seed 구간이 겹치지 않는지.

## ⑤ 영상 육안 확인

리워드가 "성공"이라고 한 순간 영상에서도 실제로 뚜껑이 닫혀 있는지 눈으로 대조.

- 웹에서: http://100.86.183.64:8899/ → RoboCasa Task → CloseBlenderLid ▶ 데모
- 로컬 파일: `rl-env-t_4f3f2b20/results/baseline-run1/*.mp4`
- 각 mp4 옆 `*_steps.jsonl`에 스텝별 predicate가 있어 프레임↔상태 대조 가능.

---

## 통과 기준 요약
- [ ] ① reward 단위 테스트 0 FAILED
- [ ] ② in-sim 불일치 0건, 같은 seed 재현
- [ ] ③ log-prob/gradient 유한, ratio=1.0, group 분기
- [ ] ④ 랜덤화 config·manifest 존재, split 분리
- [ ] ⑤ 성공 판정과 영상 일치

하나라도 이상하면 그 항목 말씀해 주세요 — 같이 파고들게요.
