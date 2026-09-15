# RoboCasa · Xiaomi-Robotics-1 실험 트래킹

> 작업 현황 단일 소스. 이 파일을 수정하면 `index.html`이 그대로 렌더링합니다.
> 최종 업데이트: 2026-09-15

## 프로젝트 목표
Xiaomi-Robotics-1(VLA)을 RoboCasa365 **CloseBlenderLid**에서 skill-conditioned로
평가·강화학습(GRPO)하여 성능을 끌어올린다. 실패를 planning vs 조작으로 구분하고,
재현 가능한 파이프라인과 리워드 레이어를 확보한다.

- 서버: v4 (RTX 3090, CUDA 12.1 컨테이너)
- 체크포인트: XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 (rev 3a6d0293)
- 코드베이스: `/home/aiden/Desktop/lab/robot/robocasa-docker`

---

## 파이프라인 현황

| # | 단계 | 상태 | 산출물 |
|---|------|------|--------|
| 1 | CUDA 12.1 포팅 · rollout 검증 | ✅ 완료 | `xiaomi-cu121/` |
| 2 | 종류별 rollout · 실패 분류 | ✅ 완료 | `rollouts/xiaomi-robotics-1/` |
| 3 | CloseBlenderLid 추가 rollout (seed 7–12) | ✅ 완료 | `rollouts/closeblenderlid-final-t_de8b0c1e/` |
| 4 | skill 3종 분해 rollout (GRASP/MOVE/PLACE) | ✅ 완료 | `videos/CloseBlenderLid/skill_instruction_eval/` |
| 5 | RL 환경 구축 (flow-SDE, reward) | ✅ 완료 | `rl-env-t_4f3f2b20/` |
| 6 | 리워드 정합성 검증 (19/19) | ✅ 통과 | `reward_audit_viewer/` |
| 7 | GRPO pilot 학습 (ARM A/B) | 🔄 진행 | `rl-train-t_3ed65912/` |
| 8 | ARM C · 스케일 수렴 학습 | ⏸ 대기 | — |
| 9 | 디렉토리 구조 개편 · 중복 통합 | 🔍 리뷰 | `docs/MIGRATION.md` |

---

## 핵심 결과

### skill 3종 (지시문 다르게 부여)
| skill | 지시문 | 결과 |
|-------|--------|------|
| GRASP | "Pick up the blender lid securely." | ❌ 빈 gripper로 후퇴 |
| MOVE_HOLDING | "Move the grasped blender lid above the blender..." | ✅ 74스텝, 충돌 없이 도달 |
| PLACE | "Place the ... lid securely on top ..., release ..." | ❌ release 시 7° 기울어져 실패 |

### GRPO pilot (N=50, 동일 seed)
| Arm | 방식 | 공식 성공률 (전→후) | held-out |
|-----|------|--------------------|----------|
| A | adapter만 | 0.20 → 0.16 | 0.08 |
| B | adapter+expert | 0.20 → 0.20 | 0.16 |
| C | +VLM 일부 | 예산상 보류 | — |

> 결론: pilot 규모(30 iter × group 4)에선 **순 성능 향상 없음**. 확보한 것은
> 검증된 재현 가능한 GRPO 파이프라인 + 리워드 레이어. 한계는 샘플 예산.

### 리워드 검증 게이트 (학습 전 필수)
- reward 단위 테스트 **19/19 통과** (13개 그룹) — PLACE tilt-on-release edge case 포함
- flow-SDE eta>0 log-prob·gradient 유한, importance ratio=1.0
- 시뮬레이터 official_check와 0 불일치 (120스텝)

---

## 선례 (RL 방법론)
- **Z-1** (arXiv 2606.31846) — π0.5 flow VLA에 task-wise GRPO, RoboCasa 24 task 80.6%(+13.2%p).
  shared-prefix rollout, tree branching, completion-aware reward, **selective joint training**.
- Flow-GRPO / ReinFlow / πRL — LoRA, tighter clip, ratio guard 권고.

---

## 다음 할 일
- [ ] ARM B 학습 완료 후 arm별 비교표 확정
- [ ] ARM C (adapter+expert+VLM) 실행 (~2h)
- [ ] simulator-only vs simulator+VLM 보조 리워드 ablation
- [ ] 스케일 수렴 학습 (larger group / more iters)
- [ ] 디렉토리 개편 리뷰 반영 (중복 실행 코드 통합)

---

## 열린 이슈 / 블로커
- `t_d4567990` 비교군 실험: blocked (사람 입력 대기)
- GRPO 향상 부재 원인: 샘플 예산 + group 내 리워드 분산 부족 가능성 → 조사 필요
