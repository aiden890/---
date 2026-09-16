# RoboCasa · Xiaomi-Robotics-1 실험 트래킹

> 연구 현황 요약. 웹 UI 소스는 역할별로 분리하며 임시 복사본을 저장하지 않습니다.
> 최종 업데이트: 2026-09-16

웹 소스 구조:
- `index.html` — 탭과 본문 구조
- `assets/site.css` — 메인 UI 스타일
- `assets/site.js` — 데이터 로딩과 상호작용
- `experiments.json` — 검색 가능한 실험 목록
- `experiments/*.html` — 실험별 상세 페이지
- `media/experiments/` — 실험 영상

관리 원칙:
- HTML에는 구조만 두고 공통 CSS/JS를 inline으로 중복하지 않는다.
- 새 실험은 `experiments.json` record 1개와 상세 페이지 1개만 추가한다.
- 공통 컴포넌트를 재사용하고 임시 backup·중복 파일을 남기지 않는다.

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

### GRASP horizon 진단 (t_7b40aba0, 2026-09-16)
훈련 루프 GRASP 저성공(G5 base 16.7%)의 원인 = **롤아웃 horizon 절단**, 모델 한계 아님.
훈련 롤아웃 경로(base 폴리시, LoRA zero-init) seed 0~9·split=pretrain, horizon×eta 스윕:

| 샘플러 | horizon 120 (훈련 기본) | horizon 208 (standalone) |
|--------|------------------------|--------------------------|
| eta=0 (결정론) | 0/10 (0%) | 5/10 (50%) |
| eta=0.1 (훈련 실제 롤아웃) | 0/10 (0%) | 4/10 (40%) |
| standalone 레퍼런스 | — | 4/10 (40%) |

- horizon 208의 40%가 standalone 40%와 일치 → 폴리시/샘플러 동일. GRASP 20-hold 완성이 step ~128–136이라 `horizon_grasp=120`이 모든 성공을 잘랐다.
- **조치**: 훈련 `horizon_grasp`를 성공이 담기는 값(≥180, 권장 208)으로 상향. 산출물: 실험결과 탭 EXP-20260916-02(영상 20), `rl-train-t_3ed65912/scripts/grasp_horizon_diag.py`(run-train.sh `diag`).

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

## Skill-conditioned RL 재개 계획 (설계만, 현재 보류)

| 순서 | 내용 |
|---|---|
| 스킬 계약 | `grasp(lid_handle)` → `move_to_closed(blender_lid)` → `release(lid_handle)`; VLA에는 현재 스킬 instruction만 전달 |
| 기준선 | clean/chained 초기 상태에서 스킬 성공률과 instruction 반응성 측정 |
| SFT | instruction 반응이 부족할 때만 스킬별 LoRA; tiny-overfit → heldout → rollout 게이트 |
| RL | simulator predicate 중심 reward로 스킬별 학습; clean 상태에서 chained 상태로 확장 |
| 연결 | handoff-ready와 전체 성공률 평가 후 planner/verifier 연결 |

- 종료와 다음 스킬 전환은 외부 manager가 담당한다.
- VLM은 simulator predicate로 판단하기 어려운 경우의 보조 verifier로만 사용한다.
- 카드 `t_e5cd5736`은 구현 승인 전까지 보류한다.

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
