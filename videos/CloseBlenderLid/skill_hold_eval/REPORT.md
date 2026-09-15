# 스킬 경계 유지(hold) 실험 리포트 — 성공 후 정지 vs 오버피팅 지속

Task: t_53b09e39  (parent t_4a072806)
Task 질문: 각 스킬(grasp / move_holding / place)이 **지시를 완료한 뒤 멈추는가**,
아니면 같은 instruction을 계속 받으면 **관성으로 다음 동작까지 이어서 하는가?**

결론(요약): **모든 스킬에서 policy는 성공 후 멈추지 않고 계속 움직인다.**
성공 판정(predicate)은 대체로 유지되지만 eef는 다음 스킬 방향으로 표류한다.
즉 이 checkpoint는 "스킬 경계"를 인지하지 못하고, 스킬-conditioned instruction을
받아도 obs 관성으로 다음 phase 동작을 이어서 수행한다. **오버피팅/obs-조건화 우세.**

---

## 실험 방법 (재현성 보존)

- `rollouts-xiaomi-t_4a072806/tools/skill_eval.py`에 `--post-success-hold N` 인자 추가.
  - **하위호환**: 기본값 0 = 기존 "성공 즉시 break" 동작 그대로 (생성 스냅샷 에피소드는 영향 없음).
  - N>0: 스킬이 처음 성공한 스텝(`success_step`)에서 break하지 않고, **같은 instruction으로
    N 스텝 더** 실행. 성공은 latch되고(이후 predicate가 떨어져도 success=True 유지),
    각 스텝에 `phase`(pre_success/post_success)와 `success_step`을 로그.
  - git commit `c9c0a1e` (force-add; `rollouts-xiaomi-*/`는 gitignore 대상이나 tools/는 소스).
- v4에서 검증된 설정 그대로: server `xiaomi-server-t_460aea68`(:10086, 중단 없음),
  checkpoint `robocasa-docker-t_9f03a613/checkpoint`, assets `robocasa-assets-t_5af7225b`,
  client image `xiaomi-client:t_9f03a613`, replan 16 / obs 4·2 / crop 0.95 / stride 2 @ 20fps.
- `--post-success-hold 100`으로 여러 seed 실행:
  - grasp: seed 0-8,10,11,12 (grasp만, 생성 불필요) → 12개 중 **9개 성공**
  - move_holding: seed 9 (생성 스냅샷 필요) → **성공**
  - place: seed 9,7,0,1,2,3 → **3개 성공(9,0,1)**

판정 지표(성공시점→끝 100스텝 구간):
- eef 총 이동(mm), 스텝당 평균/최대 이동
- 성공 predicate 유지율(grasp=lid_grasped, move=lid_grasped&in_preplace, place=official_success)
- 다음 스킬로의 표류(grasp: lid z 상승 / move: lid xy→closed 감소 / place: 재접촉·재파지)

판정: eef 총이동 < 20mm 이고 predicate 유지 ≥ 90% → 경계존중(good). 아니면 KEEPS_MOVING.

---

## 결과

### GRASP (9 seeds 성공, 각 성공 후 최대 100스텝 유지)

| seed | success_step | post held | eef 총이동(mm) | eef/스텝(평균/최대) | lid_grasped 유지 | lid z 상승(mm) | 판정 |
|------|----|----|------|------|------|------|------|
| 0  | 155 | 100 | 284.7 | 5.09 / 11.08 | 100% | +180.8 | KEEPS_MOVING |
| 1  | 245 | 55  | 312.5 | 8.81 / 11.84 | 100% | +228.9 | KEEPS_MOVING |
| 2  | 200 | 100 | 393.1 | 5.92 / 12.38 | 97%  | +201.0 | KEEPS_MOVING |
| 3  | 144 | 100 | 280.6 | 5.23 / 13.63 | 100% | +145.1 | KEEPS_MOVING |
| 4  | 125 | 100 | 296.1 | 5.24 / 11.79 | 100% | +175.7 | KEEPS_MOVING |
| 5  | 151 | 100 | 367.2 | 5.32 / 11.91 | 100% | +152.2 | KEEPS_MOVING |
| 6  | 168 | 100 | 301.7 | 5.08 / 12.27 | 100% | +207.2 | KEEPS_MOVING |
| 11 | 202 | 98  | 367.4 | 5.26 / 11.42 | 100% | +179.3 | KEEPS_MOVING |
| 12 | 184 | 100 | 284.4 | 4.48 / 12.78 | 96%  | +193.8 | KEEPS_MOVING |

해석: grasp "집어" 성공 후에도 policy가 멈추지 않고 **lid를 평균 ~18cm 들어올려 다음
move_holding 동작을 이어서 시작**한다. lid_grasped는 대체로 유지되므로 파지 자체는 안정적이나,
"집기만 하고 정지"가 아니라 "집고 곧바로 들어올려 이동"으로 넘어간다.
grasp만으로 다음 스킬(들어올리기)로의 표류가 명확 = 스킬 경계 미인지.

### MOVE_HOLDING (seed 9 성공, 100스텝 유지)

| seed | success_step | post held | eef 총이동(mm) | eef/스텝 평균 | 성공 predicate 유지 | lid xy→closed | 판정 |
|------|----|----|------|------|------|------|------|
| 9 | 76 | 100 | 261.5 | 4.45 | 31.7% | 47.6→3.9mm | KEEPS_MOVING |

해석: pre-place 영역 도달(성공) 후에도 계속 하강·전진하여 **lid를 closed 위치 바로 위(xy 3.9mm)까지
더 내려 place 동작으로 진입**. 성공 predicate(pre-place 영역 내 유지)는 31.7%만 유지 —
영역을 벗어나 place로 넘어갔기 때문. move 성공 후 "정지"가 아니라 "place로 이어짐".

### PLACE (seeds 9,0,1 성공, 각 100스텝 유지) — 가장 중요한 케이스

| seed | success_step | post held | eef 총이동(mm) | eef/스텝(평균/최대) | official_success 유지 | lid 이동(mm) | 최종 상태 | 판정 |
|------|----|----|------|------|------|------|------|------|
| 9 | 160 | 100 | 1123.9 | 15.01 / 37.11 | 100% | 0.0 | on_blender·upright·grip_far 모두 True | KEEPS_MOVING |
| 0 | 101 | 100 | 1363.6 | 15.65 / 35.42 | 100% | 0.0 | 동일 모두 True | KEEPS_MOVING |
| 1 | 156 | 100 | 1251.6 | 13.78 / 31.58 | 100% | 0.0 | 동일 모두 True | KEEPS_MOVING |

해석(중요): place 성공(lid 안착+gripper 후퇴) 후,
- **놓인 lid는 완벽히 그대로 유지**된다: lid 이동 0.0mm, official_success/lid_on_blender/
  upright(≤7°)/gripper_far 모두 100스텝 내내 100% 유지. gripper_lid_contact 재접촉 0%.
  → **다시 집지 않는다. 작업 결과를 망가뜨리지 않는다.**
- 그러나 **eef 자체는 빈 공간에서 1.1~1.4m나 계속 배회**(스텝당 평균 14~16mm, 최대 37mm).
  → policy는 "완료했으니 정지"를 하지 못하고, task가 이미 성공했는데도 gripper를 계속 움직인다.

즉 place는 "결과물은 보존하되 손은 계속 움직이는" 형태의 오버런. 다음 스킬로의 파괴적 표류(재파지)는
없지만, **정지(멈춤) 자체는 실패**.

---

## 종합 판정

- 질문 "성공 후 정지하는가?" → **아니오.** 3개 스킬 전부 성공 후에도 eef를 계속 움직인다
  (grasp 12개 seed 중 9개, move 1개, place 3개 — 성공한 모든 케이스에서 예외 없이 KEEPS_MOVING).
- 질문 "오버피팅/obs-조건화로 다음 동작을 이어서 하는가?" → **그렇다.**
  - grasp 성공 → lid를 ~18cm 들어올림(move 진입)
  - move 성공 → lid를 closed 바로 위까지 하강(place 진입)
  - place 성공 → 손만 배회(결과물은 보존; 재파지는 없음)
- 사용자 가설("스킬 성공 후에도 policy가 관성으로 계속 움직여 다음 스킬로 넘어갈 수 있다") **확인됨.**
  이 checkpoint는 단일 스킬 instruction을 받아도 그 경계에서 멈추지 않고 full-task 관성으로
  다음 phase를 이어서 수행한다 = 스킬-경계 정지가 학습되어 있지 않음.

### RL 시사점
- "성공 후 정지" 능력은 별도 학습/보상 신호가 필요(예: 스킬 성공 후 정지 유지 보너스, 또는
  종료 head). 현재 policy는 스킬 성공을 종료 신호로 사용하지 못함.
- place는 결과 보존은 되므로(재파지 없음), grasp/move의 "다음 phase 표류"가 더 시급한 문제.

---

## 산출물 (SHA256 검증, videos/CloseBlenderLid/skill_hold_eval/)

대표 케이스(성공+100스텝 hold) 로컬 수신, remote↔local SHA256 일치 및 전 프레임 디코드 확인:

| 파일 | 케이스 | decoded frames = recorded |
|------|------|------|
| grasp_hold_seed4.mp4 | GRASP 성공→100 hold | 164 = 164 |
| move_hold_seed9.mp4 | MOVE 성공→100 hold | 105 = 105 |
| place_hold_seed9.mp4 | PLACE 성공→100 hold | 181 = 181 |
| place_hold_seed0.mp4 | PLACE 성공→100 hold | 152 = 152 |

각 mp4에 `*_steps.jsonl`(success_step 표시, 성공 후 predicate 포함) + `*.json` 동봉.
전체 seed 분석은 `hold_analysis_all.json`.

## 재현 커맨드 (v4)
```
export SSH_AUTH_SOCK=$(find /tmp -maxdepth 2 -name 'agent.*'|head -1)
ssh v4@115.145.175.197
cd /home/v4/rollouts-xiaomi-t_4a072806
bash tools/run-skill-eval.sh <run> --skills grasp --post-success-hold 100 --seed <S>
bash tools/run-skill-eval.sh <run> --skills move_holding,place --post-success-hold 100 --seed <S>
python3 tools/analyze_hold.py <run_dir> [...]
```
