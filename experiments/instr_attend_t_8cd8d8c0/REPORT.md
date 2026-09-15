# 추론 instruction-attendance 검증 (t_8cd8d8c0)

**질문:** RoboCasa365 정책이 언어 instruction에 실제로 어텐드하는가, 아니면 obs만 보고 지시를 무시하는가?
**배경:** hold 실험(t_53b09e39)에서 skill 부분지시를 줘도 성공 후 다음 동작으로 계속 이동 → "instruction 무시, obs 조건화" 의심.

## 실험 설계 (obs 고정, instruction만 변경)

핵심 아이디어: **동일한 초기 MuJoCo state(obs)를 고정**하고 **instruction만 바꿔** 액션을 반복 샘플링한다.
Xiaomi 체크포인트는 flow x0를 **전역 unseeded RNG**에서 뽑으므로 같은 입력이라도 액션이 매번 조금씩 다르다 → **샘플링 노이즈 바닥(noise floor)** 이 존재한다. 따라서:

- `within(I)` = 같은 instruction N회 샘플의 평균 pairwise L2 (노이즈 바닥)
- `between(A,B)` = instruction A 평균 액션청크 vs B 평균 액션청크의 L2
- 귀무가설(instruction 무시)에서 두 평균의 기대차이 ≈ noise_floor/√N. 이 스케일 대비 `between`이 크면 어텐드.
- **순열검정(permutation test, 20k iters, 192-D 전체 액션청크)**: 두 instruction 라벨을 섞어 평균차이 분포를 만들고 관측 평균차이의 p값 산출.

- **고정 obs 3종** (skill_eval.Sim snapshot/restore 재사용):
  - `reset`: 공식 CloseBlenderLid reset(seed 9) — lid가 counter 위, task 시작 전.
  - `move`: generation snapshot — lid 잡고 들어올린 상태(move_holding 시작).
  - `place`: generation snapshot — 잡은 lid가 blender 위 pre-place 영역(place 시작).
- **instruction 배터리 9종**: 정답(full), skill 3종(grasp/move/place), 반대(open drawer), 무관(pick cup / turn on stove), degenerate(빈 문자열 / 무의미 토큰).
- N=24 샘플/instruction, 액션청크 16×12=192-D 전체로 비교.

코드는 검증된 `/work/rollout.py`(EvalClient)와 `/skilltools/skill_eval.py`(Sim, run_episode)를 **import만** 해서 재사용 — 새 정책/프레임워크 없음. 서버/이미지/체크포인트/assets는 t_4a072806 skill-eval과 동일.

## 결과 (Xiaomi-Robotics-1-RoboCasa365, seed 9)

`effect` = between / (noise_floor/√N). effect≈1 이면 obs-only(지시무시), ≫1 이면 어텐드. cos = 정답지시 평균청크와의 코사인.

### reset (task 시작 전, lid on counter)
| instruction | cat | d(mean vs 정답) | effect | cos | p |
|---|---|---|---|---|---|
| correct_full | correct | 0.00 | — | 1.000 | — |
| skill_grasp | skill | 7.61 | 10.5 | 0.145 | <1e-4 |
| skill_move | skill | 7.42 | 10.2 | 0.157 | <1e-4 |
| skill_place | skill | 4.91 | 6.8 | 0.567 | <1e-4 |
| opposite_open_drawer | opposite | 8.04 | 11.1 | 0.123 | <1e-4 |
| irrelevant_pick_cup | irrelevant | 5.48 | 7.5 | 0.451 | <1e-4 |
| irrelevant_turn_on_stove | irrelevant | 2.12 | 2.9 | 0.939 | <1e-4 |
| empty | degenerate | 5.48 | 7.5 | 0.456 | <1e-4 |
| nonsense | degenerate | 7.56 | 10.4 | 0.122 | <1e-4 |

### move (mid-task, lid 잡은 상태)
| instruction | cat | d(mean vs 정답) | effect | cos | p |
|---|---|---|---|---|---|
| correct_full | correct | 0.00 | — | 1.000 | — |
| skill_grasp | skill | 0.33 | 1.5 | 0.999 | 0.03 |
| skill_move | skill | 0.39 | 1.8 | 0.998 | 0.003 |
| skill_place | skill | 0.40 | 1.9 | 0.998 | 0.002 |
| opposite_open_drawer | opposite | 1.76 | 8.3 | 0.962 | <1e-4 |
| irrelevant_pick_cup | irrelevant | 1.44 | 6.8 | 0.975 | <1e-4 |
| irrelevant_turn_on_stove | irrelevant | 1.45 | 6.8 | 0.974 | <1e-4 |
| empty | degenerate | 1.36 | 6.4 | 0.978 | <1e-4 |
| nonsense | degenerate | 0.74 | 3.5 | 0.994 | <1e-4 |

### place (mid-task, blender 위 pre-place)
| instruction | cat | d(mean vs 정답) | effect | cos | p |
|---|---|---|---|---|---|
| correct_full | correct | 0.00 | — | 1.000 | — |
| skill_grasp | skill | 0.36 | 0.9 | 0.998 | <1e-4 |
| skill_move | skill | 0.12 | 0.3 | 1.000 | 0.03 |
| skill_place | skill | 0.15 | 0.4 | 1.000 | <1e-4 |
| opposite_open_drawer | opposite | 1.11 | 2.7 | 0.981 | <1e-4 |
| irrelevant_pick_cup | irrelevant | 0.94 | 2.3 | 0.990 | <1e-4 |
| irrelevant_turn_on_stove | irrelevant | 3.68 | 8.9 | 0.761 | <1e-4 |
| empty | degenerate | 1.14 | 2.8 | 0.985 | <1e-4 |
| nonsense | degenerate | 1.03 | 2.5 | 0.986 | <1e-4 |

원자료: `run2/attendance_stats.json`, `run2/probe_{reset,move,place}.json`, `run2/probe_*_raw_full.json`.

## 판정: **어텐드 O — 단, state-의존적** (obs-only 아님)

1. **정책은 instruction에 실제로 어텐드한다.** 모든 상태에서 off-task 지시(open drawer / pick cup / turn on stove / empty)는 정답지시 대비 액션 평균이 노이즈 바닥의 **6~11배**(reset·move) 벗어나고 p<1e-4. obs-only 정책이라면 지시를 바꿔도 effect≈1이어야 하는데 그렇지 않다. reset에서는 무관/반대 지시가 코사인 0.12까지 떨어져 **방향 자체가 반대**로 뒤집힌다.

2. **어텐드 강도는 task 시작에서 가장 크고, task가 진행될수록(obs가 phase를 강하게 규정할수록) 줄어든다.**
   - reset(시작 전): 모든 비정답 지시가 정답과 크게 다름(effect 2.9~11, cos 0.12~0.94). 어떤 행동을 할지 obs로 결정되지 않아 **지시가 지배**.
   - move/place(진행 중): obs가 "잡고 있음/올려둠"을 강하게 규정 → **현재 phase와 일관된 skill 지시**(grasp/move/place)는 정답과 **거의 동일**(effect 0.3~1.9, cos 0.998~1.000). 반면 **명백히 모순되는 off-task 지시**는 여전히 액션을 유의하게 흔든다(effect 6~9, cos 0.96).

3. **이 결과가 hold 실험의 "성공 후 표류"를 설명한다.** mid-trajectory에서 현재 obs와 일관된 skill 부분지시는 full-task 지시와 거의 같은 액션을 낳는다(obs가 지배). 즉 "grasp만 해라"라고 줘도 정책은 obs에서 읽은 **전체 task를 계속 수행**한다 — instruction을 무시해서가 아니라, **부분지시가 학습분포 밖(OOD, 체크포인트는 full-task 지시로만 학습)** 이고 현재 obs와 일관되기 때문에 obs-driven full-task 행동으로 수렴한다. hold 실험의 KEEPS_MOVING은 "instruction 무시"가 아니라 **OOD 부분지시 하에서의 obs-우세 행동**으로 재해석해야 한다.

## 시각 근거 (같은 obs, 다른 지시 → 궤적 분기)

reset 고정 obs에서 120-step open-loop 롤아웃, 지시만 변경 (`div_reset/divergence_reset.json`):

| 지시 | eef 궤적 평균 L2 vs 정답 | eef 종료 위치 요약 |
|---|---|---|
| correct_full | 0.000 | lid로 하강·접근 |
| skill_grasp | 0.108 | lid로 하강·접근 (정답과 유사) |
| opposite_open_drawer | 0.254 | 다른 방향(아래·측면)으로 이동 |
| irrelevant_pick_cup | 0.321 | 위로 크게 상승(lid 반대) |

영상: `videos/CloseBlenderLid/instr_attend/div_reset_{correct_full,skill_grasp,opposite_open_drawer,irrelevant_pick_cup}.mp4`
(원본 `experiments/instr_attend_t_8cd8d8c0/div_reset/`).

## 범위·한계

- **이 리포트는 Xiaomi-Robotics-1-RoboCasa365만 정량 측정 완료.** GR00T N1.5와 π0.5는 미측정 — 이유:
  - GR00T: 추론 하네스(`SimulationInferenceClient.run_simulation`)가 시뮬레이션을 내부에서 통째로 돌려 성공률만 반환, **고정 obs 단발 infer API를 노출하지 않음**. 동일 probe를 하려면 `Gr00tPolicy.get_action(obs)`를 직접 부르는 드라이버가 추가로 필요.
  - π0.5: openpi 추론 하네스가 아직 미구축(별도 task t_ee702b9c에서 구축 예정).
  - 두 모델 모두 **본 probe 패턴(고정 obs·지시만 변경·순열검정)을 그대로 재사용** 가능하므로, 각 모델의 단발 infer 래퍼가 생기면 동일 분석을 붙일 수 있음. 후속 카드로 분리 권장.
- 통계적 유의성은 N=24에서 모든 쌍이 p<0.05이나(샘플 노이즈가 작아 미세차도 유의), **판정은 p가 아니라 effect size/cosine으로** 내렸다(일관 skill 지시 mid-task effect 0.3~1.9·cos≈1.0 = 실질적 동일).
- degenerate(empty/nonsense) 지시도 정답과 유의하게 다름 → 정책이 빈/무의미 입력에 특정 기본 행동을 보임(reset에서 nonsense는 오히려 정답과 크게 다른 방향).

## 재현

```
# v4에서 (xiaomi-server-t_460aea68 기동 상태 필요)
cd /home/v4/experiments/instr_attend_t_8cd8d8c0
./run-probe.sh run2 --states reset,move,place --samples 24 --compare-steps 16
ENTRY=instr_rollout.py ./run-probe.sh div_reset --state reset --horizon 120 \
  --instructions correct_full,skill_grasp,opposite_open_drawer,irrelevant_pick_cup
# 분석(순열검정)은 로컬: experiments/instr_attend_t_8cd8d8c0/ 의 raw_full.json 사용
```
서버 기동: `docker start xiaomi-server-t_460aea68` (image xiaomi-cu121:t_9f03a613, 127.0.0.1:10086).
