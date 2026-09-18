# fixed-20 planning-vs-execution 병목 정밀 분해

판정: `ARCHITECTURE_BOTTLENECK_LOCALIZED`

이 판정은 production-ready 주장이 아니다. 새 GPU rollout 없이 보존된 20개 seed의 `result.json`, `trace.jsonl`, 최종 simulator predicate, paired baseline, 대표 영상을 offline으로 교차검증했다. Runtime planner/verifier가 simulator GT를 사용했다고 간주하지 않는다.

## 결론

Planner는 병목이 아니다. 20/20 plan이 같은 canonical `GRASP_OBJECT -> MOVE_OBJECT -> PLACE_OBJECT` 순서로 유효했고 runtime boundary planner call도 0이다.

가장 좁게 국소화되는 병목은 **skill transition semantics와 fixed per-skill budget 배분의 결합**이다.

1. `run_closed_loop_accuracy.py:169-180`은 각 `manager.execute(call)` 결과가 `SUCCESS`인지 `TIMEOUT`인지와 무관하게 다음 preplanned skill로 진행한다.
2. 반면 `run_closed_loop_accuracy.py:99-110`의 `failure_from`은 첫 non-`SUCCESS` status를 episode failure stage로 기록한다. 따라서 보고된 `GRASP 13 / PLACE 4 / none 3`은 “첫 물리 실패”가 아니라 “첫 verifier timeout” 집계다.
3. `PLACE_OBJECT` verifier success는 0/20인데 official task success는 3/20이다. 성공 seed 5005, 5011, 5017 모두 PLACE `TIMEOUT` 뒤 최종 `lid_on_blender=true`, `lid_upright_7deg=true`, `gripper_lid_far_0.15=true`다. 최소 3건의 PLACE false-negative 또는 terminal-state semantics 불일치가 확정된다.
4. paired baseline-only seed 5004와 5015는 성공 경로에서 PLACE에 각각 172, 112 step을 사용했다. Planned runtime은 PLACE budget을 96으로 고정했다. 5015는 paired 성공 경로보다 정확히 16 step 짧고, 5004는 76 step 짧다.
5. 최종 실패 17건 중 6건은 GRASP policy, 6건은 PLACE policy로 비교적 직접 국소화된다. 나머지는 timeout 뒤 무조건 전이 때문에 첫 물리 실패를 단정할 수 없거나, near/upright/clear와 official support predicate가 불일치한다.

즉 planner conditioning의 aggregate 효과가 0인 핵심 이유는 plan 생성이 아니라, plan을 실행할 때 timeout을 advisory로 취급하면서도 실패 stage는 fatal처럼 기록하고, budget을 다음 stage로 재배분하지 않는 architecture다.

## coarse failure_stage 교차검증

보존 report:

- coarse `GRASP_OBJECT`: 13
- coarse `PLACE_OBJECT`: 4
- `none`: 3

재분류한 primary localization:

| Primary class | Seeds | Count |
|---|---|---:|
| GRASP policy | 5003, 5006, 5010, 5013, 5014, 5018 | 6 |
| MOVE policy | 없음 | 0 |
| PLACE policy | 5002, 5008, 5009, 5012, 5016, 5019 | 6 |
| verifier false-negative / terminal semantics | 5005, 5011, 5017 | 3 |
| transition / budget / timeout semantics | 5000, 5001, 5007, 5015 | 4 |
| official-predicate mismatch | 5004 | 1 |

Primary label은 seed당 하나인 상호배타적 localization이다. Secondary finding으로 seed 5010의 MOVE false-positive, seed 5000/5004/5019의 GRASP sequence-positive/strict-gate-negative가 별도로 있다.

중요한 반례:

- seed 5017은 `GRASP:TIMEOUT -> MOVE:TIMEOUT -> PLACE:TIMEOUT`인데 official success다. coarse `none`이 맞는 이유는 실제 task success 때문이며, 세 timeout 모두 task failure stage가 아니다.
- seed 5005는 GRASP만 `SUCCESS`, MOVE/PLACE는 `TIMEOUT`인데 official success다.
- seed 5010은 `MOVE:SUCCESS`지만 MOVE-end 영상에서 lid가 counter에 남아 있고 최종 `lid_on_counter=true`, XY error 0.444 m다. MOVE verifier false-positive가 확인된다.
- seed 5003과 5006은 GRASP-end 영상에서 lid가 counter에 남아 있어 GRASP policy failure가 확인된다.

## seed별 첫 localization

`S/T`는 obs-only verifier의 `SUCCESS/TIMEOUT`이며 simulator GT가 아니다.

| Seed | G/M/P | Official | Primary | Confidence | 근거 |
|---:|:---:|:---:|---|:---:|---|
| 5000 | T/T/T | F | transition/budget | medium | GRASP sequence p=.999지만 strict gate 0.172<0.453; 이후 전이. 최종 near-target이나 upright/on-blender 실패 |
| 5001 | T/S/T | F | transition/budget | low | GRASP timeout 뒤 MOVE를 실행/성공 처리; boundary GT가 없어 verifier miss와 recovery를 분리 불가 |
| 5002 | S/S/T | F | PLACE policy | high | G/M 통과 후 lid가 near-target이나 upright/on-blender 실패 |
| 5003 | T/T/T | F | GRASP policy | high | GRASP-end 영상에서 lid가 counter에 남음 |
| 5004 | T/T/T | F | official mismatch | medium | XY=.040 m, dz=.008 m, upright/clear=true이나 `lid_on_blender=false`; baseline은 PLACE 172 step으로 성공 |
| 5005 | S/T/T | T | verifier FN | high | MOVE/PLACE timeout인데 모든 terminal placement predicate와 official=true |
| 5006 | T/T/T | F | GRASP policy | high | GRASP-end 영상 및 최종 predicate에서 lid가 counter에 남음 |
| 5007 | T/T/T | F | transition/budget | low | 모두 timeout이나 이후 policy가 lid를 near-target까지 이동; boundary GT 부재 |
| 5008 | S/S/T | F | PLACE policy | high | G/M 통과 후 upright/on-blender/clear 실패 |
| 5009 | T/T/T | F | PLACE policy | medium | 최종 near-target이나 upright/on-blender/clear 실패; 앞선 timeout은 물리 실패로 확정 불가 |
| 5010 | T/S/T | F | GRASP policy | high | MOVE-end 영상에서 lid가 counter에 남는데 MOVE `SUCCESS`; secondary MOVE false-positive |
| 5011 | S/S/T | T | verifier FN | high | PLACE timeout 뒤 official 및 terminal placement predicate=true |
| 5012 | S/S/T | F | PLACE policy | high | 최종 lid_on_counter=true, XY=.228 m |
| 5013 | T/T/T | F | GRASP policy | high | 최종 lid_on_counter=true, XY=.471 m |
| 5014 | T/T/T | F | GRASP policy | medium | boundary success 없음; 최종 dz=-1.346 m로 object drop 상태 |
| 5015 | T/S/T | F | transition/budget | high | baseline PLACE 112 step 성공, planned PLACE 96 step timeout; 최종 near-target이나 upright/on-blender 실패 |
| 5016 | S/S/T | F | PLACE policy | high | G/M 통과 후 upright/on-blender/clear 실패 |
| 5017 | T/T/T | T | verifier FN | high | 세 boundary timeout 모두 official success와 불일치 |
| 5018 | T/T/T | F | GRASP policy | high | 최종 lid_on_counter=true, XY=.632 m |
| 5019 | T/T/T | F | PLACE policy | medium | 최종 near-target이나 upright/on-blender/clear 실패; GRASP sequence signal은 strict gate에서 거부 |

경계별 simulator GT가 저장되지 않았으므로 confidence=low인 5001/5007은 category를 더 세밀하게 확정할 수 없다. 이를 숨기지 않고 `summary.json`에 명시했다.

## `PLACE 0/20` 대 official `3/20`

이 불일치는 단순 집계 오류가 아니다.

- PLACE verifier는 20개 seed에서 모두 `TIMEOUT`이다.
- official success seed 5005, 5011, 5017의 최종 simulator predicate는 모두 task 완료 상태다.
- PLACE는 마지막 skill이라 verifier miss가 다음 skill 선택을 망치지는 않지만, runtime status와 failure accounting을 잘못 만든다.
- 더 중요한 문제는 같은 semantics가 GRASP/MOVE에서도 timeout 뒤 unconditional advance와 결합된다는 점이다. 5017처럼 세 단계가 모두 timeout이어도 task가 성공할 수 있고, 5010처럼 MOVE가 success여도 물리 상태는 불충족일 수 있다.

따라서 `PLACE 0/20`은 **확정된 verifier false-negative/terminal semantics 문제**다. 하지만 official 17개 실패까지 verifier만으로 설명할 수는 없다. G/M을 통과한 4개 실패(5002, 5008, 5012, 5016)는 PLACE execution failure이고, final-state evidence도 이를 지지한다.

## paired discordance 직접 비교

### Baseline-only: planner-conditioned runtime이 손해

| Seed | Baseline path | Planned path | 손해 경로 |
|---:|---|---|---|
| 5004 | G128 S -> M82 S -> P172 S | G208 T -> M288 T -> P96 T | verifier/transition overrun 뒤 PLACE 96 cap. 최종은 near/upright/clear이나 support predicate 불충족 |
| 5015 | G159 S -> M82 S -> P112 S | G208 T -> M55 S -> P96 T | GRASP miss 뒤 전이, PLACE가 paired 성공 경로보다 16 step 짧음 |

두 seed 모두 planner가 잘못된 skill sequence를 생성한 것이 아니다. 같은 3-skill 구조에서 verifier timing과 budget schedule이 성공 경로를 잃게 했다.

### Planned-only: planner-conditioned runtime이 이득

| Seed | Baseline path | Planned path | 이득 경로 |
|---:|---|---|---|
| 5005 | G110 S -> M67 S -> P200 T | G200 S -> M288 T -> P96 T -> official S | 더 긴 GRASP/MOVE conditioning이 terminal state를 만들었으나 MOVE/PLACE verifier는 놓침 |
| 5011 | G126 S -> M89 S -> P200 T | G178 S -> M106 S -> P96 T -> official S | instruction/timing 변화가 성공 trajectory를 만들었으나 PLACE verifier는 여전히 timeout |

planned-only 이득도 planner의 symbolic plan 차이 때문이 아니다. canonical sequence는 동일하며, rendered instruction과 boundary duration이 policy trajectory를 바꾼 효과다.

## 대표 영상 / trace index

모든 영상은 20 fps다. `video_s = frame / 20`; `trace_elapsed_s`는 planner 및 RPC 시간을 포함하므로 영상 시각과 다르다.

| Case | Video | GRASP boundary | MOVE boundary | PLACE boundary |
|---|---|---|---|---|
| success | `results/videos/success_seed5005.mp4` | frame 100 / 5.00s / trace 29.217s | 244 / 12.20s / 46.355s | 292 / 14.60s / 52.041s |
| coarse-GRASP, late-state failure | `results/videos/execution_failure_grasp_seed5000.mp4` | 104 / 5.20s / 34.723s | 248 / 12.40s / 54.040s | 296 / 14.80s / 61.003s |
| late PLACE failure | `results/videos/execution_failure_late_seed5002.mp4` | 104 / 5.20s / 27.487s | 143 / 7.15s / 32.596s | 191 / 9.55s / 39.423s |

각 boundary는 해당 `trace.jsonl`의 `skill_result` 직전 마지막 `frame` event와 연결했다. 원본 seed 영상도 동일 frame index로 재검토할 수 있다.

## 가장 작은 다음 개선 한 가지

**Fixed per-skill unconditional advance를 shared episode-budget handoff rule로 교체한다.**

- `SUCCESS`면 다음 skill로 전이한다.
- `TIMEOUT`이면 obs-only next-skill handoff gate가 통과할 때만 전이한다.
- handoff gate가 실패하면 현재 skill을 유지/재시도한다.
- 사용하지 않은 episode budget은 다음 skill, 특히 PLACE로 이월한다.
- planner, policy weight, runtime simulator-GT 입력은 바꾸지 않는다.

이 한 가지가 가장 작은 이유는 planner나 policy를 재학습하지 않고도 (a) timeout 뒤 무조건 전이, (b) 첫-timeout failure accounting, (c) PLACE 96-step cap으로 확인된 paired regression을 동시에 겨냥하기 때문이다.

### 검증 가능한 acceptance gate

동일 seed 5000..5019, 동일 model/checkpoint, planner 1회, repair/fallback 없음, obs-only runtime으로 paired rerun한다.

필수 통과 조건:

1. planner-valid 20/20 유지, mechanics violation 0 유지.
2. baseline-only regression seed 5004와 5015가 둘 다 official success.
3. 현재 성공 seed 5005, 5011, 5017을 잃지 않으면서 official success >=5/20.
4. `TIMEOUT` 뒤 unconditional advance 0건.
5. 보고된 failure stage가 실제로 전이하지 못한 최초 handoff와 일치.
6. Offline final official-positive에 대한 PLACE false-negative 0건. 이 평가는 offline labeling이며 simulator predicate를 runtime 입력으로 사용하지 않는다.

## Evidence 및 한계

- Evidence commit: `cf07ec11db1f03b02e972cd4300e0ed254babead`
- CSV 정규화 후속 commit: `4c4558f09caa07a1c65e0034393f1c0726562618`
- Execution source commit: `45280448bd9f650c3cfd94f26939a8830374a05b`
- Paired baseline: `rl-train-t_3ed65912/results/t_5ede00c3/eval_final30_t5ede00c3/data/trained_paired.json`, SHA-256 `80887a458d1db6ada52e0cb188660405d96216f74d95f4cbfcc1d6d4d1ff68f6`
- 새 GPU rollout: 실행하지 않음.
- simulator predicate는 episode final에만 보존되어 boundary-level GT timeline은 없다. 그래서 일부 seed는 physical first-failure를 high confidence로 단정하지 않았다.
- 영상 판독은 offline labeling에만 사용했다.
