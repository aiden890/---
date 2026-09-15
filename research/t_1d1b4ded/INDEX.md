# RoboCasa planning-failure 탐색용 task 선정 / t_1d1b4ded

## 범위와 운영 주의

이 산출물은 공식 환경 소스의 정적 조사와 실행 계획이다. 정책 실행, 원격 상태 실측, 신규 영상 판정은 이 카드에서 수행하지 않았다. 기존 파일은 수정하지 않았으며 새 연구 폴더 안에만 소스 사본과 보고서를 작성했다.

운영자 전달 사항: PID55518은 ps에 없지만 xiaomi-client-t_460aea68-lettuce-01, -steam-01, -lunches-01 및 xiaomi-server-t_460aea68가 docker ps에 실행 중으로 보였다. 이는 운영자의 시점별 관측이며 현재도 실행 중이라고 단정하지 않는다. 후속 담당자는 새 실행 전에 SSH agent, docker ps/inspect, nvidia-smi, 기존 stats/log/MP4를 다시 확인해야 한다. PID 소멸만으로 종료나 GPU 유휴를 판단하지 않는다. 동시에 여러 클라이언트가 같은 stateful inference server를 사용했는지도 점검하고, episode 상태 격리/요청 순서가 불명확하면 interface-confounded로 보존한다. 임의 중지나 중복 실행 금지. 승인 게이트는 우회하지 않는다.

사용자 지시로 원격 파일의 로컬 전송은 연기한다. 기존 원격 결과 검증/보존을 우선하며 이번 로컬 연구문서는 이미 존재하는 로컬 checkout만 사용했다. 후속 실행/환경 카드 t_f22ea8b3 및 t_700146a2에 위 경고를 전달했다.

## 기존 결과 및 검증된 조건

직접 읽은 자료: /home/aiden/Desktop/lab/robot/robocasa-docker/rollouts/xiaomi-robotics-1/INDEX.md, summary.json, run-t_5af7225b.sh. 읽은 자료의 사본과 해시는 sources/prior/ 및 source_manifest.json에 있다.

기존 고유 task는 TurnOnSinkFaucet, PrepareCoffee, KettleBoiling, HeatKebabSandwich, WaffleReheat이다. 이 5개와 CloseBlenderLid를 전부 제외했다. 이전 summary는 6개 완료 episode(성공 2, horizon 실패 4), 별도 reset 오류를 기록한다. HeatKebabSandwich seed14는 baguette 누락 후보지만 seed15의 실패 양상은 달라 재현된 planning failure는 아니다. 본 조사에서 이전 영상 자체를 재분석하지 않았다.

이전 검증 조건(이전 기록의 사실이지 현재 원격 상태 보장이 아님):

- RoboCasa commit: 4f8a2980def75a55dff96b990745b83540425f09. 로컬 vendor checkout HEAD와 clean working tree를 실제 git으로 확인했다.
- Xiaomi code: 0dd7aef8dc87296246aae812a1f59ccb708e5546.
- Model: XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365; checkpoint revision 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4.
- CUDA 12.1.1, torch 2.5.1+cu121, RTX3090 24GB, host driver 535.183.01.
- Images: xiaomi-cu121:t_9f03a613, xiaomi-client:t_9f03a613. 정확한 image ID는 selection.json의 versions에 보존.
- split=target; replan_steps=16, obs_history=4, obs_interval=2, crop_ratio=0.95, video_stride=2, video_fps=20; task default horizon 유지.
- writable asset clone robocasa-assets-t_5af7225b 사용. read-only asset으로 upstream 임시 XML 작성이 실패했던 기록이 있으므로 원본 자산을 수정하거나 RO로 단순 회귀하지 않는다.
- 이전 서버 RNG는 명시적으로 seed되지 않았다. simulator seed 고정은 bitwise policy replay 보장이 아니다.

경로:

- 기존 로컬 결과: /home/aiden/Desktop/lab/robot/robocasa-docker/rollouts/xiaomi-robotics-1
- 기존 원격 결과: /home/v4/rollouts-xiaomi-t_5af7225b
- 모델/실행 코드 원격: /home/v4/robocasa-docker-t_9f03a613
- 이미 시작된 추가 batch 결과(후속 담당자 확인 필요): /home/v4/rollouts-xiaomi-t_460aea68
- 본 연구: /home/aiden/Desktop/lab/robot/robocasa-docker/research/t_1d1b4ded

## 선정표

분류의 기준은 추정 난이도나 경로명이 아니라 pinned dataset_registry.py TARGET_TASKS이다. composite_seen은 2823-2840, composite_unseen은 2841-2858이다. 모든 후보는 공식 target task이다. horizon은 같은 파일 COMPOSITE_TASK_DATASETS의 항목이며 get_task_horizon(240-253)이 그대로 반환한다.[1][2]

| 우선순위 | 정확한 task ID | 공식 분류 | 원래 horizon | registry 항목 줄 | base7의 예상 actual seed |
|---|---|---|---:|---|---:|
| primary | BreadSelection | Composite-Unseen | 1950 | 715-720 | 9 |
| primary | CategorizeCondiments | Composite-Unseen | 1650 | 745-750 | 10 |
| primary | PackIdenticalLunches | Composite-Seen | 3900 | 1342-1350 | 11 |
| primary | SteamInMicrowave | Composite-Seen | 2100 | 1960-1968 | 19 |
| primary | WashLettuce | Composite-Seen | 1650 | 2107-2115 | 22 |
| reserve | PreSoakPan | Composite-Seen | 2400 | 1477-1485 | 12 |
| reserve | MakeIceLemonade | Composite-Unseen | 3000 | 1174-1179 | 15 |
| reserve | WashFruitColander | Composite-Unseen | 3150 | 2101-2106 | 21 |

primary는 이전 t_460aea68 시도에서 이미 시작한 5개와 일치하도록 선택했다. 따라서 신규란 t_5af7225b 및 CloseBlenderLid 대비 신규이며, 이미 시작된 추가 batch를 무시하고 다시 실행하라는 뜻이 아니다. 연구된 8개 중 primary만으로 5개/Unseen 2개 요건을 충족한다. reserve는 필요할 때만 사용하고 task를 교체하면 Unseen 최소 수를 다시 검증한다. WashLettuce는 다물체 분배 과제보다 짧은 누적 상태 조건을 가진 비교군이다.

## 공통 판정 helper의 의미

아래 R(a,b)는 OU.check_obj_in_receptacle이다. 실제로는 물체-용기 접촉 AND xy 중심거리 < threshold이며 기본 threshold=용기 horizontal_radius*0.7이다. 엄밀한 3D 내부 포함 판정과 다르다(587-601). F(a)는 gripper_obj_far: 오른쪽 EEF와 물체 중심의 3D 거리 > 0.25(645-652), release/grasp 해제 그 자체를 판정하는 함수가 아니다. I(a,fixture)는 obj_inside_of: 기본 partial_check=False로 물체 bbox 8점을 fixture interior region의 투영 경계와 비교한다(14-64). 보이는 위치만으로 이 Boolean들을 확정하지 않는다.[3]

U(a)는 sink.check_obj_under_water: 물체 중심 xy가 water site에 물체 horizontal_radius보다 가까우며, z가 water site z + size[1] 미만이고 water_on인 조건이다(316-327). 유체 시뮬레이션, 실제 젖음, basin 내부 포함 자체를 검사하는 것은 아니다.[4]

다음 instruction은 get_ep_meta의 기본 문자열/템플릿이다. novel instruction 옵션 및 실제 object naming에 따라 달라질 수 있으므로 실행 시 observation의 annotation.human.task_description과 ep_meta를 반드시 보존한다.

## Primary task별 근거 및 수집안

### BreadSelection

소스: robocasa/environments/kitchen/composite/making_toast/bread_selection.py. get_ep_meta 31-40, scene 42-47, success 120-126.[6]

Instruction: "From the different types of pastries on the counter, select a croissant and place it on the cutting board. Then retrieve a jar of jam from the cabinet and place it alongside the croissant on the cutting board."[6]

필수 목표는 croissant와 jam을 cutting_board에 놓기다. 문장은 croissant 다음 jam을 요구하지만 성공 코드는 R(croissant,cutting_board) AND F(croissant) AND R(jam,cutting_board)만 요구한다. 역사적 순서, jam의 F, cabinet 닫기, distractor 제거는 조건이 아니다. cabinet은 초기 open이다. 역순 성공은 instruction-order deviation으로 별도 기록하고 환경 실패라고 부르지 않는다.[6]

후보 관측 기준(제안): croissant를 무시한 채 jam으로 전환하고 croissant가 끝까지 원위치, 혹은 distractor pastry를 선택하여 필수 croissant 누락. evidence: 각 pastry의 실제 ID/asset/화면 위치, 두 R, croissant EEF 거리, 대상별 이동/접촉 시점, cabinet 접근 구간. croissant를 옮기려다 흘렸으면 control/execution; 다른 pastry 선택은 시각적 식별 실패와 목표선택 실패를 분리할 수 없으면 ambiguous. jam 누락과 jam 운반 실패를 구분한다.

### CategorizeCondiments

소스: robocasa/environments/kitchen/composite/arranging_condiments/categorize_condiments.py. instruction 24-32, scene 34-36, 거리 112-119, success 121-138.[7]

Instruction: "Put the shaker and condiment bottle from the counter next to their counterparts in the cabinet."[7]

obj1=bottle, obj2=shaker; cab_obj1=bottle counterpart, cab_obj2=shaker counterpart. 성공은 I(obj1,cab) AND I(obj2,cab) AND xy_distance(obj1,cab_obj1)<=0.15 AND xy_distance(obj2,cab_obj2)<=0.15 AND F(obj1) AND F(obj2). 픽업 순서 없음. cabinet은 초기 open. 상대 물체를 제자리에 유지하거나 cabinet을 닫는 조건은 별도로 없다. xy 근접은 같은 높이/선반 조건 그 자체가 아니다.[7]

후보 관측 기준(제안): 한 종류만 수납 후 완성된 쪽을 반복, 두 종류를 서로 반대 counterpart 옆에 배치, 둘 다 임의 cabinet 위치에 둔 후 matching 목표를 포기. evidence: 네 물체 ID와 pose, 두 I, 대응/비대응 xy 거리 전부, F, placement 접촉, 두 번째 물체 방치 구간. 올바른 상대에게 접근했으나 최종 거리만 0.15를 벗어나면 정밀 조작 실패 가능성이 크다. counterpart가 가려져 있으면 grounding/ambiguous로 남긴다.

### PackIdenticalLunches

소스: robocasa/environments/kitchen/composite/packing_lunches/pack_identical_lunches.py. scene 29-31, instruction 33-47, success 178-204.[8]

Instruction template: "Place one {veg_lang} and one {meat_lang} in each tupperware on the nearby counter, to pack two identical lunches."[8]

필수 목표는 vegetable0/1, meat0/1의 네 물체를 두 tupperware에 분배하여 각각 채소 정확히 1개와 고기 정확히 1개를 담기다. 코드가 8개 R을 수집하여 각 container별 두 종류의 count==1을 요구하고, 전체 membership에 중복 물체가 없으며 membership 모든 물체가 F여야 한다. fridge는 초기 open. 픽업/분배 순서, fridge 닫기, tupperware 위치 유지/뚜껑은 별도 검사하지 않는다.[8]

후보 관측 기준(제안): 한 lunch만 완성 후 종료성 반복 행동; 같은 종류 두 개를 한 용기에 계속 모으기; 다른 용기의 미충족 종류를 보완하지 않기. evidence: 전체 4x2 R membership matrix, 각 count, no_duplicates, 네 F, object identity/pose 및 이동 이벤트. 올바른 용기에 운반했으나 튀어나옴은 manipulation; 식재료 분류가 화면상 불명확하면 grounding. 단순 horizon 종료로 누락을 단정하지 않는다.

### SteamInMicrowave

소스: robocasa/environments/kitchen/composite/steaming_food/steam_in_microwave.py. instruction 31-46, scene 48-51, success 118-127.[9]

Instruction template: "Pick the {vegetable_name} from the sink and place it in the bowl. Then pick the bowl and place it in the microwave. Then close the microwave door and press the start button."[9]

필수 목표: vegetable in bowl, bowl in microwave, door closed, turned_on. 성공은 R(vegetable,bowl) AND I(bowl,microwave) AND microwave.is_closed AND get_state()['turned_on']; F나 누적 steaming 시간 없음. microwave는 초기 open, sink는 off. task 코드 자체는 과거에 bowl을 어디서 언제 채웠는지 기록하지 않는다.[9]

문장의 순서는 채소 담기 -> bowl 운반 -> door 닫기 -> start. fixture update_state(66-90)는 is_open(th=0.90)이면 turned_on=False로 하고 그렇지 않으면 start/stop 접촉을 반영한다. 성공의 is_closed 기본 th=0.005와 start 가능 조건의 is_open 임계는 다르므로 '완전히 닫힌 뒤에만 start가 가능'이라는 강한 순서 제약을 만들어내면 안 된다.[5]

후보 관측 기준(제안): 채소가 sink에 남았는데 빈 bowl을 microwave에 넣고 닫기/start로 전환; bowl이 밖인데 door/start를 반복. evidence: 두 R/I, 채소/그릇 pose, door joint/is_open/is_closed, start/stop contact, turned_on, 전환 시점. bowl 운반 중 채소 탈락은 control; 정상 적재 후 button 반복은 button/door 정밀 조작 또는 grounding으로 분리. 사전조건 미충족 상태의 일시적 start 후 스스로 복구한 경우 최종 누락과 구분한다.

### WashLettuce

소스: robocasa/environments/kitchen/composite/making_salads/wash_lettuce.py. instruction 23-29, scene 31-33, update_state 57-60, success 62-63.[10]

Instruction: "Wash the lettuce in the sink by running water over it." washed_time은 reset에 0이고 U(lettuce)인 update마다 1 증가한다. 성공은 washed_time>=25뿐이다. 연속 25회일 필요 없고 물 밖으로 나왔다고 카운터가 0이 되지 않는다. 최종 faucet on, lettuce가 sink 안, gripper far, colander 운반 자체는 추가 조건이 아니다.[10][4]

필수 subgoal은 lettuce와 running water를 겹치게 유지하여 누적 카운터 달성이다. 물 켜기와 이동의 엄격한 역사적 순서는 없다. 후보(제안): lettuce를 물 아닌 basin 한쪽에 두고 faucet만 반복하거나 colander만 옮겨 lettuce를 방치. evidence: lettuce/colander pose, U의 xy/z/water_on 분해, horizontal_radius, water_site pose/size, washed_time의 매 update 변화. water 아래 정렬 실패는 control/grounding일 수 있고, timer 미충족만으로 planning이라 하지 않는다. 누적 시간은 source update 횟수이며 실제 호출 주기를 로깅하기 전 초 단위로 환산하지 않는다.

## Reserve task별 근거 및 수집안

### PreSoakPan

소스: robocasa/environments/kitchen/composite/washing_dishes/pre_soak_pan.py. instruction 25-33, scene 35-37, success 85-96.[11]

Instruction: "Pick the pan and sponge and place them into the sink. Then turn on the water." 성공은 water_on AND I(obj1=pan,sink,partial_check=False) AND I(obj2=sponge,sink,partial_check=False) AND F(pan) AND F(sponge). pan/sponge 픽업 순서는 없고, 문장의 물 켜기 후행 조건은 이력으로 강제하지 않는다. soaking 시간이나 water 아래 정렬도 없다.[11]

후보(제안): pan만 수납하고 sponge를 남긴 채 faucet/완료 목표 반복. evidence: 두 I/F, water_on, pan bbox와 basin 경계, 두 물체 이동 이력. pan을 넣으려다 bbox가 걸리는 상황은 control; water를 먼저 켰어도 두 물체를 결국 넣으면 환경 성공 가능하며 순서 이탈만 별도 기록.

### MakeIceLemonade

소스: robocasa/environments/kitchen/composite/adding_ice_to_beverages/make_ice_lemonade.py. instruction 26-35, success 131-147.[12]

Instruction: "Grab a lemon wedge from the fridge and one ice cube from the ice bowl, and put them in the glass of lemonade." 성공은 R(lemon_wedge,glass_cup,th=0.5) AND (R(ice_cube1,glass_cup,th=0.5) OR R(ice_cube2,glass_cup,th=0.5)) AND F(lemon_wedge,th=0.15) AND F(ice_cube1,th=0.15) AND F(ice_cube2,th=0.15). 어느 얼음이나 가능하며 문장의 one과 달리 두 얼음 모두 들어가도 배제하지 않는다. lemon/ice 순서, fridge 닫기, 사용하지 않은 얼음의 bowl 잔류는 강제하지 않는다.[12]

후보(제안): 얼음만 반복 넣고 lemon 누락, lemon만 넣고 얼음 누락, glass 대신 ice_bowl로 목표 전환. evidence: 세 R과 거리/contact, 세 F, glass/ice_bowl/lemon 및 얼음 ID/pose. 작은 얼음 grasp 실패는 control, 투명 얼음 식별 문제는 perception 가능. 얼음 두 개를 넣었다는 사실만으로 환경 실패나 planning-failure로 집계하지 않는다.

### WashFruitColander

소스: robocasa/environments/kitchen/composite/washing_fruits_and_vegetables/wash_fruit_colander.py. count 15-18, instruction 20-37, success 86-93.[13]

Instruction template: "Put the colander in the sink, put the {fruit_lang} in the colander, and turn on the sink faucet and pour water over the colander." num_fruit는 ep_meta refs에서 복원하거나 1/2/3 중 sampling. 성공은 모든 fruit_i의 R(fruit_i,colander) AND U(colander). I(colander,sink), 개별 fruit가 물 아래인지, 연속 세척 시간, F, 과거 순서는 별도 조건이 아니다.[13][4]

후보(제안): num_fruit가 2/3인데 일부만 넣고 water로 전환 후 나머지 방치, 완성된 과일을 계속 꺼냈다 넣기. evidence: 실제 num_fruit 및 fruit별 이름, 모든 R, colander U의 xy/z/water_on 분해, grasp/drop 이벤트. 물로 이동 중 과일이 떨어짐은 control, 과일이 가려졌다면 ambiguous. 임의로 num_fruit를 바꾸거나 원하는 다물체 seed만 성공률 집계에 남기지 않는다.

## 후속 실행 및 증거 수집 계획

1. t_700146a2의 원격 점검 handoff를 읽고 위 in-progress root의 시도부터 inventory한다. primary의 bread-01, condiments 관련 시도(정확한 디렉터리는 현장 확인), lunches-01, steam-01, lettuce-01을 찾아 task/config/stats/log를 대조한다. 완성된 유효 episode를 재사용하고 손상/중단 시도도 별도 보존한다. 완료되지 않은 시도를 완료로 세지 않는다.
2. baseline 설정과 모델은 유지한다. --horizon override를 주지 않는다. 추가 실행은 GPU가 점검되고 승인이 확보된 뒤 단일 client 순차 실행으로 수행한다. 기존 /home/v4/rollouts-xiaomi-t_460aea68/run-t_460aea68.sh는 읽어 parent 대비 실제 차이를 확인한 뒤 사용한다. 아직 생성/실행된 것으로 취급하면 안 되는 아래 명령은 예시이며 unique attempt directory가 없어야 한다.

    cd /home/v4/rollouts-xiaomi-t_460aea68
    bash run-t_460aea68.sh episode composite-unseen BreadSelection 7 bread-recovery-02
    bash run-t_460aea68.sh episode composite-unseen CategorizeCondiments 7 condiments-recovery-02
    bash run-t_460aea68.sh episode composite-seen PackIdenticalLunches 7 lunches-recovery-02
    bash run-t_460aea68.sh episode composite-seen SteamInMicrowave 7 steam-recovery-02
    bash run-t_460aea68.sh episode composite-seen WashLettuce 7 lettuce-recovery-02

3. 명령의 base seed와 실제 episode seed를 모두 기록한다. 기존 rollout.py 216-225, select_tasks 360-378은 category 전체 index를 유지하므로 num_trials=1이면 actual=base+category index다. 이 보고서의 expected seed는 그 계산이지 이미 수행된 원격 episode 관측값이 아니다. 실제 stats로 다시 확인한다.
4. 각 task의 observation instruction, ep_meta(물체 이름/자산, layout/style, fixture refs), reset seed, horizon, 코드/image/checkpoint 해시, action mapping 및 shape/finite 검사 결과를 보존한다. 성공/실패, 종료 이유, exception은 분리한다. reset/interface error는 policy failure가 아니다.
5. 각 step의 RGB/MP4와 step-to-frame mapping, action, gripper/물체 pose 및 접촉, 위 predicate vector를 상태 로그로 수집할 것을 권장한다. env 성공 Boolean과 분석용 Boolean을 분리한다. 읽기 전용 계측만 허용하며 update_state를 추가 호출하거나 stateful 판정을 다시 계산해 washed_time을 늘리는 계측을 금지한다. env/fixture/model을 변경하지 않고 기존 callback 후 값을 읽는다. 계측이 승인 불가이면 원래 결과를 보존하고 상태 증거 없음/ambiguous로 남긴다.
6. 로그 schema 제안: episode record에 task_id, split, category, base_seed, actual_seed, instruction, horizon, steps, termination_reason, success, versions, command, artifact_paths; step record에 step, sim_time, frame_index, video_seconds, predicates, object_poses, fixture_state, action; failure record에 start/end_step, timestamp, observed_action, missing_goal, enforced_or_instruction_only, alternative_explanations, taxonomy, confidence, evidence_paths. state unavailable은 null로 남긴다.
7. MP4 전체 decode/size/SHA256은 원격에서 확인하고 검증 stdout/log를 원격 결과와 함께 보존한다. 영상 timestamp는 playback time이며 stride2/fps20의 실제 캡처 mapping을 이용한다. 월시계 시간이나 simulator time과 혼동하지 않는다. 이후 사용자 승인 전 로컬 다운로드하지 않는다.
8. planning-failure candidate는 관측 가능한 잘못된 목표/누락/반복/순서 이탈의 외부 행동 가설이다. 모델 내부 reasoning을 관측했다는 주장은 금지한다. 올바른 대상 접근 후 grasp/drop/collision은 control/execution, 시각 식별 혼동은 perception/grounding, 원래 horizon에 걸린 것은 time-limit(인과 원인과 별도), reset/action/서버 오류는 environment/interface, 분리 안 되면 ambiguous로 기록한다. 환경 성공과 instruction deviation도 별도 축으로 둔다.
9. 유력 후보는 추가 base seed 8과 9에서 같은 task/horizon으로 재검증한다(각각 actual seed를 기록). 후보 동작이 반복되는지와 반례/성공도 함께 기록하며 실패 두 번이라는 이유만으로 같은 planning 오류 재현이라 하지 않는다. 재현된 행동 패턴만 reproduced planning failure candidate로 표시하고 인과적 내부 planning 결함 확정과 구분한다. 후보가 없으면 없다고 보고한다.

## 파일 및 검증

selection.json은 정확한 task ID, category, horizon, 예상 seed, method line range와 기존 versions를 담는다. source_manifest.json은 pinned GitHub URL과 로컬 원본/사본/SHA256을 담는다. sources/는 원본 줄 번호를 유지하는 전체 소스 사본이며 공개 웹 재요청 대신 clean pinned git checkout에서 읽었다. validation.json은 정적 검증 결과다. 본 결과는 원격 새 task의 성공이나 정책 호환성 실행 검증을 주장하지 않는다.

## Sources

[1] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/utils/dataset_registry.py
[2] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/utils/dataset_registry_utils.py
[3] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/utils/object_utils.py
[4] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/models/fixtures/sink.py
[5] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/models/fixtures/microwave.py
[6] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/making_toast/bread_selection.py
[7] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/arranging_condiments/categorize_condiments.py
[8] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/packing_lunches/pack_identical_lunches.py
[9] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/steaming_food/steam_in_microwave.py
[10] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/making_salads/wash_lettuce.py
[11] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/washing_dishes/pre_soak_pan.py
[12] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/adding_ice_to_beverages/make_ice_lemonade.py
[13] https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/washing_fruits_and_vegetables/wash_fruit_colander.py
