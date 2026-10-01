# π₀.₅ · PrepareCoffee 컵 놓기 Online BC

## 폴더와 코딩 규칙

Python의 일반적인 src layout을 사용한다. 코드 읽기를 시작할 곳은 `src/online_bc/`다.

```
online-bc/
  pyproject.toml            # 패키지 정의, 공통 서식 규칙
  src/online_bc/
    data/                  # 성공 구간 추출, 검증, replay 샘플링
    learning/              # gradient 업데이트와 learner 서비스
    models/                # π, Xiaomi, GR00T 모델 연결
    rollout/               # 시뮬레이터 실행과 워커
    orchestration/         # 수집 → 학습 → 가중치 교체 조정
    transport/             # HF 데이터 및 관리 설정 동기화
    review/                # 영상 목록과 제외/재포함 API
    validation/            # 모델 추론·HTTP 연결 검증
    _vendor/               # 재사용한 원본 RLinf 코드와 라이선스
  configs/                 # 서버별 설정; 알고리즘 코드와 분리
  scripts/                 # Lab / SKKU 실행 진입점
  web/                     # 관리 화면
  tests/                   # 자동 검증
  reports/verification/    # 실제 검증 기록
  requirements/            # 모델별 학습 환경 의존성
  vendor/openpi/           # 고정한 OpenPI 원본 소스 압축 파일
```

- 파일 하나는 한 역할을 맡는다. 모델마다 다른 라이브러리는 `models/`에 두고, UI에서 모델을 직접 호출하지 않는다.
- 의존성은 절대 import(`from online_bc.data.replay import Replay`)로 명시한다. 같은 이름의 외부 파일을 잘못 불러오는 일을 방지한다.
- 일반 환경에서는 `python -m pip install -e . --no-deps`로 개발용 설치 후 `python -m online_bc.learning.train_online_bc ...`처럼 실행한다. 모델 라이브러리는 해당 환경에 별도로 설치한다.
- GPU 이미지와 호스트 도구는 소스 마운트와 `PYTHONPATH`를 실행 스크립트에서 지정한다. 개별 파일을 직접 실행하는 방식은 쓰지 않는다.
- 들여쓰기는 4칸, 함수·파일 이름은 `snake_case`, 클래스는 `PascalCase`, 한 줄에 한 문장으로 작성한다. `ruff format src/online_bc tests`로 서식을 통일한다. `_vendor` 원본은 자동 수정하지 않는다.
- 데이터·체크포인트·로그·인증 토큰은 코드 폴더 밖에 저장한다.
- 변경한 기능의 동작을 `tests/`에서 검증한다. 폴더를 더 나누는 것은 역할이 실제로 늘어날 때만 한다.

Lab: `bash scripts/manage.sh status` / `review` / `collect 5` / `test-review`.
SKKU: `bash scripts/start-pi-cup-learner.sh setup|check|download|validate|start`.

처음에는 `data/build_cup_dataset.py` → `data/replay.py` → `learning/train_online_bc.py` → `models/pi05_backend.py` 순서로 읽으면 된다. 관리 기능은 `review/review_server.py` → `transport/control_sync.py` → `data/data_control.py`에서 이어진다.

현재 본 학습은 시작하지 않았다. 사용자가 지정할 SKKU GPU 세션 또는 AMP learner에서 아래 검증을 마친 뒤 시작한다. 다른 두 모델의 backend는 보존하며 이번 학습은 π₀.₅의 컵 놓기만 수행한다.

## 지시문과 데이터

`Place the mug you are holding upright on the coffee machine tray directly under the dispenser, then release the mug.`

- 시작: native `mug_grasped`이고 컵이 초기 위치에서 10cm 이상 이동한, 실제로 저장된 16스텝 청크 경계.
- 끝: RoboCasa의 디스펜서 아래 위치 판정과 잡기·접촉 해제가 5스텝 연속 성립한 시점. 앞서 버튼을 누른 궤적은 제외.
- 초기 데이터: PrepareCoffee pretrain split의 실제 성공 실행 11개, 235개 청크. π 5개/126청크, Xiaomi 6개/109청크. 사람 전문가 시연으로 표기하지 않는다.
- 각 예제는 행동 실행 직전의 3개 카메라와 원래 로봇 상태, 실제 실행한 12D 액션을 가진다. π에는 native quaternion 16D state를 넣는다. 별도로 저장한 Xiaomi의 axis-angle 14D history와 혼동하지 않는다.
- 50스텝 action target과 validity mask를 저장한다. 구간 종료 이후 padding은 loss에서 제외한다. 버튼 스킬 라벨은 이번 실험에 넣지 않는다.
- 초기 평가 981001~981030. 새 온라인 수집은 라운드 r의 993000+100*r+1~8. 별도 평가 시드는 992001부터이며 학습 replay에 넣지 않는다.

[데이터 영상 예시 10개](http://100.86.183.64:8899/media/pi05-cup-bc-20261001/index.html)

## 구현 출처

- 모델, native CFM loss, JAX/Flax 모델 구조, LoRA, 토크나이저: [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi).
- 현재 RoboCasa 어댑터와 모델 구성: [robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi/tree/5a6beda9ff99da30b4e1b59320f6a32971d7c397), commit `5a6beda9ff99da30b4e1b59320f6a32971d7c397`.
- TrajectoryCache는 [RLinf](https://github.com/RLinf/RLinf/blob/034579cbfc4643f72c184ffc06458f788c09b3e6/rlinf/data/storage/replay/buffer.py) commit `034579cbfc4643f72c184ffc06458f788c09b3e6`의 코드를 그대로 재사용한다. 출처·Apache 2.0 라이선스를 vendor에 포함한다.
- RoboCasa365 수집, 성공 구간 추출, native π₀.₅ learner 연결, HF 전송과 orchestration은 이 프로젝트의 연결 코드다. RLinf의 공식 DAgger 예시는 π₀용이며 현재 파이프라인은 성공 실행만 학습하는 online BC다. 전문가가 방문 상태를 다시 라벨링하는 DAgger로 표시하지 않는다.
- 기본 가중치는 [RoboCasa 공식 체크포인트](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000)를 사용한다.

## 실행 설정과 흐름

1. 초기 BC는 생략한다 (`--bootstrap-steps 0`, coordinator `bootstrap_weights=false`). 기존 성공 데이터는 replay에 보존하지만 새 롤아웃을 먼저 수집한 뒤 온라인 업데이트에서 함께 사용한다.
2. AMP·v4의 π rollout server가 각각 simulator worker 2개로 병렬 실행. 첫 수집은 16개씩 총 32회, 이후에는 4개씩 총 8회 단위로 수집한다. 유효 새 성공 8개에 도달하면 업데이트하며 라운드당 최대 32회 시도한다. 상한까지 부족하면 학습 job을 발행하지 않고 데이터를 보존한다. 첫 라운드는 32회 수집 후 성공 1개 이상이면 50 updates 파일럿을 진행한다. 전체 태스크 명령으로 컵을 잡은 뒤 컵 놓기 지시문으로 전환하고 컵 놓기에 성공하면 종료한다.
3. 각 rollout 서버에서 성공한 컵 구간만 추출·검증·압축해 HF Bucket으로 직접 업로드한다. Lab은 job을 조정하며 rollout 데이터를 한 곳에 모아 다시 업로드하지 않는다.
4. Learner가 새 데이터와 최근 replay를 사용해 컵 놓기 50 updates를 수행하고 LoRA 및 optimizer를 저장·업로드한다.
5. 실행 중인 에피소드가 모두 끝난 다음 adapter를 명시적으로 교체한다. 다음 라운드는 새 버전만 사용한다.

LoRA는 native `gemma_300m_lora`, rank 32, trainable 22,118,400 parameters다. 기본 모델은 고정하며 AdamW lr 1e-4, grad clip 0.5, batch 1을 사용한다. native horizon 50, 실행 replan 16, denoising 10회다. Replay는 성공 에피소드 최대 256개, 에피소드 균등 샘플링이다. 성공이 없는 라운드에서도 기존 성공 replay를 사용할 수 있으며 실패한 액션을 정답으로 학습하지 않는다.

학습과 rollout은 별도 GPU 프로세스다. 동일 3090을 사용할 경우 inference 프로세스를 종료한 뒤 gradient를 계산하고 다시 실행해야 한다. 기본 coordinator 구성은 AMP rollout + 별도 SKKU learner다.

HF Bucket: `khmin101/vla-rollout-transfer`
실험 prefix: `pi05-cup-online-bc-20261001`
초기 데이터: `bootstrap/pi05`, `bootstrap/xiaomi`
Jobs: `jobs/pi05/round-NNNN`; weights: `weights/pi05/round-NNNN`; 새 수집: `data/pi05/round-NNNN`.
`round-0000` weights는 초기 BC를 명시적으로 활성화했을 때만 생성한다. 현재는 초기 BC를 생략하므로 사용하지 않는다.

## 완료한 검증

- 실제 초기 데이터 235청크의 camera dtype/shape, finite state/action, 마스크·구간 경계, 원래 실행 액션과 정렬 검사.
- 상태 표현 검사: native π 16D quaternion / Xiaomi 14D axis-angle.
- π 체크포인트의 zero-std action dim 10 검사. Xiaomi의 최대 0.0078125 jitter만 native mean으로 정규화하며 0.02 이상 실제 명령은 오류로 중단한다.
- 성공 순서·잡힌 컵의 오판 방지·연속 유지·Replay FIFO 및 길이 증가·전송 checksum 및 경로 검사: 9 tests 통과.
- 실제 GPU에서 컵 놓기 2 updates, finite loss/grad, LoRA 갱신 확인.
- 전체 고정 backbone fingerprint가 동일함을 확인.
- 저장·복원 후 파라미터 동일, 동일 배치와 RNG로 optimizer를 재개한 결과 동일.
- 별도 native inference 프로세스에서 3개 병렬 HTTP 요청을 보내 모두 finite 50×12 actions와 기대 policy version 수신.
- 다음 라운드의 새 장면 8개 실제 reset 성공.

검증 보고서는 `verification/pi05-cup.json`이다. 위 검증은 기존 AMP 3090에서 수행했다. 새 SKKU 환경의 설치·CUDA 검증은 세션을 제공받은 후 같은 검사를 실행한다. 검증 통과가 과제 성능 향상을 뜻하지는 않는다.

## SKKU 세션에서 시작

Kit을 세션에 복사한 뒤:

```bash
bash scripts/start-pi-cup-learner.sh setup
bash scripts/start-pi-cup-learner.sh check
bash scripts/start-pi-cup-learner.sh download
export HF_TOKEN_FILE=/path/to/existing/private/hf-token
bash scripts/start-pi-cup-learner.sh validate
bash scripts/start-pi-cup-learner.sh start
```

본 학습용 새 가중치는 기본 RoboCasa π₀.₅ + zero-B LoRA에서 시작한다. 검증용으로 갱신한 adapter는 본 학습 초기값에 사용하지 않는다. 세션 환경의 CUDA_VISIBLE_DEVICES를 유지한다. Learner는 HF로 outbound 통신하므로 데이터 전달을 위한 inbound 서비스 프록시는 필요하지 않다.

Lab coordinator:

```bash
python3 -m online_bc.orchestration.online_rounds --config configs/pi05-cup-coordinator.json --rounds 5
```

사용자 서버 정보가 들어오면 연결 후 새 세션 검증을 실행하고 learner와 coordinator를 함께 시작한다. 세션 종료 시 checkpoint와 HF 자료로 재개한다.

## Lab control and optional review

Canonical code: existing GRPO repository, `online-bc/`, deployed at Lab `/home/aiden/Desktop/lab/robot/robocasa-docker/online-bc/`.

- `build_cup_dataset.py`: extract successful cup placement segments after the held-mug precondition.
- `replay.py`: sample initial and online success episodes together; apply current exclusions before every sample. Maximum 256 episodes.
- `train_online_bc.py`: native updates; record exact episode IDs, exclusion revision, and sample usage.
- `pi05_backend.py`: native OpenPI flow matching, action-expert LoRA, frozen backbone, optimizer resume.
- `worker.py`, `docker_worker.py`: eight attempts per round, three simulator workers. Only successful validated segments are packed and uploaded directly to HF.
- `online_rounds.py`: Lab collection coordination, video publication, learner jobs, adapter reload.
- `review_server.py`, `review.html`: optional interactive review. Unreviewed successes are accepted automatically. Exclude, restore, or pause through durable controls.
- `control_sync.py`, `data_control.py`: learner polls HF controls every ten seconds. A control channel older than 45 seconds pauses sampling. An in-flight or previously completed update is not undone by an exclusion.
- `review_catalog.py`: initial 90 videos / 11 eligible segments and subsequent online round videos. Only videos and small metadata come to Lab; action/observation arrays go directly from workers to HF.

Bootstrap BC is skipped; initial success data remain available for later online updates. First pilot: five online rounds, fifty optimizer updates per round. Labels are successful executed actions, not expert corrections. Standard BC can also use correct expert actions collected during a failed episode; this pipeline has no correcting expert, so only successful skill segments are accepted.

Dataset candidates and actual training usage are distinct. `data-usage.json` records sampled episodes; the UI shows learner-confirmed usage and the revision actually received. No production training runs until the learner GPU session is supplied. Review integration must pass its own tests in addition to previous native GPU verification.

## SKKU 최초 연결

`bash setup-skku-online-bc.sh --connect`로 Tailscale userspace와 SSH를 설정한다. 로그인 링크에서 기존 Lab과 동일한 계정으로 인증한 뒤 출력된 IP를 전달한다. Lab의 기존 HF 토큰을 세션의 `~/pi-cup-online-bc/secrets/hf-token`에 복사한 뒤 `bash setup-skku-online-bc.sh --install`로 코드를 받아 설치한다. 초기 BC와 학습 자동 시작은 하지 않는다.

## 버전 확인 (2026-10-02)

- RoboCasa OpenPI: `5a6beda9ff99da30b4e1b59320f6a32971d7c397` — 공식 main HEAD와 일치.
- RLinf: `034579cbfc4643f72c184ffc06458f788c09b3e6` — 공식 main HEAD와 일치.
- Physical Intelligence 원본 OpenPI main: `215abfb217dbac7d5f1273282331b9b1866c0479`. RoboCasa 포크와 별도 저장소이며 자동 교체하지 않는다. 학습 환경 의존성은 검증한 버전으로 고정한다.

고정 30개 평가 시드(992001~992030)는 학습 replay에 넣지 않는다. 기본 모델(version 0)과 파일럿 마지막 체크포인트(version 5)를 v4에서 평가해 HF evaluation 경로에 metrics를 저장한다. 초기 수집 32회와 평가 30회는 별도 집계다.

## 야간 운영 변경 (2026-10-02)

사용자가 평가·성능 정체 대응·속도 최적화·계속 학습을 승인했다. 초기 BC 생략은 유지한다.

- 현재 서비스는 최대 20 rounds / 1,000 updates까지 실행한다. 완료 시 자동 점검에서 평가를 보고 계속 실행 여부와 다음 실험을 결정한다. 사용자 수동 선별은 선택이고 exclusions/pause는 항상 반영한다.
- 매 50 updates 후 v4에서 고정 시드의 첫 10개 quick eval. 매 100 updates 및 마지막에는 고정 30개 full eval. 같은 10개 시드의 결과를 비교하므로 10회·30회 전체 평균을 직접 비교하지 않는다.
- v4 eval 동안 AMP는 다음 라운드의 8회 수집을 전부 맡는다. eval 후에는 두 서버로 다시 분배한다. 영상 게시도 학습 job 제출과 별도 background 작업으로 실행한다. v4는 eval이 끝나야 같은 정책 서버에서 수집하며, 실행 도중 weights를 바꾸지 않는다. SKKU는 별도 GPU learner다. 정상 단계의 병렬화이며 '항상 최적'이라는 보장은 하지 않는다.
- `orchestration/adaptation.py`: 연속 두 체크포인트가 개선하지 않으면 작은 학습률 반감 실험(하한 1e-5, 두 버전 cooldown). Full 30회에서 best 대비 6회 이상 성공 감소 시 다음 learner job은 best checkpoint/optimizer에서 재개한다. Version 0이면 zero-B LoRA로 복원한다. 소규모 평가를 통계적 확증으로 표현하지 않는다. `adaptation-version-*.json`, job, weights metadata에 변경을 남긴다.
- `validation/benchmark_pi05.py`: 임시 adapter로 A100 batch 1/2/4 각 6 native updates를 비교했다. 워밍업 후 중간값 0.132/0.218/0.365 s/update, 7.56/9.18/10.95 samples/s. Batch 4를 적용했다. 실제 업데이트 수는 라운드마다 50회이고 sample 수는 200개다. 검증/benchmark adapter는 본 학습 초기값에 쓰지 않는다.
- 추가 batch 8/16 검증도 같은 GPU lock 아래 임시 adapter로 통과했다. 각각 0.650/1.207 s/update, 12.31/13.25 samples/s이며 6회 중 마지막 4회로 측정했다. 처리량은 높지만 고정 50 updates의 실행 시간과 학습 sample 수가 늘어난다. 전체 시간은 rollout 수집이 지배하므로 현재 실험은 batch 4를 유지한다. Batch 16은 처리량 후보이며 학습 성능 향상으로 확인된 설정은 아니다. 측정 기록: `reports/verification/a100-batch-throughput-20261002.json`; W&B에는 production gradient 기록과 구분한 `pipeline/benchmark_*`로 저장한다.
- Rollout `result.json`의 `phase_seconds`는 환경 생성/reset, 관측 NPZ 저장, 추론, 환경 step, history, 성공 판정, 영상 encoding, 결과 저장을 따로 기록한다. 환경 step 시간에는 관측 렌더링이 포함되므로 물리 연산 시간으로 해석하지 않는다. 신규 batch의 평균은 `collection/episode_mean_*_seconds`로 W&B에 전달한다. 구버전 결과와 섞이면 `profiled_episodes`만 분모로 사용한다. 액션·관측 실행 순서와 성공 로직이 계측문을 제거한 AST 비교에서 동일함을 확인했다. 실행 중인 프로세스는 재시작하지 않으며 다음 collector부터 적용된다. 실제 측정 없이 카메라 갱신 주기나 관측 데이터를 줄이지 않는다.
- AMP worker 2→4 실험은 runtime `benchmarks/amp-worker-trial/status.json`에 기록한다. 다음 AMP 수집 batch만 4개로 실행하고, `worker-3.log` 생성으로 실제 시작을 확인하면 설정은 즉시 2개로 돌린다. `watch.pid`/`watch.log`를 먼저 확인해 중복 실험을 만들지 않는다. 시작되지 않으면 20분 뒤 복원한다. Collection timings의 `simulator_workers`는 빈 shard를 제외한 실행 워커 수다. 비교 시 episode 길이·총 sim steps·policy version·GPU/CPU 경합을 함께 기록하며 서로 다른 seed의 한 batch만으로 최적 구성을 확정하지 않는다.
- 위 일회 실험은 완료됐다. 같은 policy version 2에서 AMP 4 workers는 13,263 control steps / 316.6s = 41.90 steps/s, 2 workers는 10,408 steps / 440.7s = 23.62 steps/s였다. 서로 다른 seed의 두 batch 비교이며 최적 설정의 확증은 아니다. 메모리 관측 최대 15,119MiB, 오류 없이 완료해 AMP의 다음 collector부터 4 workers를 잠정 적용했다. V4는 2 workers를 유지하고 평가 중 정책을 바꾸지 않는다. 현재 실행 중이던 2-worker batch는 그대로 마친다. 비교 기록은 `reports/verification/amp-worker-same-policy-20261002.json`; 계속 처리량·GPU 메모리·실패 로그를 관찰한다.
- 정책 교체와 비동기 평가의 순서 경합을 수정했다. `online_rounds.py`는 v4의 직전 평가 결과를 회수한 뒤 가중치를 교체한다. `rollout/worker.py`는 수집·평가 동안 shared file lock, reload 동안 exclusive lock을 유지한다. 이미 실행 중이던 구버전 worker는 같은 config의 `/proc` PID를 확인해 종료를 기다리므로 활성 coordinator/평가를 재시작하지 않고 보호한다. Lock을 연 신규 reader는 PID 대기 대상에서 제외해 reload를 기다리는 reader와 교착하지 않는다. 평가 중 다음 라운드의 SKKU 학습은 가능하며 늦게 도착한 평가의 학습률 결정은 다음 제출 job부터 적용된다. 실패 rollout을 성공으로 처리하거나 데이터 정렬/성공 판정을 변경하지 않았다.
- Version 2 full30 결과는 컵6/30, 잡기25/30이며 기본 모델 컵6/30과 같다. 잡기 없이 컵 위치 조건을 충족한 1회는 전체 컵 판정에 포함되지만 잡기 조건부 성공 분자에서는 제외한다. `cup_given_grasp`는 (컵 성공 AND 잡기 성공)/잡기 성공 = 5/25(20%)로 정정했다. 원본24%는 `metrics-original.json`으로 보존하고 revision1 결과를 HF/W&B에 기록했다. 수집 데이터 자격 조건은 바꾸지 않는다. 중앙 W&B writer만 cursor를 유지해 재시작했으며 learner/coordinator/policy/collector는 재시작하지 않았다.
- Version4 full30은 컵4/30, 잡기25/30, 컵|잡기4/25(16%)다. 기본 모델6/30과 같은 시드에서 기존 성공4개를 잃고2개를 얻었다. 첫10개는 컵2/10·잡기8/10이다. 개선을 확인하지 못했지만 확증된 퇴행으로 단정하지 않는다. Native plateau 결정은 다음 LR2.5e-5이며 round5 실제50 updates·metadata에서 확인했다(누적250 updates). `validation/policy_snapshots.py`는 GPU lock 아래 별도 프로세스로 기본/adapter frozen inference cache를 번갈아 쓰며3개 실제 관측에서 각각 원래 액션과 오차0을 확인했다. 추론 약0.096s, 전체29.34s이며 gradient 업데이트0회다. 컵 스킬 전환 라우팅·롤아웃 개선 검증은 아직 진행하지 않았고 production 정책 동작은 그대로다. 보고서: `reports/verification/eval-version4-20261002.json`, `policy-snapshots-20261002.json`.
- `models/phase_policy.py`와 `validation/shadow_phase_eval.py`는 선택적 base-prefix 비교를 구현한다. 기본 잡기 정책을 frozen snapshot으로 보존하고 기존 16스텝 경계의 cup_skill_start 이후에만 학습한 adapter를 선택한다. 서버 기본값은 standard이며 새 요청의 variant/phase/version 응답을 확인한다. Unit 32개와 실제 GPU 관측 3개의 base/adapter 액션 오차0 검증을 통과했다. Shadow runner는 v4의 같은 정책 버전·첫10 평가 완료와 exclusive policy lock을 기다려 기존 reader가 없는 동안만 서버를 교체한다. 전체10회 동안 v4를 예약하므로 다음 v4 수집/reload는 기다릴 수 있다. AMP 수집과 SKKU 서비스는 유지한다. 결과는 별도 shadow-evaluation 경로에 저장하며 replay·표준 평가 합계에 넣지 않는다. 성공률·전환 정렬을 확인하기 전에는 학습 수집에 적용하지 않는다. 실패하면 해당 shadow 컨테이너만 종료하고 표준 정책을 복원한다. 검증: `reports/verification/phase-routing-20261002.json`.
- Base-prefix shadow checkpoint5의 첫10 평가는 컵3/10, 잡기9/10, 컵|잡기3/9, 586.04초로 완료했고 전환 정렬·정책 버전 검사를 통과했다. 같은 시드의 과거 standard5는 컵0/10·잡기8/10이다. 표본과 프로세스 재현성의 한계가 있어 개선을 확정하지 않는다. `--variant standard_control`은 동일한 routing 서버·checkpoint에서 표준 요청 10회를 별도 경로로 평가한다. 서버를 재시작하거나 바꾸지 않고 shared policy lock을 유지하며 native 수집과 공존한다. 다음 adapter reload는 control이 끝나야 진행되며 GPU 경합도 별도로 측정한다. Shadow 결과는 표준 eval와 replay에 합치지 않는다. 영상: `/media/pi-cup-shadow-base-prefix-v5/index.html`.
- AMP round4/batch6에서 루트 디스크 부족으로 로그가 8192 bytes에서 끊겨 Docker worker가 exit120으로 종료됐다. 네 결과·영상·액션·manifest와 v4 업로드가 모두 보존된 것을 확인했다. 완료된 round1~3을 파일별 SHA256 검증 후 `/mnt/data/guest/online-bc-archives/rollouts`로 옮겼으며 기존 호스트 경로는 symlink로 유지했다. AMP simulator의 추가 read-only mount는 과거 symlink도 컨테이너에서 읽게 한다. root에 약8.5GB를 확보했고 실패 로그를 별도로 보존했다. Coordinator는 새 PID로 round4 pending batch6에서 재개했으며, `run_coffee.py`의 기존 result skip 로직으로 다시 시뮬레이션하지 않았다. 이미 등록된 v4 source도 중복 실행하지 않았다.
- `scripts/archive-rollout-data.py`는 AMP에서 singleton lock으로 실행한다. 배포 버전보다 최소 두 라운드 오래되고 모든 batch가 complete/upload manifest를 가진 라운드만 대상으로 전체 tar를 HF `archives/pi05/amp/round-XXXX`에 업로드한다. SDK 업로드 성공 및 원격 파일 크기를 재확인하고 나서 로컬 관측/학습 NPZ를 제거한다. 영상·actions.npy·trace·JSON manifest·기존 성공 shard tar는 로컬에 유지한다. 전체 원본은 HF tar와 SHA256.json으로 복원할 수 있다. 추론 가중치·optimizer 및 SKKU replay는 건드리지 않는다. `uploaded.json`을 먼저 저장해 업로드 이후 실패를 재시도할 때 이미 정리된 원본을 다시 tar로 덮어쓰지 않는다. 최근 두 라운드는 그대로 유지한다. 확인: AMP `storage-archiver.pid/log`, `/mnt/data/guest/online-bc-archives/cloud-stage/*/archived.json`, 두 디스크의 free bytes. Cache도 데이터 디스크를 쓰고 Xet chunk cache는 비활성화한다.
- AMP round6~20은 `/mnt/data/guest/online-bc-live/rollouts`에 직접 저장한다. 기존 rollout 경로는 사전에 빈 디렉토리 symlink로 유지하며 simulator/dataset 컨테이너에 전용 경로만 read-write mount한다. 실제 컨테이너에서 round6 경로로 쓰고 읽는 검사를 통과했다. 진행 중인 round5는 옮기지 않는다. 완료 round4는 전체 파일 SHA256 검증 후 보관 디스크로 옮긴다. 최근 두 라운드의 NPZ 보존 규칙은 유지하며 이후 디스크 용량과 archiver 성공을 계속 확인한다. 20라운드 이후 연장 시 새 경로를 같은 방식으로 준비해야 한다.
- 복구 시 이미 완료된 episode를 재사용한 worker의 `collection_seconds`는 unknown(null), `reused_completed_episodes`를 따로 기록한다. 원래 시뮬레이션 시간을 복구·전송 시간으로 대체해 처리량을 과장하지 않는다. 잘못 기록된 round4/batch6 처리 시간은 collection metrics revision으로 정정한다. 원본 실패 로그·복구 기록은 유지한다.
- JAX persistent compilation cache를 세션 개인 디렉토리에 두어 라운드별 process restart의 재컴파일을 줄인다. GPU 선점은 physical 80GB 기준 35%(약28GB)로 설정했고 fractional 할당의 전체 VRAM과 동일하다고 해석하지 않는다.
- `orchestration/health.py`: 60초마다 AMP/v4 policy health, GPU 메모리·사용률, 최근 rollout log, SKKU learner PID/status, Lab coordinator/telemetry PID를 기록한다. `health.json`, `health-history.jsonl`을 확인한다.
- `orchestration/telemetry.py`: Lab self-hosted W&B 한 run에 loss/grad/batch/lr/속도·수집·eval·정책 변경·host health를 보낸다. 기존 Lab netrc 인증을 메모리에서 재사용한다. W&B SDK 0.30.0(기존 긴 API 키 지원). URL: http://100.86.183.64:8080/aiden-lab-desktop/coffee-online-bc/runs/03a19d3315a4 . SKKU→HF→Lab로 작은 업데이트 상태를 보내므로 SKKU의 userspace Tailscale outbound HTTP에 의존하지 않는다.
- 이 대화의 10분 heartbeat `preparecoffee-online-bc`가 실제 상태를 읽고 문제 원인을 찾고 고치며 공식 문서를 조사한다. 변화 없는 경우 알림을 만들지 않는다. Lab의 실제 학습/평가/health/W&B 프로세스는 데스크톱 대화와 별도로 실행된다.

복구 시 주의:

1. SKKU `/home/work/robot_aiden_260930/pi-cup-online-bc/learner.pid`, `learner.log`, `run/service-status.json` 및 실제 child train process를 먼저 확인한다. 설치 스크립트 start는 `verification.json` 통과를 요구한다. 재개 환경: `PI_CUP_ROOT=/home/work/robot_aiden_260930/pi-cup-online-bc`, 같은 루트의 `secrets/hf-token`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`, `PI_CUP_BATCH_SIZE=4`, `PI_CUP_ROUNDS=20`.
2. Lab runtime `/home/aiden/Desktop/lab/robot/pan-skill-models-20261001/coffee-online-bc/pi-cup-run`의 `coordinator-night-active.log`, pending/collection/status를 확인한다. 초기 인계 시 AMP collector PID2650016을 기존 로그로 join했다. 현재 PID는 pid 파일과 실제 ps로 다시 확인한다. 완료 summary가 있는 파일을 덮어쓰지 않는다. `pending-round-*`의 external PID는 실행 당시 정보를 보존하는 용도다.
3. v4 전송 Python은 `/home/v4/skku-vla-grasp-only-20260930/transport/bc-venv/bin/python`(Python3.11/Hub2.1.1). 이전 root-owned venv는 Python3.8이고 HF 라이브러리가 없어 baseline 업로드에 실패했다. 결과를 보존한 채 환경을 새로 만들고 upload만 재시도해서 복구했다. Baseline: 컵6/30, 잡기27/30, 컵|잡기6/27. 평가 duration은 기존 프로세스 시작을 관측하지 못해 unknown이며 coordinator adoption 시간과 혼동하지 않는다.
4. 새 성공8개를 max32 시도에서 확보하지 못하면 정상 data gate가 learner job을 만들지 않는다. 자동 점검은 보존된 shards를 먼저 확인하고 시도 상한을 64 또는96까지 늘리는 수집 실험을 할 수 있다. 현재 seed stride100이므로 상한100 미만을 유지해 라운드 간 중복을 막는다. 현재 코드는 수집 부족 시 32→64→96 상한 확장을 자동으로 기록한다. 실패 action을 성공으로 처리하거나 official success 조건을 약화하지 않는다. 초기 BC 생략 상태에서 첫 라운드도 새 성공이 최소1개 필요하다.
5. Best weights 경로는 HF `weights/pi05/round-XXXX`와 SKKU training/round-XXXX에 보존한다. Eval은 `evaluation/pi05/version-XXXX`, 학습 데이터는 `data/pi05/<node>/round-XXXX/batch-XX`. 두 경로를 합치지 않는다.

확인한 공식 자료: [JAX compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html), [W&B SDK](https://github.com/wandb/wandb), [W&B long-key issue](https://github.com/wandb/wandb/issues/11614), [RLinf](https://github.com/RLinf/RLinf). 추가 조사는 실제 bottleneck/failure에 맞춰 진행하고 변경 전후의 측정값을 보존한다.
