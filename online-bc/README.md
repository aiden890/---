# π₀.₅ · PrepareCoffee 컵 놓기 Online BC

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

1. 위 성공 데이터로 bootstrap 50 optimizer updates.
2. AMP의 π rollout server에서 새 장면 8개를 simulator worker 3개가 병렬 실행. 전체 태스크 명령으로 컵을 잡은 뒤 컵 놓기 지시문으로 전환하고 컵 놓기에 성공하면 종료한다.
3. 각 rollout 서버에서 성공한 컵 구간만 추출·검증·압축해 HF Bucket으로 직접 업로드한다. Lab은 job을 조정하며 rollout 데이터를 한 곳에 모아 다시 업로드하지 않는다.
4. Learner가 새 데이터와 최근 replay를 사용해 컵 놓기 50 updates를 수행하고 LoRA 및 optimizer를 저장·업로드한다.
5. 실행 중인 에피소드가 모두 끝난 다음 adapter를 명시적으로 교체한다. 다음 라운드는 새 버전만 사용한다.

LoRA는 native `gemma_300m_lora`, rank 32, trainable 22,118,400 parameters다. 기본 모델은 고정하며 AdamW lr 1e-4, grad clip 0.5, batch 1을 사용한다. native horizon 50, 실행 replan 16, denoising 10회다. Replay는 성공 에피소드 최대 256개, 에피소드 균등 샘플링이다. 성공이 없는 라운드에서도 기존 성공 replay를 사용할 수 있으며 실패한 액션을 정답으로 학습하지 않는다.

학습과 rollout은 별도 GPU 프로세스다. 동일 3090을 사용할 경우 inference 프로세스를 종료한 뒤 gradient를 계산하고 다시 실행해야 한다. 기본 coordinator 구성은 AMP rollout + 별도 SKKU learner다.

HF Bucket: `khmin101/vla-rollout-transfer`
실험 prefix: `pi05-cup-online-bc-20261001`
초기 데이터: `bootstrap/pi05`, `bootstrap/xiaomi`
Jobs: `jobs/pi05/round-NNNN`; weights: `weights/pi05/round-NNNN`; 새 수집: `data/pi05/round-NNNN`.
`round-0000` weights는 초기 50회 BC 이후의 bootstrap adapter다.

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
bash start-pi-cup-learner.sh setup
bash start-pi-cup-learner.sh check
bash start-pi-cup-learner.sh download
export HF_TOKEN_FILE=/path/to/existing/private/hf-token
bash start-pi-cup-learner.sh validate
bash start-pi-cup-learner.sh start
```

본 학습용 새 가중치는 기본 RoboCasa π₀.₅ + zero-B LoRA에서 시작한다. 검증용으로 갱신한 adapter는 본 학습 초기값에 사용하지 않는다. 세션 환경의 CUDA_VISIBLE_DEVICES를 유지한다. Learner는 HF로 outbound 통신하므로 데이터 전달을 위한 inbound 서비스 프록시는 필요하지 않다.

Lab coordinator:

```bash
python3 online_rounds.py --config configs/pi05-cup-coordinator.json --rounds 100
```

사용자 서버 정보가 들어오면 연결 후 새 세션 검증을 실행하고 learner와 coordinator를 함께 시작한다. 세션 종료 시 checkpoint와 HF 자료로 재개한다.
