# π0.5 (openpi) vs GR00T N1.5 vs Xiaomi-Robotics-1 — RoboCasa365 CloseBlenderLid 동일 task 비교

π0.5(openpi, Physical Intelligence의 RoboCasa365 공개 파인튜닝 체크포인트)를 우리 Xiaomi
RoboCasa365 및 GR00T N1.5와 **동일 task·동일 robot(single-arm Panda+Omron)·동일
predicate·동일 replan(16)·동일 split(pretrain)** 조건에서 추론 비교했다. base 파인튜닝/학습
없음 — 공개 365 파인튜닝 체크포인트로 **추론만**.

## 결과 (apples-to-apples, pretrain split, 50 episodes)

| 모델 | task | split | robot | N | 성공 | 성공률 |
|---|---|---|---|---|---|---|
| **Xiaomi-Robotics-1-RoboCasa365** | CloseBlenderLid | pretrain | Panda+Omron | 50 | 18 | **36.0%** |
| **GR00T N1.5-RoboCasa365** | CloseBlenderLid | pretrain | Panda+Omron | 50 | 10 | **20.0%** |
| **π0.5-RoboCasa365 (openpi)** | CloseBlenderLid | pretrain | Panda+Omron | 50 | 7 | **14.0%** |

성공 episode index (seed=7, replan=16):
- Xiaomi: {검증된 18개, GR00T 카드 기준}
- GR00T N1.5: {5,8,17,21,22,26,27,29,35,48} (0-index; pretrain)
- **π0.5: {5,10,12,17,19,22,27}** (성공시점 step: 426,697,412,686,459,478,671 / horizon 900)

**핵심 결론:** 동일 조건에서 **Xiaomi 36% > GR00T 20% > π0.5 14%**. 세 공개 RoboCasa365
파인튜닝 체크포인트 중 Xiaomi-Robotics-1이 CloseBlenderLid 단일 task에서 가장 강하고,
π0.5가 가장 낮다. π0.5의 14%는 공식 리더보드 π0.5 RoboCasa365 **평균 16.9%**(Xiaomi 논문
Table 3)와 정합적 — CloseBlenderLid는 평균보다 약간 낮은 hard task이며, 성공 포장 없이
수치 그대로 보고한다.

## 재현 방법 (2-part serve+client)

openpi는 **JAX 기반**이라 GR00T의 torch/cu121 이미지를 재사용할 수 없다. 정책 서버는 vendor
JAX 이미지(`openpi-server-rc`)에서, eval 클라이언트는 검증된 sim 이미지(`pi05-client`,
groot-eval + openpi_client)에서 돌리고, 사설 docker 네트워크로 websocket 연결한다.

```
# v4에서, WD=/home/v4/robocasa-docker-t_ee702b9c
bash pi05/scripts/run.sh build-client    # sim + openpi_client 이미지
# (server 이미지 openpi-server-rc = serve_policy.Dockerfile + robocasa registry, 아래 참조)
bash pi05/scripts/run.sh download         # pi0.5 RoboCasa365 ckpt(params+assets)만
bash pi05/scripts/run.sh eval 50          # serve pi0.5 + CloseBlenderLid 50ep
```

### 고정 버전 (git 재현)
- **벤더**: `robocasa-benchmark/openpi @ 5a6beda9ff99da30b4e1b59320f6a32971d7c397` (main).
  RoboCasa 공식 문서가 지목하는 포크 — config `pi05_pretrain_human300` + eval 하네스
  (`examples/robocasa/main.py`, `scripts/serve_policy.py`, `get_eval_stats.py`) 포함.
  (카드 본문이 지목한 `Leo428/openpi_robocasa@robocasa-finetune`에는 RoboCasa eval 하네스가
  없어 PickPlaceCounter 연구용 — 미사용으로 정정. commit 606537d/PLAN.md 참조.)
- **체크포인트**: HF `robocasa/robocasa365_checkpoints`
  `pi05_pretrain_human300/multitask_learning/75000` (JAX/orbax params + assets/norm_stats.json).
  추론에 불필요한 train_state(optimizer 9.1G)는 제외, params(7.7G)+assets만 사용.
- **robosuite/robocasa**: 4f8a2980 / 5ce6643f (GR00T·sim 이미지와 동일 pinned commit).
- **이미지**: `openpi-server-rc:t_ee702b9c`(eb7c01bd93a9), `pi05-client:t_ee702b9c`(25cc71bb6448).
- **asset volume**: `robocasa-assets-t_9f03a613`(ro) 재사용, output은 별도.

### 벤더 소스 수정 2건 (문서화·재현 가능, `pi05/openpi-norm-stats-fallback.patch`)
1. **norm stats fallback 가드** (`src/openpi/training/config.py`): `LeRobotRobocasaDataConfig`가
   추론 시에도 원본 pretrain_human300 데이터셋에서 norm stats를 eager-load하려다
   `FileNotFoundError`로 죽는다. 데이터셋은 추론에 불필요하고 체크포인트가 자체
   `assets/norm_stats.json`을 갖고 있으므로, raw-dataset fallback을 try/except로 감싸
   없으면 None 반환 → `create_trained_policy`가 체크포인트 assets norm stats를 로드하게 했다.
   (로그: `Loaded norm stats from /ckpt/assets` 확인.)
2. **server 이미지 robocasa registry**: vendor serve_policy.Dockerfile venv(JAX/numpy 1.26.4)에
   robosuite/robocasa를 `--no-deps`로 추가. config.py가 dataset-soup REGISTRY dict를
   import하기 때문. server는 sim을 돌리지 않으므로(추론 전용) robocasa `__init__`의
   numpy==2.2.5 assert만 중화. macros_private.py는 setup_macros 대신 직접 복사(순환 import 회피).

## 관측·한계 (정직 기술)

- **GPU**: v4 단일 RTX 3090(24GB). π0.5 JAX 추론은 XLA autotuning이 상당한 여유 VRAM을
  요구 — xiaomi-server(~10.7GB) 동시 점유 시 XLA 컴파일이 `INTERNAL: not supported`(실제로는
  autotuning 중 OOM)로 실패했다. 사용자 승인(2026-09-15) 하에 xiaomi-server를 잠시 stop해
  전량 확보 후 정상 추론(action chunk (50,12) 확인), eval 종료 후 xiaomi-server를 `docker start`로
  **원복**(경로/config 불변). GR00T와의 동시 co-locate는 하지 않았다(순차).
- **skill별 + hold 미지원**: openpi eval 하네스도 GR00T와 동일하게 **full-task instruction만**
  사용한다(`obs["annotation.human.task_description"]` = "Close the lid blender by securely
  placing the lid on top."). skill 단위(grasp/move_holding/place) 지시 주입·성공 후 hold(100스텝)
  경로가 없다. 이는 우리 skill_eval.py(Xiaomi 전용)의 기능이며, π0.5 하네스 그대로는 미지원.
  GR00T 카드와 동일한 결론 — 미지원이면 미지원. (skill×hold 어댑터는 별도 작업 권장.)
- **predicate 동일성**: 성공 판정 = 벤더 robocasa CloseBlenderLid fixture의 `info["success"]`
  (lid_on_blender: 닫힘 <0.04m AND upright ≤7°). Xiaomi/GR00T와 동일 소스·동일 commit 계열.
- **action space**: π0.5는 60-D 패딩 모델 공간에서 뽑아 vendor `convert_action`으로 robocasa
  12-D 액션으로 변환(RobocasaOutputs). state는 eef_pos_rel(3)+eef_rot_rel(4)+base_pos(3)+
  base_rot(4)+gripper_qpos(5) 순서 concat 후 모델 dim으로 패딩 — vendor main.py와 동일.

## 산출물

- `videos/CloseBlenderLid/pi05_eval/` : 50 mp4 + 50 `*_steps.jsonl`(성공시점·predicate 타임라인) +
  `stats_CloseBlenderLid_pretrain.json` + `SHA256SUMS.txt`(101개 전부 로컬 검증 통과).
- `pi05/scripts/eval_closeblenderlid.py` : 단일 task eval 클라이언트(vendor main.py 로직 그대로,
  단일 task로 좁히고 per-step jsonl+SHA256 추가).
- `pi05/docker/Dockerfile.server`, `Dockerfile.client`, `pi05/scripts/run.sh` : 재현 파이프라인.
- `pi05/openpi-norm-stats-fallback.patch` : 벤더 수정 기록.
