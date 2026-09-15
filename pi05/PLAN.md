# π0.5 (openpi) RoboCasa CloseBlenderLid 추론 비교 — 실행 계획 (검증됨)

목적: π0.5 RoboCasa365 파인튜닝 체크포인트를 로드해 CloseBlenderLid를
Xiaomi-Robotics-1 / GR00T N1.5와 **동일 task·동일 split**로 추론 비교.
GR00T 카드(t_ff372eea)와 동일한 계약: 별도 checkpoint/output 경로, xiaomi 자원 불침범,
base 파인튜닝 금지 — 공개 365 파인튜닝 체크포인트로 추론만.

## 검증된 사실 (2026-09-15, 오프라인 조사)

### 체크포인트 (HF `robocasa/robocasa365_checkpoints`)
- π0.5: `pi05_pretrain_human300/multitask_learning/75000`
  - 구성: JAX/orbax params(`params/`, ocdbt) + `assets/norm_stats.json` + `_CHECKPOINT_METADATA`.
  - 대용량 → .gitignore. `huggingface_hub`으로 해당 서브트리만 스냅샷.
- (참고) π0: `pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000` — 동일 구조.

### 벤더 포크 — **정정 필요**
- 카드 본문/코멘트는 `Leo428/openpi_robocasa` 브랜치 `robocasa-finetune`를 지목하나,
  그 포크에는 **RoboCasa eval 하네스가 없다**(PickPlaceCounter 파인튜닝 연구용;
  `src/openpi/policies/robocasa_policy.py` + `docs/robocasa_finetuning_research.md`만 존재).
- **올바른 포크는 `robocasa-benchmark/openpi` (main)**. RoboCasa 공식 문서
  (policy_learning_algorithms / multitask_learning)가 지목하는 포크이며, 다음을 모두 포함:
  - config: `pi05_pretrain_human300` (체크포인트 config와 일치), `pi0_robocasa_pretrain_human300`
  - eval 하네스: `examples/robocasa/main.py`, `scripts/serve_policy.py`,
    `examples/robocasa/get_eval_stats.py`, `examples/robocasa/convert_robocasa_to_lerobot.py`
- 결론: **vendor = robocasa-benchmark/openpi @ main (고정 커밋 기록)**. Leo428 포크는 미사용.
  (robocasa_policy 입출력 어댑터 자체는 두 포크 동일 계열.)

### 공식 eval 레시피 (robocasa.ai 문서 = 2-part server+client)
```
# part a: 추론 서버
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config=pi05_pretrain_human300 \
  --policy.dir=<CKPT>/pi05_pretrain_human300/multitask_learning/75000

# part b: eval 클라이언트 (동일 GPU, 별도 프로세스)
python examples/robocasa/main.py \
  --args.port 8000 \
  --args.task_set atomic_seen \
  --args.split pretrain \
  --args.log_dir <OUT>/pi05_closeblenderlid
# 결과 집계
python examples/robocasa/get_eval_stats.py --dir <OUT>/pi05_closeblenderlid
```
- **VRAM**: openpi 추론 권장 8GB (문서). 단일 RTX 3090(24GB)에 충분.
  GR00T(t_ff372eea)와 동시 co-locate 시 합계 확인, 부족하면 순차.
- **split=pretrain** 필수(Xiaomi 공식 수치가 pretrain scene 기준 → apples-to-apples).
  GR00T 카드와 동일하게 pretrain 행이 정식 비교.
- **replan/action horizon**: Xiaomi replan_steps=16과 정합되게 확인
  (openpi robocasa main.py의 action chunk 실행 스텝 = pi05 action_horizon; 서빙 config 확인).
- CloseBlenderLid ∈ `atomic_seen`(= target50). 단일 task만 필터해 추론하려면
  main.py의 task 필터 인자/환경 확인(task_set 단위가 기본).

### 동일-task 비교 성립 근거
- 성공 판정 predicate = 벤더 robocasa CloseBlenderLid fixture(동일 commit 계열,
  `env._check_success()` = lid_on_blender: 닫힘 <0.04m AND upright ≤7°). GR00T/Xiaomi와 동일 소스.
- robot: PandaOmron(single-arm Panda + Omron base). GR00T 카드와 동일 embodiment 계열.

### skill별 + hold — 예상 한계 (GR00T와 동일 구조)
- openpi robocasa 하네스도 **full-task instruction만** 사용(`env.get_ep_meta()['lang']`).
  skill 단위 지시 주입·post-success hold 경로 없음 → 우리 skill_eval.py(Xiaomi 전용) 기능.
- 따라서 skill×hold는 openpi eval 하네스 그대로는 **미지원** 가능성 높음. 실물 확인 후
  정직히 결론(GR00T와 동일 톤: 미지원이면 미지원, 어댑터는 별도 작업 권장).

## 환경(도커) 방침
- openpi는 **JAX 기반** → GR00T의 torch/cu121 이미지 재사용 불가. 별도 이미지 필요.
  - 베이스: CUDA 12.x + `jax[cuda12]` (openpi pyproject 핀 준수) + openpi(-e) + robosuite/robocasa
    (동일 pinned commit: ROBOCASA=4f8a2980, ROBOSUITE=5ce6643f) + EGL 렌더 스택(GR00T Dockerfile 참고).
  - 체크포인트 로드: orbax/JAX. `serve_policy.py policy:checkpoint` 경로 그대로.
- asset volume은 GR00T와 동일 `robocasa-assets-t_9f03a613` (ro) 재사용, 별도 output dir.

## 산출물 (목표)
- `videos/CloseBlenderLid/pi05_eval/` : full-task mp4 + (가능시)steps.jsonl, SHA256 검증 로컬 수신.
- `pi05/REPORT.md` : Xiaomi vs GR00T vs π0.5 동일 task 비교표(성공률/split), skill/hold 지원범위 정직 기술.
- `tracking/rlenv_pi05.json` + 영상 복사로 트래킹 페이지 추가 가능하게.

## 현재 블로커
- **v4 SSH 서명 실패**: 패스프레이즈 키(id_ed25519_macbook_aiden)가 두 ssh-agent 소켓
  (/tmp/ssh-XXXXXX0rvlUn/agent.2039712, /run/user/1000/keyring/ssh) 모두에서
  "agent refused operation" → 서명 불가. 로컬 GPU/도커 이미지 없음 → v4에서만 실행 가능.
- 해제: 사용자가 자기 터미널에서 `ssh-add /home/aiden/.ssh/id_ed25519_macbook_aiden` 실행
  ("Identity added" 확인). 그 후 워커가 소켓 재탐색해 재개.
