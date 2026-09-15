# GR00T N1.5 vs Xiaomi-Robotics-1 — RoboCasa365 CloseBlenderLid 동일 task 비교

## 요약

NVIDIA GR00T N1.5의 **공개 RoboCasa365 파인튜닝 체크포인트**를 그대로 로드해
CloseBlenderLid를 우리 Xiaomi RoboCasa365와 **동일 task·동일 robot(single-arm Panda+Omron)**
규격으로 추론했다. base(GR00T-N1.5-3B)로는 추론하지 않았고, 직접 파인튜닝/학습도 하지 않았다.

| 모델 | task | split | robot | n_ep | 성공 | 성공률 |
|------|------|-------|-------|------|------|--------|
| **Xiaomi-Robotics-1-RoboCasa365** | CloseBlenderLid | pretrain | Panda+Omron | 50 | 18 | **36.0%** |
| **GR00T N1.5 (RoboCasa365 multitask ckpt)** | CloseBlenderLid | target | Panda+Omron | 50 | 11 | **22.0%** |
| **GR00T N1.5 (RoboCasa365 multitask ckpt)** | CloseBlenderLid | pretrain | Panda+Omron | 50 | (실행 중) | (실행 중) |

> 주의: Xiaomi 공식 CloseBlenderLid 36%는 **split=pretrain scenes** 기준(target50 task 집합,
> pretrain scene에서 평가). 직접 비교를 위해 GR00T도 **split=pretrain**을 별도로 돌리는 중이다.
> split=target(위 22%)은 target scene(다른 씬 randomization) 기준이라 Xiaomi 36%와 scene 조건이 달라
> 1:1 비교가 아니다. pretrain 결과가 진짜 apples-to-apples.

## 무엇을 확인했나 (검증된 사실)

- **동일 task/embodiment 비교가 성립한다.** GR00T N1.5의 RoboCasa365 공식 지원은
  robocasa-benchmark/Isaac-GR00T fork(commit 9d7d7a9)로, single-arm **Panda+Omron**
  embodiment(`panda_omron` data_config) + `robocasa/CloseBlenderLid` gym env를 쓴다.
  이는 우리 Xiaomi RoboCasa365 target50와 동일한 robot·task·predicate 계열이다.
  (혼동 주의: Isaac-GR00T의 GR-1 humanoid tabletop 24-task 벤치는 별개이며 우리 대상이 아니다.)
- **CloseBlenderLid는 RoboCasa365 target50(atomic_seen)에 포함**되어 우리 결과와 직접 비교 가능.
- **동일 predicate 기준.** GR00T eval의 성공 판정은 gym_wrapper의 `env._check_success()`
  → 우리 vendored robocasa(동일 commit 4f8a2980)의 CloseBlenderLid fixture 판정과 같은 소스.
  즉 성공 정의(lid_on_blender: 닫힘 위치 <0.04m AND upright ≤7°)가 양쪽 동일하다.
- **replan/action chunk 일치.** GR00T n_action_steps=16 = Xiaomi replan_steps=16.

## 성능 해석 (정직 보고)

- split=target(다른 scene 조건)에서 GR00T N1.5 = 22%(11/50). 리더보드의 GR00T N1.5
  Atomic-Seen 50.7%는 18개 atomic task **평균**이고, CloseBlenderLid 단일 task 수치는 아니다.
  단일 task는 task 난이도 편차가 커서 평균보다 낮을 수 있다.
- Xiaomi(36%, pretrain) vs GR00T(22%, target)은 **scene split이 달라** 이 표만으로 우열을
  단정하면 안 된다. pretrain split GR00T 결과가 나오면 그 값으로 비교/갱신한다.
- 성공 포장 없음: GR00T가 RoboCasa365 파인튜닝 후에도 CloseBlenderLid 단일 task에서
  Xiaomi보다 낮게 나올 개연성이 있으며, 최종 수치로 정직히 보고한다.

## skill별 + 성공 후 hold — 지원 범위 (한계 명시)

- GR00T 공식 eval 하네스(run_eval.py / gym_wrapper)는 **full-task instruction만** 사용한다.
  instruction = `env.get_ep_meta()["lang"]`(전체 task 지시, 예 "close the blender lid").
  - **skill 단위 지시**(grasp/move_holding/place) 주입 경로가 GR00T 하네스에 없다.
    우리 skill_eval.py의 skill instruction 주입·seed 고정·chained 초기상태는 Xiaomi 전용 하네스 기능.
  - **성공 후 hold(post-success 100스텝)** 도 GR00T 하네스에 없다(성공 시 episode 종료).
- 따라서 skill×hold는 GR00T eval 하네스를 그대로 쓰는 한 **미지원**이다. GR00T를 우리
  skill_eval.py 경로에 얹으려면 action space/normalization 정합(별도 어댑터)이 필요하며,
  이는 별도 작업으로 분리 권장. 현재 산출물은 full-task 동일 비교에 집중했다.

## 재현 정보 (provenance)

- 체크포인트: HF `robocasa/robocasa365_checkpoints`
  `gr00t_n1-5/multitask_learning/checkpoint-120000` (weights only, optimizer.pt 제외).
  shard sha256(선두): 08f1891947.., deb9c9cf40.. / embodiment_tag=new_embodiment.
- vendor: robocasa-benchmark/Isaac-GR00T @ 9d7d7a9 (RoboCasa365 eval fork).
  robosuite @ 5ce6643, robocasa @ 4f8a2980 (robocasa-sim 이미지와 동일 commit).
- 이미지: groot-eval:t_ff372eea (sha256:980a0893cdae..), base=xiaomi-cu121:t_9f03a613
  (torch 2.5.1 cu121 + flash-attn 2.8.3 재사용). driver 535.183.01, RTX 3090 24GB.
- 실행: `bash groot/scripts/run.sh eval-task 50` (GROOT_SPLIT=target|pretrain).
  server+client 단일 프로세스·단일 GPU, 정책 VRAM ~5.5GB.
- 산출물: `groot/results/CloseBlenderLid/<split>/stats.json`(per-episode success 포함),
  MP4 50+개, SHA256 매니페스트(로컬 `videos/CloseBlenderLid/groot_eval/`).
