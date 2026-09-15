# Skill-conditioned CFM LoRA SFT (task t_e5cd5736) — INDEX

주 모델: pinned XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 rev 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4.
산출물은 rl-train-t_3ed65912 안에만 확장(data/skill_sft, results/skill_sft, src/scripts/REPORT). 새 중복 프로젝트 없음.

## 상태 요약 (2026-09-15, run29)
- Goal 1 (데이터 분할/manifest): PASS. 106 CloseBlenderLid demos, gripper-command 기반 3-skill 분할, 74/16/16 traj split, action loss mask, SHA-256 manifest. RELEASE 데이터 존재.
- Goal 2 (conditioning): PASS. overall goal + skill_id + NL instruction + object/dest/constraints -> 3 arm(nl_only/nl+skill_id/shared_lora+learned_embedding).
- Goal 3 (per-skill DiT LoRA CFM SFT): 구현 PASS, GPU 게이트는 아래.
  - Gate 1 tiny-overfit: PASS (fresh adapter, 3 demos x700 step, base MC-loss 1.20 -> 0.024, drop 50.0x).
  - Gate 2 heldout val: PASS (74-demo 1epoch 학습, 16 heldout demos: 1.370 -> 0.218, 6.3x, 16/16 개선).
  - Gate 3 valid entry-state rollout: **FAIL (net 개선 없음)**. skill_sft_rollout_findings.md 참조.
- Goal 4~6: 미착수. 정책상 rollout 게이트 통과(개선 확인) 뒤에만 N=50/ARM 장기 실험 허용 -> 현재 보류.

## 재현 (v4, /home/v4/rl-train-t_3ed65912)
사전 스테이징(오프라인 데이터/deps, 최초 1회):
  bash scripts/skill_sft_stage_data.sh
트레이너 기동(adapter_only, LoRA는 optimizer state 작아 AdamW OK):
  bash scripts/run-train.sh trainer-start --optimizer adamw --lr 1e-4 --train-mode adapter_only --rank 8
Gate 1:
  bash scripts/run-train.sh sft overfit_grasp --op overfit --skill GRASP_HANDLE --arm nl_plus_skill_id --demos 3 --steps 700
Gate 2 (baseline은 fresh 트레이너에서 --op val; trained는 --op train 후 --op val):
  bash scripts/run-train.sh sft train_grasp_nlid --op train --skill GRASP_HANDLE --arm nl_plus_skill_id --epochs 1 --ckpt /out/grasp_nlid.pt
  bash scripts/run-train.sh sft val_grasp_trained --op val --skill GRASP_HANDLE --arm nl_plus_skill_id --val-n 16
Gate 3 (sim client, before=zero-init base / after=trained adapter):
  bash scripts/run-train.sh train sft_rollout_gate_grasp --iters 0 --eval-n 12 --heldout-n 0 --eval-split target --train-skill grasp --load-ckpt /train/results/skill_sft/ckpt/grasp_nlid.pt

## 근거 경로
- data/skill_sft/{data_manifest,skill_segments,closeblenderlid_episodes,conditioning_schema}.json
- results/skill_sft/{overfit_grasp,val_grasp_base,val_grasp_trained}/ , results/sft_rollout_gate_grasp/{eval_before,eval_after}.json
- results/skill_sft/ckpt/grasp_nlid.pt (216 LoRA + optimizer + RNG + data_hash + model_rev)
- src/{grpo_trainer_server,skill_sft_dataset,skill_sft_conditioning,lora,flow_policy}.py, scripts/{skill_sft_train,skill_sft_stage_data}.py, scripts/run-train.sh (sft 서브커맨드)
