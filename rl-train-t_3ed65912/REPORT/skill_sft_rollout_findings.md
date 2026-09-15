# Gate 3 (valid entry-state rollout) 결과 및 판정

## 측정 (동일 12 eval seed, target split, GRASP skill, eta=0 결정론, before/after paired)
| config | LoRA scaling | lr | epochs | heldout CFM loss | GRASP rollout base->trained |
|---|---|---|---|---|---|
| strong (grasp_nlid.pt) | 4.0 (alpha32/r8) | 1e-4 | 1 | 1.370 -> 0.218 (6.3x) | 2/12(16.7%) -> **0/12(0%)** 회귀 |
| gentle (grasp_gentle.pt) | 1.0 (alpha8/r8) | 2e-5 | 1 | (train loss ~0.8-1.2, 거의 미학습) | 1/12(8.3%) -> 2/12(16.7%) ~= base, no-op |

(base rollout는 n=12에서 1~2/12 사이 표본변동. 강한 config는 그 변동 아래로 완전 붕괴, 약한 config는 사실상 base와 동일.)

## 판정: Gate 3 미통과 (net skill-success 개선 없음)
- SFT loss(open-loop BC 목표)는 크게 줄었으나 closed-loop skill success로 전이되지 않음. 오히려 강한 개입은 base 능력을 덮어써 붕괴.
- 전형적 behavior-cloning distribution shift: 부분-스킬 데모(전체 태스크로 훈련된 정책의 하위 구간)로 미세조정 시 compounding error/OOD로 닫힌 루프가 무너짐. 약하게 하면 개입 자체가 소멸.
- 정책 준수: tiny-overfit/게이트 실패 시 장기 SFT 금지에 준해 rollout 게이트를 개선으로 통과하지 못했으므로 Goal 4~6의 N=50/ARM 장기 실험은 **진행 금지**. 정직 보고.

## 확인된(수정 완료) 인프라/코드 이슈 — 결과 신뢰성에 영향 없음(모두 게이트 前 수정)
1. sft 서브커맨드가 트레이너의 --network none 상속 -> 데이터/deps 사전 스테이징 + HF_HUB_OFFLINE=1 (scripts/skill_sft_stage_data.sh).
2. staged huggingface_hub 1.31 vs 이미지 transformers 4.57 충돌 -> hf_hub는 이미지(0.36.2) 사용, pylibs엔 pyarrow/av만.
3. center_crop_np가 원해상도 복원 안 함 -> 243px 프레임이 Qwen3-VL patchify 거부. rollout.py처럼 crop 후 256^2 resize 복원.
4. SFT 클라 스킬명(GRASP_HANDLE) != LoRA 키(grasp) -> set_active_skill 무효 -> backward 실패. 서버 SFT op에 _SFT_SKILL_TO_LORA 매핑.
5. op_save가 트레이너에서 실행되는데 /out은 클라 전용 -> --ckpt /out/... 소실. /train/results/skill_sft/ckpt/로 리맵.
6. overfit 게이트 지표를 노이즈 큰 per-step train loss -> 고정시드 MC(16) val loss 전/후 비교로 교체.
7. op_save에 optimizer/RNG(torch/cuda/numpy)/config/data_hash/model_rev 추가(재현성).

## 다음 후보(operator 결정 필요) — 어느 것도 아직 실행 안 함
A) SFT 레시피 재설계: (a) KL-to-base 정규화로 base 능력 보존, (b) adapter+action-expert(arm B) 저lr, (c) DAgger-style on-policy 보정, (d) 더 많은 epoch + 매우 낮은 lr 스윕. 각각 rollout 게이트로 재검증.
B) SFT를 스킬 능력 부여 수단으로 보지 않고, base 정책 위에 곧바로 Goal 6 GRPO(시뮬 리워드)로 스킬을 강화(SFT 건너뜀). 단 브리핑은 선택된 skill-SFT ckpt에 GRPO를 요구 -> operator 승인 필요.
C) 3-arm 비교(Goal 5)를 rollout이 아닌 heldout CFM loss 기준으로만 축소 보고(개선 전이 없음을 명시).
