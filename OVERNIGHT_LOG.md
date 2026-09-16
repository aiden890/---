
## 02:44 KST — adaptive curriculum sweep 정상 가동중 (막힘 없음)
- v4 `adaptive_sgd2e3`(SGD 2e-3) + amp_csi `adaptive_adamw1e5`(AdamW 1e-5) 둘 다 살아있음. GPU v4 11.9G/24G(25%), amp 10.8G/24G(0%) — 여유 충분, OOM/RPC drop 없음.
- 둘 다 EVAL before 단계(seed 5000대 고정 held-out), v4 10/20 진행. 아직 train iter 0 → GATED/eval-delta 집계 불가(정상, 초기).
- 커밋 a0d76be(online moving-band) 배포 완료. detached adaptive_finisher(v4 pid 1825574) 양쪽 DONE 대기중 → 완료시 자동 집계(BAND/adaptive_aggregate).
- 조치: 없음(정상 초기 단계). 다음 cron서 iter별 n_succ 밴드분류·GATED비율·mixed% 곡선 확인 예정.

## 03:46 KST — adaptive curriculum 2-arm 페어드 스윕 정상 진행중 (막힘 없음, 조치 없음)
- v4 `adaptive_sgd2e3`(SGD 2e-3) it=14/30, amp_csi `adaptive_adamw1e5`(AdamW 1e-5) it=13/30. 페어드 재현성 유지(iter별 동일 seed, 예 it13=1222 양쪽).
- 프로세스/컨테이너 살아있음(v4 sweep+trainer+finisher pid 1825574; amp client+trainer). GPU v4 10.9G/24G, amp 12.0G/24G, util 0%(rollout 사이). OOM/RPC drop/에러 0.
- GATED 비율(it0-13): v4 8/14=57%, amp 9/14=64% vs baseline 68% — 소폭 개선(아직 결론X). 커리큘럼 자기교정 관측: seed 1092=mid로 발굴→it8-11,14 exploit이 mixed 배치(real gradient) 생성, 후반 mixed% 상승.
- post-step: v4 it14 grad_norm=2464(높음) but adapter_dL2=0.001(실제 업뎃 미미)→느리지만 안정, 발산X.
- 조치: 없음(정상, ~절반 진행). 개입=페어드 비교 파괴+1h 낭비라 안 함. finisher가 양쪽 30 iter→EVAL(after)+heldout→ADAPTIVE_FINAL.md 자동집계 대기. 다음 cron서 sentinel/eval-delta 확인.

## 04:48 KST — adaptive curriculum 페어드 스윕 거의 완료, 막힘 없음 (조치 없음)
- v4 `adaptive_sgd2e3` it=27/30, amp_csi `adaptive_adamw1e5` it=25/30. 둘 다 살아있음(v4 client+trainer+finisher pid 1825574; amp client+trainer 컨테이너 2h+). GPU v4 10.9G/21%, amp 13.2G/52%. OOM/RPC drop/에러 0.
- GATED 비율(it0-25 공통): 10/26 = 38% vs baseline 68% — 커리큘럼이 GATED 절반 가까이 줄임(mixed iter 다수 확보). 자기교정 관측: seed 1092가 explore로 발굴→it8-24 exploit이 mixed(3~5/8) 안정 배치, seed 1175·1181도 mid로 편입.
- **핵심 페어드 관측(optimizer 대조)**: AdamW arm은 adapter_dL2 0.009→0.002로 실제 어댑터 이동(real update). SGD 2e-3 arm은 dL2 flat 0.00100 — 사실상 정지(학습 미미). 같은 seed·같은 mixed 배치인데 AdamW만 파라미터가 움직임 → AdamW가 이 세팅서 유효 LR 확보, SGD 2e-3는 너무 작음. after/heldout eval delta로 이 차이가 정책 개선으로 이어지는지 확인 예정.
- post-step 안정: 양쪽 post_kl 0.16~0.62, clip 0.84~0.98, ratio 0.81~1.21 — 발산 없음. grad_norm 높으나(660~6100) dL2로 실제 업뎃은 절제됨.
- 조치: 없음(정상, ~90% 진행). 개입=페어드 비교 파괴. finisher가 30 iter→EVAL(after)+heldout→ADAPTIVE_FINAL.md 자동집계 대기. 다음 cron서 ADAPTIVE_ALL.DONE sentinel + before/after delta 확인 후 개선 여부 정직 판정.

## 17:02 KST — exp1 DONE(정직 판정) → exp2 AdamW LR-ladder 착수
- exp1(30 iter, optimizer 대조) 완료. **1차목표 달성**: GATED 68%→33%(SGD)/36%(AdamW), mixed 63%. 커리큘럼이 GRPO 신호 복구는 확실히 함.
- **2차목표(고정 eval 개선) 미달, 노이즈 바닥**: SGD 2e-3 adapter_dL2=0.001(사실상 정지)→grasp 0.60→0.55 REGRESSED. AdamW 1e-5 dL2=0.0038(유일하게 어댑터 이동)→grasp+0.05, official 0.05→0.15. 병목=신호부재 아니라 **업뎃 강도**로 지목(단일변수 optimizer 대조).
- 조치: exp2 착수 — optimizer를 mover(AdamW) 고정, **LR만** 변수. v4 adaptive_adamw3e5(3e-5) + amp_csi adaptive_adamw5e5(5e-5), paired seed 12345. 둘 다 실행중(EVAL-before 단계). finisher(파라미터화) detached 집계 대기. 코드 커밋 e8ca603, 양서버 배포. exp1 리포트 REPORT/에 보관.
