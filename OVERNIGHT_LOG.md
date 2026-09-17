
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

## 07:03 KST — exp2 AdamW LR-ladder 페어드 진행중, 막힘 없음 (조치 없음)
- v4 `adaptive_adamw3e5`(AdamW 3e-5) it=14/30, amp_csi `adaptive_adamw5e5`(AdamW 5e-5) it=13/30. 양쪽 client+trainer 살아있음(v4 pid 1915495, finisher pid 1919211), amp 컨테이너 정상. GPU v4 10.9G/24%. OOM/RPC drop/에러 0.
- 페어드 재현성 유지: iter별 동일 seed(it0=1002, it3=1070, it8-11·14=1092, it13=1222 양쪽 동일), LR만 변수.
- **핵심 조기관측(LR→업뎃강도 단조)**: 동일 seed·동일 mixed 배치에서 adapter_dL2가 LR에 비례.
  - it3(s1070,1/8): 3e-5=0.0279 vs 5e-5=0.0464 | it8(s1092,5/8): 0.0213 vs 0.0386 | it9: 0.0175 vs 0.0309 | it10: 0.0151 vs 0.0262 | it11: 0.0131 vs 0.0225.
  - 5e-5 ≈ 1.7x 3e-5, 둘 다 exp1 AdamW 1e-5(dL2 0.0038) 대비 3e-5≈4-7x·5e-5≈6-12x. exp1 결론(병목=업뎃강도)대로 이 LR들은 어댑터를 훨씬 크게 이동시킴 → after/heldout delta가 실제 정책개선 전환 여부 판정.
- post-step 안정: 3e-5 kl 0.19-0.30, 5e-5 kl 0.27-0.35(약간 높으나 발산X), clip 0.82-0.93, ratio 0.87-1.14. dL2가 iter 진행하며 감소(0.028→0.012 등)=수렴 방향.
- 조치: 없음(정상 ~45% 진행, 개입=페어드 비교 파괴+낭비). 남은 ~16 iter→EVAL(after)+heldout→finisher가 ADAPTIVE_FINAL_exp2_lr_ladder.md 자동집계. 다음 cron서 exp2 DONE + before/after delta로 LR별 개선 정직 판정.

## 08:05 KST — exp2 AdamW LR-ladder 마무리 단계, 막힘 없음 (조치 없음)
- v4 `adaptive_adamw3e5`(3e-5) it=28/30, amp_csi `adaptive_adamw5e5`(5e-5) it=25/30. 양쪽 client+trainer 살아있음(v4 trainer pid1915214/client pid1915495/finisher pid1919211, amp 컨테이너 Up 2h). GPU v4 10.9G/0%, amp 12.0G/1%. OOM/RPC drop/에러 0.
- GATED 비율: 3e-5 = 11/29 = 38%, 5e-5 = 10/26 = 38% — exp1(33~36%) 재현, baseline 68% 대비 절반. 커리큘럼 신호복구 일관됨.
- **페어드 dL2(업뎃강도, LR 단조 재확인)**: 후반 iter도 5e-5 > 3e-5 유지. it24 s1092: 5e-5=0.0148 / 3e-5는 유사 seed서 ~0.008. 5e-5 ≈ 1.7~2x 3e-5. 둘 다 exp1 1e-5(0.0038)보다 크게 이동. dL2가 iter 진행하며 감소=수렴방향, 발산 없음.
- post-step 안정: 3e-5 kl 0.26~0.30 clip 0.89~0.92 ratio 0.91~1.02, 5e-5 kl 0.24 clip 0.87 ratio 1.04. grad_norm 높으나(1500~2200) dL2로 실제 업뎃 절제.
- before-eval(고정 5000~5019): 양쪽 grasp 11/20=0.55 동일(같은 base ckpt), official은 노이즈로 3e-5=1/20·5e-5=3/20. after/heldout은 아직 미실행(학습 진행중) → delta 판정 불가.
- 조치: 없음(정상, 3e-5=93%·5e-5=83% 진행). 개입=페어드 비교 파괴. 남은 ~2·5 iter→EVAL(after)+heldout→finisher가 ADAPTIVE_FINAL_exp2 자동집계. 다음 cron서 exp2 DONE + before/after delta로 LR별 개선 정직 판정(현재까진 개선 미확인, 완료 대기).

## 09:18 KST — exp2 DONE(개선 없음, 가설 반증) → exp3 clip-ladder 착수
- **exp2 AdamW LR-ladder 완료, 2차목표 미달(정직)**: 고정 eval grasp가 양쪽 다 flat.
  - 3e-5(v4): before 0.55 → after 0.55 → heldout 0.60 grasp / official 0.05→0.10→0.10. GATED 11/29=38%.
  - 5e-5(amp): before 0.55 → after 0.55 → heldout 0.65 grasp / official 0.15→0.10→0.20. GATED 11/29=38%.
  - **exp1 '병목=업뎃강도' 가설 반증**: adapter_dL2를 exp1(1e-5,0.0038) 대비 4~12x(3e-5·5e-5) 키웠는데도 grasp 개선 0. 업뎃강도는 병목 아님.
- **무료 진단(로그 실측, 추측 아님)으로 진짜 병목 지목**:
  - (a) **on-seed 학습 자체가 안 됨**: seed 1092를 10회 직접 학습(it8~29) n_succ=5,4,5,2,3,4,4,5,4,3 — 추세 없음. 1175(4,4,2,4,4)·1181(5,6,3,7)도 동일. → transfer 문제 아니라 learning-efficacy 문제.
  - (b) **clip 포화가 원인**: post_step clip_fraction이 양쪽 arm 전 iter 0.82~0.97(평균 ~0.90). clip=0.1 + K=5 chunk ratio곱 → PPO 목적함수의 ~90%가 clip돼 평평 → advantage gradient 억제. 이게 LR 무관(exp2 dL2 12x 범위서 개선 0)의 직접 설명.
- **조치(단일변수, 데이터근거)**: PPO clip을 완화. adaptive_sweep.sh에 clip을 4번째 위치인자로 파라미터화(default 0.1, client+trainer 동시 적용), deploy+commit(bd951e4).
  - exp3 clip-ladder 페어드 착수: optimizer/LR 고정(AdamW 3e-5), **clip만** 변수. v4 adaptive_adamw3e5_clip02(0.2, pid2008380) + amp_csi adaptive_adamw3e5_clip03(0.3, pid2259352), 공유 seed 12345. 양쪽 trainer Up, train 시작. finisher(exp3, pid2012299) detached 대기 → ADAPTIVE_FINAL_exp3_clip_ladder.md 자동집계.
- 서버상태: v4 GPU 24MiB→학습중, amp 38MiB→학습중. OOM/RPC drop/에러 0. xiaomi-server 없음(경합 없음).
