# 스킬별 내부 attention × 행동 민감도 결합 분석

## 먼저 읽는 법

1. 모든 화살표는 **query→key**다. 예를 들어 `action→instruction`은 16개 DiT action query가 VLM KV-cache의 instruction key에 둔 확률 질량이다.
2. heatmap의 파랑→청록→노랑은 낮음→높음이며, 한 그림의 12개 panel은 **하나의 실제 값 color scale**을 공유한다. panel 제목은 `observation state / instruction condition`이다.
3. attention은 전체 허용 key 축에 softmax한 뒤 대상 key를 합한 값이다. instruction/image 내부에서 재정규화하지 않았다. 따라서 instruction 길이가 길면 total mass가 커질 수 있다.
4. scatter의 x는 attention mass, y는 무차원 행동 ES다. **attention != causal importance/성능/정확도**이며, 두 지표가 불일치하는 것이 가능하다.
5. 개별 timestep/layer/head를 먼저 보고, 상태 평균은 마지막에 본다. 같은 trajectory의 5 timestep은 독립 표본이 아니다.

## 축과 tensor 의미 — 코드·metadata 검증

| tensor | shape | 축(0-based) | recorder에서 확인한 근거 |
|---|---|---|---|
| VLM summary | `36×32` | VLM layer 0–35 × attention head 0–31 | `text_layers` 수와 `num_attention_heads`로 할당; instruction query 평균 뒤 image key 합 |
| DiT summary | `5×36×8` | flow call 0–4 × DiT layer 0–35 × attention head 0–7 | `dit_forward` 호출 순서, `dit_layers` 수, `num_heads`로 할당 |
| action-query detail | `5×36×8×16` | flow call × layer × head × action query 0–15 | `action_length=16` metadata와 `query[:, :, -action_length:, :]`에서 직접 기록 |

분석기는 12개 실제 NPZ 각각에 대해 위 shape, float32 저장값, finite 여부를 재검증한다. shape만 보고 의미를 추측하지 않았다. VLM/DiT 모두 checkpoint Q/K projection과 RoPE로 logit을 재구성하고, mask 적용 후 full key 축에 `softmax`한 **후**의 확률이다. 정책 자체의 FlashAttention/SDPA forward는 교체하지 않았다.

## 집계와 정규화

- 조건당 attention forward는 1회이며 총 12조건(`3 observation states × 4 instructions`)이다. 조건 간 가중치는 동일하다.
- `action→instruction/image`: 먼저 instruction/image key 확률을 합하고 16 action query를 평균한다. raw DiT cell은 timestep×layer×head별 값이다.
- timestep heatmap은 8 heads 평균, head heatmap은 5 flow timesteps 평균이다. VLM heatmap은 instruction query 위치 평균이며 image key는 합산한다.
- 조건 scalar는 모든 timestep×layer×head cell의 단순 평균, state scalar는 해당 state의 4조건 scalar 단순 평균이다.
- 수치에는 min-max/z-score 정규화를 하지 않았다. 색만 각 **그림 전체의 실제 최소–최대**에 선형 매핑한다. 서로 다른 그림의 색은 직접 비교하지 않는다.

## token 경계와 의미 token

모든 index는 0-based다. image는 VLM sequence의 `<|video_pad|>` 위치이고 여러 카메라/프레임 구간이라 불연속일 수 있다. action query `0–15`는 DiT query 축이며 VLM sequence index가 아니다.

| instruction | seq len | image index 범위 | instruction index 범위 | action query | GRASP/MOVE/PLACE 위치 | special token 위치 |
|---|---:|---|---|---:|---|---|
| full task | 483 | 13–76, 85–148, 161–224, 233–296, 310–373, 382–445 | 455–466 | 0–15 | GRASP=[460]; MOVE=없음; PLACE=[461, 465] | `<|im_end|>` 471, 481; `<|im_start|>` 0, 473; `<|video_pad|>` 13–76, 85–148, 161–224, 233–296, 310–373, 382–445; `<|vision_end|>` 77, 149, 225, 297, 374, 446; `<|vision_start|>` 12, 84, 160, 232, 309, 381 |
| GRASP | 478 | 13–76, 85–148, 161–224, 233–296, 310–373, 382–445 | 455–461 | 0–15 | GRASP=[455, 460]; MOVE=없음; PLACE=없음 | `<|im_end|>` 466, 476; `<|im_start|>` 0, 468; `<|video_pad|>` 13–76, 85–148, 161–224, 233–296, 310–373, 382–445; `<|vision_end|>` 77, 149, 225, 297, 374, 446; `<|vision_start|>` 12, 84, 160, 232, 309, 381 |
| MOVE | 487 | 13–76, 85–148, 161–224, 233–296, 310–373, 382–445 | 455–470 | 0–15 | GRASP=없음; MOVE=[455, 461, 469]; PLACE=없음 | `<|im_end|>` 475, 485; `<|im_start|>` 0, 477; `<|video_pad|>` 13–76, 85–148, 161–224, 233–296, 310–373, 382–445; `<|vision_end|>` 77, 149, 225, 297, 374, 446; `<|vision_start|>` 12, 84, 160, 232, 309, 381 |
| PLACE | 494 | 13–76, 85–148, 161–224, 233–296, 310–373, 382–445 | 455–477 | 0–15 | GRASP=[461]; MOVE=[472]; PLACE=[455, 463, 468] | `<|im_end|>` 482, 492; `<|im_start|>` 0, 484; `<|video_pad|>` 13–76, 85–148, 161–224, 233–296, 310–373, 382–445; `<|vision_end|>` 77, 149, 225, 297, 374, 446; `<|vision_start|>` 12, 84, 160, 232, 309, 381 |

원문·token id·decoded text·각 index의 kind는 immutable raw `*.tokens.json`에 있다. 표의 위치는 그 mapping에서 읽어 생성했다.

## plot별 읽는 법

- `action_instruction_layer_timestep.png`: x=DiT layer, y=flow timestep. 각 cell은 8 heads 및 16 action queries 평균 뒤의 instruction mass다.
- `action_instruction_layer_head.png`: x=DiT layer, y=DiT head. 각 cell은 5 timesteps 및 16 action queries 평균이다.
- `vlm_instruction_image_layer_head.png`: x=VLM layer, y=VLM head. instruction query 평균이 앞선 image key에 둔 mass다. causal mask 때문에 반대 `image→instruction`은 0이다.
- `attention_vs_action_sensitivity.png`: state별 panel에서 x=전체 DiT 평균 instruction mass, y=행동 ES. 점 색/라벨은 instruction condition이다. 세 panel은 같은 축 범위를 쓴다.

## 행동 민감도 지표

- `ES = d_means / (noise_floor / sqrt(N))`. 여기서 `d_means`는 full-task instruction 대비 평균 16×12 action chunk의 L2 차이, noise floor는 instruction별 pairwise L2의 전역 평균, 각 state의 `N=24`다. ES는 **noise 대비 행동 변화 배수**인 무차원 값이지 attention 값이 아니다.
- cosine은 full-task instruction 평균 action chunk와의 방향 유사도다. 1에 가까울수록 방향이 거의 같다.
- reset/move/place의 `(noise floor, noise/sqrt(N))`은 각각 `(3.5582, 0.7263)`, `(1.0396, 0.2122)`, `(2.0316, 0.4147)`이다.
- attention은 routing 관찰값이고 ES/cosine은 별도 고정-observation instruction 교체 intervention이다. attention이 높아도 ES가 낮을 수 있다.

## 상태별 수치

| state | action→instruction | action→image | instruction→image | skill 행동 ES 평균 | attention↔행동 ES Pearson |
|---|---:|---:|---:|---:|---:|
| reset | 0.032515 | 0.159809 | 0.106916 | 9.154 | -0.7999899172921282 |
| move | 0.037973 | 0.153435 | 0.120544 | 1.755 | 0.9331047922171283 |
| place | 0.042043 | 0.145277 | 0.119216 | 0.500 | -0.9141819533065857 |

비정답 skill 9조건에서 total attention mass–ES Pearson은 `-0.4140238704426003`, instruction token 수로 나눈 mass–ES Pearson은 `-0.18891751775397708`이다. 상관은 기술적 교차검증이며 인과 추정치가 아니다.

## reset / move / place 핵심 요약

- reset: state 평균 action→instruction은 `0.0325`, skill ES 평균은 `9.154`; GRASP/MOVE/PLACE ES는 `10.476/10.221/6.765`다.
- move: state 평균 attention mass는 `0.0380`지만 skill ES는 `1.554–1.872`, cosine은 `0.9980–0.9986`으로 full-task 행동 방향과 거의 같다.
- place: MOVE/PLACE attention routing은 각각 `0.0454/0.0530`로 유지되지만 ES는 `0.284/0.358`, cosine은 `0.9998/0.9997`이다.
- 즉 reset→place에서 state 평균 routing mass는 `0.0325→0.0420`으로 늘지만 skill ES는 `9.154→0.500`으로 줄었다. routing의 존재를 행동 인과 중요도로 읽을 수 없다.

## 한계

- attention probability는 value vector, residual stream, MLP, AdaLN gate를 포함하지 않으므로 feature importance가 아니다.
- 5 flow timestep은 같은 noisy action trajectory 안의 반복 계산이며 독립 표본이 아니다. attention 조건당 표본 수도 1이다.
- 행동 민감도는 별도 N=24 intervention run에서 왔으므로 attention run과 표본 단위가 다르다.
- token masking/attention ablation은 policy forward를 바꾸므로 이번 read-only 측정에 포함하지 않았다. 성능·정확도·성공률은 이 산출물만으로 주장할 수 없다.

## 산출물

- `combined_summary.json`: 축/집계/정규화/token/ES 해석 guide를 포함한 machine-readable 결과
- `combined_summary.csv`: 조건별 scalar
- 3개 heatmap과 1개 attention–행동 ES scatter
- raw `*.npz`, `*.tokens.json`, `manifest.jsonl`: 변경하지 않은 원본
