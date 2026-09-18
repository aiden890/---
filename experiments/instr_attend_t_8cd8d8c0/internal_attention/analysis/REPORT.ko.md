# 스킬별 내부 attention × 행동 민감도 결합 분석

## 검증 범위

- 실제 Xiaomi-Robotics-1-RoboCasa365 checkpoint forward의 Q/K projection과 RoPE를 그대로 사용했다.
- 정책 실행은 원래 FlashAttention/SDPA 경로를 유지했고, hook은 선택 query/key 확률만 read-only로 재구성했다.
- 12개 조건(reset/move/place × correct/grasp/move/place), VLM 36층×32헤드 및 DiT 5 flow timestep×36층×8헤드를 모두 기록했다.
- RoboCasa365 processor의 실제 action chunk는 16 query다. 원 tensor summary shape: VLM `(36, 32)`, DiT `(5, 36, 8)`, action-query 상세 `(5, 36, 8, 16)`; 저장 dtype float32.

## 상태별 요약

| state | action→instruction | action→image | instruction→image | skill 행동 ES 평균 | attention↔행동 ES Pearson |
|---|---:|---:|---:|---:|---:|
| reset | 0.032515 | 0.159809 | 0.106916 | 9.154 | -0.7999899172921282 |
| move | 0.037973 | 0.153435 | 0.120544 | 1.755 | 0.9331047922171283 |
| place | 0.042043 | 0.145277 | 0.119216 | 0.500 | -0.9141819533065857 |

비정답 skill 9개 조건 전체에서 total attention mass와 기존 action effect size 간 Pearson은 `-0.4140238704426003`, instruction token 수로 나눈 mass와 ES 간 Pearson은 `-0.18891751775397708`이다. 이 값들은 attention이 행동 민감도와 함께 움직이는지 보는 기술적 교차검증일 뿐 인과 추정치가 아니다.

## 해석

1. `action→instruction`은 DiT action query가 VLM KV cache의 instruction span에 둔 정규화 attention mass다. `action→image`와 동일 분모에서 측정했다.
2. `instruction→image`는 causal VLM에서 instruction query가 앞선 visual token에 둔 attention이다. 반대 방향 `image→instruction`은 image token이 instruction보다 먼저 오므로 causal mask상 정확히 0이다.
3. GRASP/MOVE/PLACE 열은 각 instruction 안에서 명시적으로 발견된 관련 lexeme token에 대한 mass다. 해당 lexeme가 없는 지시는 0으로 기록하며, token index는 각 `*.tokens.json`에 보존했다.
4. **불일치가 핵심이다.** state 평균 action→instruction mass는 reset `0.0325`에서 place `0.0420`으로 증가하지만 skill 행동 ES 평균은 reset `9.154`에서 place `0.500`으로 급감한다. place에서 move/place 지시는 attention mass `0.0454/0.0530`을 받으면서도 행동 ES는 `0.284/0.358`, cosine은 `0.9998/0.9997`이다. 즉 내부 routing이 존재해도 출력 행동은 observation에 의해 거의 고정될 수 있다.
5. reset에서도 place 지시가 네 지시 중 가장 높은 total mass(`0.0421`)를 받지만 skill 행동 ES(`6.765`)는 grasp/move(`10.476/10.221`)보다 낮다. total mass의 instruction 길이 의존성 때문에 크기 순위를 causal importance 순위로 읽을 수 없다.
6. 높은 attention weight는 routing의 관찰값이지 causal importance가 아니다. 기존 고정-observation counterfactual action ES/cosine과 일치·불일치를 함께 봐야 한다.
7. token masking은 checkpoint forward를 바꾸므로 이번 read-only 측정에는 포함하지 않았다. 기존 instruction 교체 실험이 행동 수준의 독립적 intervention 역할을 한다.

## 산출물

- `combined_summary.json`, `combined_summary.csv`: machine-readable 결합 결과
- `action_instruction_layer_timestep.png`: timestep×layer heatmap
- `action_instruction_layer_head.png`: head×layer heatmap
- `vlm_instruction_image_layer_head.png`: VLM instruction→image heatmap
- `attention_vs_action_sensitivity.png`: attention–행동 ES 결합 scatter
- raw `*.npz`, `*.tokens.json`, `manifest.jsonl`: 조건별 tensor와 token mapping

## 한계

- attention probability는 value vector, residual stream, MLP, AdaLN gate의 영향을 포함하지 않으므로 단독으로 feature importance를 뜻하지 않는다.
- flow timestep별 attention은 같은 noisy action trajectory 안의 관찰이며 독립 표본이 아니다.
- 기존 행동 민감도는 instruction 교체 intervention이고 내부 attention run은 각 조건 1회이므로, 샘플링 노이즈까지 포함한 인과효과 크기로 읽으면 안 된다.
