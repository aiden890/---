# Qwen3-VL 초기 관측 기반 planning 검증

## 판정

**FAIL** — direct valid plan 0/12, exact sequence 0/12, first-skill(full plan) 0/12, fallback 사용 0.

현재 단일-next-call prompt는 strict-valid call 0/12, 첫 skill 정확도 0/12였다. Full-plan 실패와 첫 단계 선택 실패를 이 두 수치로 분리했다.

## 세부 수치

- valid JSON: 0/12
- strict schema-valid full plan: 0/12
- args / instruction / contract / budget validity: 0 / 0 / 0 / 0 (각 12 기준)
- hallucinated skill / extra step / missing step / duplicate-cycle: 0 / 0 / 12 / 0
- parser repair로만 읽힌 full-plan: 0/12 (direct 성공에 포함하지 않음)

## 실패 양상과 후속 후보

24회 generation(12 current-next + 12 full-plan)의 raw text는 모두 정확히 `<cot></cot>`였고 JSON 내용은 한 번도 생성되지 않았다. 따라서 이번 실패는 올바른 계획을 parser가 놓친 경우가 아니며, JSON repair나 constrained decoding만으로 회복됐다고 볼 근거도 없다. 현재 증거와 일치하는 원인 후보는 action-policy용으로 학습된 checkpoint의 VLM이 임의 structured-planning 응답을 학습하지 않았거나, 기존 planner의 VQA preprocessing 경로에서 빈 CoT 뒤 즉시 종료하는 것이다. 후속 후보는 (1) 동일 관측·catalog으로 planner SFT, (2) production 변경 전 별도 generic Qwen3-VL chat checkpoint 대조, (3) non-empty plan token을 생성하는 것이 확인된 뒤에만 grammar-constrained decoding을 평가하는 것이다.

## 입력과 재현

세 reset seed(0, 1, 2)의 실제 RoboCasa camera 3-view를 각각 lossless horizontal contact sheet로 만들어 사용했다. simulator predicate, GT, state label은 prompt/runtime 입력에 넣지 않았다. 이미지별 경로·SHA-256은 `image_manifest.json`, 모든 raw prompt/model text/latency는 `cases.json`과 `raw/`, 표 형식은 `cases.csv`에 있다. Greedy `do_sample=false`, max_new_tokens=512이며 fallback 경로를 호출하지 않았다.

재현 명령은 `README.md`를 따른다.
