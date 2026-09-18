# Generic Qwen planner canonical contract 12-case gate

## 판정

**PASS** — direct semantic JSON 12/12, exact 3-skill sequence 12/12, model-emitted name/args validity 12/12, deterministic canonicalization 12/12, canonicalized strict 5-field plan 12/12, 첫 skill 12/12.

Hard gate는 direct semantic JSON, exact sequence, canonicalized strict plan이 각각 12/12일 때만 PASS다. 하나라도 실패하면 production integration에 NOT READY다.

## 계약 경계

모델은 `name`과 `args`만 결정했다. Local registry는 모델이 낸 순서, skill 이름, args를 바꾸지 않고 검증한 뒤에만 `instruction`, `contract`, `budget`을 결정론적으로 채웠다. 알 수 없는 skill, 누락/추가/중복 step, argument key/value 불일치는 즉시 canonicalization 실패이며 정답 sequence로 보정하지 않는다. Parser repair, constrained decoding, fallback은 사용하지 않았고 production runtime도 변경하지 않았다.

## 실행

모델 `Qwen/Qwen3-VL-4B-Instruct` revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`을 공식 processor/chat template과 greedy decoding으로 사용했다. 세 보존 reset contact sheet × 의미가 같은 네 instruction, 총 12 case다. latency min/mean/max는 3208.4/3297.0/4132.4 ms, 관측 peak allocated VRAM은 8673.0 MiB다. Raw model text는 `raw/`, canonicalized output은 `canonical/`, token/latency/validation 세부 정보는 `cases.json`에 분리 보존했다.

생성 source commit은 `61e517f233c1eee91e9154130d4fd0ef172743e0`다. 종료 후 amp_csi에서 실험 container 0, port 10087 listener 0, GPU compute process 0, RTX3090 38 MiB / 0%를 확인했다.
