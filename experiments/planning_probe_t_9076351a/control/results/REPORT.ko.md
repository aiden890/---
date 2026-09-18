# Qwen planner 빈 출력 원인 분리 control

## 최종 판정

**checkpoint limitation** — The official instruct control produced direct non-empty parseable JSON for all three observations, while the Xiaomi action-policy checkpoint emitted only an empty CoT followed by EOS for every general-QA and planning prompt.

## 관측

- Xiaomi `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` revision `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4`: 일반 image-QA 3/3과 planning 1/1이 모두 정확히 `<cot></cot>` 뒤 EOS로 종료했다. 첫 생성 token/top-k, 전체 token IDs, EOS IDs와 latency는 `xiaomi_checkpoint_sanity.json`에 있다.
- 공식 `Qwen/Qwen3-VL-4B-Instruct` revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`: 동일한 세 reset contact sheet와 동일한 full-plan payload에서 semantic non-empty 3/3, direct parseable JSON 3/3이었다. 따라서 image 입력, chat template, generation 자체가 전부 빈 출력을 만드는 harness 결함이라는 가설은 지지되지 않는다.
- 공식 모델은 세 case 모두 `GRASP_OBJECT → MOVE_OBJECT → PLACE_OBJECT` 순서를 생성했지만 strict registry plan은 0/3이다. 일부 argument/contract 문자열이 registry의 정확한 값과 달라 strict score는 실패했으며, 이 control의 원인 분리 기준은 direct raw language/JSON 생성 가능 여부다.

## 해석 범위

이 결과는 Xiaomi action-policy checkpoint가 일반 text/VQA planner로 부적합하다는 증거다. Qwen3-VL 계열 전체의 planning 능력 부재를 뜻하지 않으며, 공식 instruct control도 exact skill contract 준수까지 입증하지 않는다. parser repair, constrained decoding, fallback, production runtime 변경은 사용하지 않았다.

## 재현성과 정리

각 case의 image/prompt hash, raw output, generated token IDs, first-token top-k, EOS/stop reason, latency는 JSON/CSV/raw에 보존했다. source commit은 `1d8d20d3f0871567df6e961601d1da3e0547b8d8`, 부모 artifact commit은 `aa0807dc80aa0958a39ec315cb6bee9ea889c568`이다. amp_csi 종료 gate에서 experiment container 없음, port 10087 listener 없음, compute process 없음, GPU 38 MiB/0%를 확인했다.
