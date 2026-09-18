# Pre-episode full-plan scheduler hard gate

## 판정

**PRE_EPISODE_READY** — throwaway scheduler/contract gate의 모든 조건을 통과했다. 이는 production 배포 승인이 아니라, production 변경 없이 측정한 후보 scheduler 계약의 준비 판정이다.

## Production source 확인

`xiaomi-cu121/rollout.py`는 실제 모델 output의 action 축을 보존한 뒤 첫 12 action dimension만 사용한다(lines 184-186). Empty deque에서 동기 policy call을 하고 `--replan-steps`만큼 잘라 채운 뒤 한 action씩 `env.step`한다(lines 237-262); 기본값은 16(lines 329-331)이다. Loop sleep, target control Hz, policy/action deadline은 없다. `--video-fps=20`은 video encoder에만 전달되므로 control rate 근거가 아니다(lines 275-280, 340-342). Client/server도 동기 request/forward이며 deadline gate가 없다. 정확한 hash와 line evidence는 `source_contract.json`에 보존했다.

실제 model output은 본 실행에서 정확히 16 action/chunk로 확인됐다. Throwaway 계약은 20 action/s, 50 ms period, 16-action chunk, 800 ms chunk-generation deadline, action schedule jitter 25 ms, source-observation age 1100 ms, queue 상한 1 chunk로 명시했다.

## Episode timeline과 scheduler 결과

Reset snapshot 이후 simulator/action loop가 paused인 상태에서 generic planner를 정확히 1회 호출했다. Planner latency 3230.301 ms와 첫 action까지 episode-start latency 3550.975 ms를 action-loop 통계와 분리해 보고했다. Direct semantic JSON은 `GRASP_OBJECT → MOVE_OBJECT → PLACE_OBJECT`였고 strict canonical validation을 통과했다. Planning 중 action은 0개였으며 first action은 planning 완료 후에만 발생했다.

Deterministic executor는 canonical 3-skill start boundary를 소비했고 runtime planner call은 0이었다. 총 48 action(각 skill 16)을 20 Hz로 소비하면서 action-buffer underflow 0, stale action 0, deadline miss 0, fallback 0, producer error 0, 최대 queue depth 1 chunk였다. Skill-specific policy generation latency p50/p95/max는 275.433/276.880/277.041 ms로 800 ms chunk deadline 이내였다. 전체 event/action 시계열은 `timeline.csv`, chunk latency는 `policy_latency.csv`에 보존했다.

동일한 parent full-task input의 dual-loaded 12-forward 재측정은 p95 270.022 ms, mean 254.518 ms, throughput 3.929 Hz였다. Parent p95 254.668 ms 대비 +6.03%, throughput 3.946 Hz 대비 -0.42%로 각 10% 허용 범위 이내다.

## Fail-closed와 정리

Malformed JSON, unknown skill, wrong sequence, extra field, stale reset observation, reset divergence 6개 case는 모두 action 0, fallback 0, automatic correction 0으로 종료했다. 상세 오류는 `failure_modes.json`과 `report.json`에 있다.

종료 후 amp_csi에서 experiment container 0, ports 10086/10087 listener 0, compute process 0, GPU 38 MiB/0%를 확인했다. Production runtime과 기존 `integration_readiness/` 내용은 수정하지 않았다.
