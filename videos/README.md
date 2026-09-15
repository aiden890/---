# 영상 모아보기

[전체 디렉토리 안내](../DIRECTORY_GUIDE.md) · [실험별 결과](../experiments/README.md)

원본을 복사하거나 옮기지 않고 상대 심볼릭 링크로 연결했습니다. `task/seed-N/실험__episode…mp4` 구조입니다. 성공/실패는 원래 파일명의 기록이며 이번 정리에서 정책 성능을 재평가하지 않았습니다.

| Task | 실제 seed | 실험 | 영상 | 원본 통계 |
|---|---|---|---|---|
| BreadSelection | 9 | latest-t_460aea68 | [MP4](BreadSelection/seed-9/latest-t_460aea68__episode_000_seed_9_failure.mp4) | [stats](../rollouts-xiaomi-t_460aea68/composite-unseen/bread-01/rollout/BreadSelection/stats.json) |
| CategorizeCondiments | 10 | latest-t_460aea68 | [MP4](CategorizeCondiments/seed-10/latest-t_460aea68__episode_000_seed_10_failure.mp4) | [stats](../rollouts-xiaomi-t_460aea68/composite-unseen/condiments-01/rollout/CategorizeCondiments/stats.json) |
| CloseBlenderLid | 7 | first-cu121 | [MP4](CloseBlenderLid/seed-7/first-cu121__episode_000_seed_7_failure.mp4) | [stats](../xiaomi-cu121/output/rollout/CloseBlenderLid/stats.json) |
| CloseBlenderLid | 9 (skill instruction 3종) | t_4a072806 | [grasp](CloseBlenderLid/skill_instruction_eval/grasp.mp4) · [move_holding](CloseBlenderLid/skill_instruction_eval/move_holding.mp4) · [place](CloseBlenderLid/skill_instruction_eval/place.mp4) | [REPORT](CloseBlenderLid/skill_instruction_eval/REPORT.md) |
| HeatKebabSandwich | 14 | previous-xiaomi-robotics-1 | [MP4](HeatKebabSandwich/seed-14/previous-xiaomi-robotics-1__episode_000_seed_14_failure.mp4) | [stats](../rollouts/xiaomi-robotics-1/composite-unseen/kebab-01/rollout/HeatKebabSandwich/stats.json) |
| HeatKebabSandwich | 15 | previous-xiaomi-robotics-1 | [MP4](HeatKebabSandwich/seed-15/previous-xiaomi-robotics-1__episode_000_seed_15_failure.mp4) | [stats](../rollouts/xiaomi-robotics-1/composite-unseen/kebab-02/rollout/HeatKebabSandwich/stats.json) |
| KettleBoiling | 9 | previous-xiaomi-robotics-1 | [MP4](KettleBoiling/seed-9/previous-xiaomi-robotics-1__episode_000_seed_9_success.mp4) | [stats](../rollouts/xiaomi-robotics-1/composite-seen/kettle-01/rollout/KettleBoiling/stats.json) |
| PackIdenticalLunches | 11 | latest-t_460aea68 | [MP4](PackIdenticalLunches/seed-11/latest-t_460aea68__episode_000_seed_11_failure.mp4) | [stats](../rollouts-xiaomi-t_460aea68/composite-seen/lunches-01/rollout/PackIdenticalLunches/stats.json) |
| PrepareCoffee | 13 | previous-xiaomi-robotics-1 | [MP4](PrepareCoffee/seed-13/previous-xiaomi-robotics-1__episode_000_seed_13_failure.mp4) | [stats](../rollouts/xiaomi-robotics-1/composite-seen/coffee-01/rollout/PrepareCoffee/stats.json) |
| SteamInMicrowave | 19 | latest-t_460aea68 | [MP4](SteamInMicrowave/seed-19/latest-t_460aea68__episode_000_seed_19_failure.mp4) | [stats](../rollouts-xiaomi-t_460aea68/composite-seen/steam-01/rollout/SteamInMicrowave/stats.json) |
| TurnOnSinkFaucet | 24 | previous-xiaomi-robotics-1 | [MP4](TurnOnSinkFaucet/seed-24/previous-xiaomi-robotics-1__episode_000_seed_24_success.mp4) | [stats](../rollouts/xiaomi-robotics-1/atomic-seen/faucet-02/rollout/TurnOnSinkFaucet/stats.json) |
| WaffleReheat | 20 | previous-xiaomi-robotics-1 | [MP4](WaffleReheat/seed-20/previous-xiaomi-robotics-1__episode_000_seed_20_failure.mp4) | [stats](../rollouts/xiaomi-robotics-1/composite-unseen/waffle-01/rollout/WaffleReheat/stats.json) |
| WashLettuce | 22 | latest-t_460aea68 | [MP4](WashLettuce/seed-22/latest-t_460aea68__episode_000_seed_22_failure.mp4) | [stats](../rollouts-xiaomi-t_460aea68/composite-seen/lettuce-01/rollout/WashLettuce/stats.json) |

이 목록은 정리 시점의 로컬 영상 스냅샷입니다. 이후 worker가 만든 영상은 자동 추가되지 않습니다.
