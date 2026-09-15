# t_15f668c5: CloseBlenderLid read-only verification

## Scope disposition
The user's latest scope, recorded in this card and root t_460aea68, cancels the original multi-task planning-failure candidate reruns. Upstream t_f22ea8b3 already performed exactly five additional CloseBlenderLid episodes (seeds 8-12). This card performed no new rollouts, no remote writes, and no changes to existing experiment results. Original candidate reruns are CANCELLED BY USER, not completed and not a finding of no candidates.

## Independently verified results

| Seed | Role | Success (stats) | Steps | Termination | Fully decoded frames |
|---|---|---|---|---|---|
| 7 | Existing baseline | false | 900 | horizon | 451 |
| 8 | Additional | false | 900 | horizon | 451 |
| 9 | Additional | true | 885 | success | 444 |
| 10 | Additional | false | 900 | horizon | 451 |
| 11 | Additional | false | 900 | horizon | 451 |
| 12 | Additional | false | 900 | horizon | 451 |

Additional: 1 success / 5 episodes. Including baseline: 1 success / 6 episodes. Horizon is 900 in all six stats files. Seed 9 is a success counterexample to any claim that this task always fails; it does not establish a planning mechanism.

## Verification executed
Command (from /home/aiden/Desktop/lab/robot):

    /home/aiden/miniconda3/bin/python robocasa-docker/research/t_15f668c5/verify.py

Exit code 0. All 20 files in the upstream SHA256SUMS manifest matched. All six MP4s decoded fully using cv2, at 20 fps and 256x768x3; frame counts matched 1 + ceil(steps / 2). Additional five videos also matched the preserved remote decode report's hash, size, and frame count. commands.jsonl contains exactly seeds 8-12, all rc=0. Stats confirm actual episode seeds and outcomes. Machine-readable absolute source paths, hashes, sizes, and episode results are in verification.json.

Source locations:
- Additional episodes: /home/aiden/Desktop/lab/robot/robocasa-docker/rollouts/xiaomi-robotics-1-t_f22ea8b3/remote/closeblenderlid
- Baseline: /home/aiden/Desktop/lab/robot/robocasa-docker/xiaomi-cu121/output/rollout/CloseBlenderLid
- Upstream environment/commands report: /home/aiden/Desktop/lab/robot/robocasa-docker/rollouts/xiaomi-robotics-1-t_f22ea8b3/REPORT.md

## Limits and downstream handoff
This is local read-only verification against preserved upstream remote manifests, not a fresh SSH inspection of remote bytes or GPU state. Environment provenance remains in the upstream report (same official checkpoint/CUDA12.1/shared server); no new environment was launched here.

Integrity and final success stats do not distinguish planning, manipulation, perception, or causal time-limit failures. No per-step failure classification or model-internal reasoning claim is made. Other task results were not analysed, per the user's scope cancellation.

Pre-created downstream card t_de8b0c1e should finish only the rescoped CloseBlenderLid final index/delivery and any authorised resource cleanup, preserving original results. Do not resume cancelled multi-task runs or manufacture a planning-candidate clip. No additional seeds remain required under the current scope.
