# Xiaomi-Robotics-1 RoboCasa365 rollout exploration

Task t_5af7225b. Actual official Xiaomi policy, parent CUDA12.1 stack, split=target.

Results: 5 tasks, 6 completed episodes, 2 successes, 4 horizon failures. One additional construction failure preserved.

## Results and taxonomy

| Category | Task | Attempt | Actual seed | Steps | Success | End | Taxonomy | MP4 |
|---|---|---|---:|---:|---|---|---|---|
| atomic-seen | TurnOnSinkFaucet | faucet-01 | None | 0 | None | environment/interface error before episode reset | environment/interface | None: construction failed |
| atomic-seen | TurnOnSinkFaucet | faucet-02 | 24 | 185 | True | success | - | atomic-seen/faucet-02/rollout/TurnOnSinkFaucet/episode_000_seed_24_success.mp4 |
| composite-seen | PrepareCoffee | coffee-01 | 13 | 1800 | False | horizon | ambiguous | composite-seen/coffee-01/rollout/PrepareCoffee/episode_000_seed_13_failure.mp4 |
| composite-seen | KettleBoiling | kettle-01 | 9 | 799 | True | success | - | composite-seen/kettle-01/rollout/KettleBoiling/episode_000_seed_9_success.mp4 |
| composite-unseen | HeatKebabSandwich | kebab-01 | 14 | 2700 | False | horizon | planning | composite-unseen/kebab-01/rollout/HeatKebabSandwich/episode_000_seed_14_failure.mp4 |
| composite-unseen | HeatKebabSandwich | kebab-02 | 15 | 2700 | False | horizon | control/execution | composite-unseen/kebab-02/rollout/HeatKebabSandwich/episode_000_seed_15_failure.mp4 |
| composite-unseen | WaffleReheat | waffle-01 | 20 | 4050 | False | horizon | ambiguous | composite-unseen/waffle-01/rollout/WaffleReheat/episode_000_seed_20_failure.mp4 |

## Planning-failure candidates

HeatKebabSandwich seed14 (kebab-01): ONE planning-failure candidate, not reproduced. Required baguette is left on plate while the robot proceeds to door/timer. Seed15 fails differently and attempts both foods. No reproduced planning failure is claimed.

The diagnosis is external behavior analysis, not access to internal model reasoning. Perception/grounding remains a competing explanation for the omitted object.

## Per-attempt evidence

### faucet-01: TurnOnSinkFaucet
Instruction: None
Diagnosis: Upstream MJCFObject writes temporary XML next to object assets (objects.py:244-265). Initial read-only parent asset mount rejected this write. Child-scoped writable clone fixed exact task without changing policy. Error log and original runner preserved.
Config: atomic-seen/faucet-01/config.json
Full log: atomic-seen/faucet-01/rollout.log

### faucet-02: TurnOnSinkFaucet
Instruction: Turn on the sink faucet.
Diagnosis: Environment-reported success; atomic positive control. No failure diagnosis.
Selection: Atomic single-fixture actuation baseline to contrast with multi-subgoal failures.
Expected subgoals: Turn faucet on.
Contact sheet: atomic-seen/faucet-02/rollout/TurnOnSinkFaucet/episode_000_seed_24_success.contact.jpg
Config: atomic-seen/faucet-02/config.json
Full log: atomic-seen/faucet-02/rollout.log

### coffee-01: PrepareCoffee
Instruction: Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.
Diagnosis: Sampled frames show mug pickup, transport toward dispenser (roughly steps 148-444), then prolonged interaction around machine/button (steps 518-1800). No clear omitted subgoal. Exact failed success predicate is not logged; pressing/placement control failure is plausible, but perception/grounding cannot be excluded.
Selection: Composite-Seen transfer followed by appliance activation; tests prerequisite completion before pressing start.
Expected subgoals: Pick mug from initially open cabinet. -> Place mug under dispenser. -> Press start button and release.
Contact sheet: composite-seen/coffee-01/rollout/PrepareCoffee/episode_000_seed_13_failure.contact.jpg
Inspected failure window (approx steps): [444, 1800]
Config: composite-seen/coffee-01/config.json
Full log: composite-seen/coffee-01/rollout.log

### kettle-01: KettleBoiling
Instruction: Pick the kettle from the counter and place it on a stove burner. Then turn the burner on.
Diagnosis: Environment-reported success; composite positive comparison.
Selection: Composite-Seen object placement plus matching burner activation; positive comparison against failed composites.
Expected subgoals: Move kettle from counter onto burner. -> Turn that burner on and release.
Contact sheet: composite-seen/kettle-01/rollout/KettleBoiling/episode_000_seed_9_success.contact.jpg
Config: composite-seen/kettle-01/config.json
Full log: composite-seen/kettle-01/rollout.log

### kebab-01: HeatKebabSandwich
Instruction: Pick up the kebab skewer and baguette bread, and place them inside the toaster oven. Close the toaster oven door and start by setting the timer.
Diagnosis: Kebab is carried toward the oven first; baguette remains visibly on the blue plate. By sampled steps 336-448 the door is closed and the arm subsequently attends to door/timer region through step 784. Baguette remains on plate at late sampled steps 1792-2700. Observable omission: proceeds to later appliance subgoals without transporting the second required food. This is an external behavioral classification, not proof of an internal planning defect; missed-object grounding remains an alternative.
Selection: Composite-Unseen two-object prerequisite followed by door and timer operations; omission/order-sensitive task.
Expected subgoals: Place kebab AND baguette on toaster-oven rack (either pickup order). -> Close door. -> Set timer to at least 0.1.
Contact sheet: composite-unseen/kebab-01/rollout/HeatKebabSandwich/episode_000_seed_14_failure.contact.jpg
Inspected failure window (approx steps): [336, 784]
Config: composite-unseen/kebab-01/config.json
Full log: composite-unseen/kebab-01/rollout.log

### kebab-02: HeatKebabSandwich
Instruction: Pick up the kebab skewer and baguette bread, and place them inside the toaster oven. Close the toaster oven door and start by setting the timer.
Diagnosis: Unlike seed14, seed15 picks up baguette first (steps112-336) and later kebab (steps896-1008). Baguette is visibly placed on/near the open door rather than securely on a rack (steps448-672), followed by prolonged placement/door interactions. The two-food omission is not reproduced. Exact contact predicates are not logged; grounding ambiguity remains.
Selection: Composite-Unseen two-object prerequisite followed by door and timer operations; omission/order-sensitive task.
Expected subgoals: Place kebab AND baguette on toaster-oven rack (either pickup order). -> Close door. -> Set timer to at least 0.1.
Contact sheet: composite-unseen/kebab-02/rollout/HeatKebabSandwich/episode_000_seed_15_failure.contact.jpg
Inspected failure window (approx steps): [448, 2700]
Config: composite-unseen/kebab-02/config.json
Full log: composite-unseen/kebab-02/rollout.log

### waffle-01: WaffleReheat
Instruction: Open the microwave, place the bowl with waffle inside the microwave, then close the microwave door and turn it on.
Diagnosis: Sampled steps168-336 show open-door approach; steps504-840 show bowl transfer and door manipulation; steps1008 onward show prolonged microwave-front/button-region interaction. Broad intended order appears present. Final failure may involve placement, closure or activation; no predicate trace separates these. Do not claim a planning failure.
Selection: Composite-Unseen nested-object transport plus door and power sequence.
Expected subgoals: Open microwave. -> Place bowl containing waffle inside. -> Close door. -> Turn microwave on.
Contact sheet: composite-unseen/waffle-01/rollout/WaffleReheat/episode_000_seed_20_failure.contact.jpg
Inspected failure window (approx steps): [1008, 4050]
Config: composite-unseen/waffle-01/config.json
Full log: composite-unseen/waffle-01/rollout.log

## Version and validation

```json
{
  "model": "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
  "checkpoint_revision": "3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4",
  "code_commit": "0dd7aef8dc87296246aae812a1f59ccb708e5546",
  "robocasa_commit": "4f8a2980def75a55dff96b990745b83540425f09",
  "inference_image": "xiaomi-cu121:t_9f03a613",
  "inference_image_id": "sha256:aac5e8e597636dc80df55a3150da68f2fb5897e28bfb02836319578fba477c87",
  "client_image": "xiaomi-client:t_9f03a613",
  "client_image_id": "sha256:c52e01c1048c46fe0fef0ed535fad3b5067f97b3d5fc397e92a8e96f9702cd19",
  "cuda": "12.1.1",
  "torch": "2.5.1+cu121",
  "driver": "535.183.01",
  "gpu": "RTX 3090 24 GB",
  "asset_volume": "robocasa-assets-t_5af7225b"
}
```

All six MP4s fully decoded on v4 and copied byte-identically to this directory (SHA256, sizes and frames in video-verification.json). Local file hash verification is performed by build_summary.py. Native local decoding was not run; local package installation was blocked, so playback integrity is established by the full remote decode plus byte identity.

Simulator source classification: vendor/robocasa/robocasa/utils/dataset_registry.py TARGET_TASKS lines2802-2859; atomic_seen/composite_seen/composite_unseen use official target categories. Task definitions and success predicates are preserved under sources/.

The initial read-only asset failure is not a failed policy episode. Parent assets were not modified; child volume is writable because official MJCF object construction creates then deletes temporary XML files.

## Reproduce on v4

Existing verified images, checkpoint and child asset volume must be retained. The stopped server container is retained; restart it rather than running the start command twice. No host port is exposed.

```bash
cd /home/v4/rollouts-xiaomi-t_5af7225b
docker start xiaomi-server-t_5af7225b
docker logs --tail 20 xiaomi-server-t_5af7225b
# Wait until Model loaded / Server running before clients.
bash run-t_5af7225b.sh episode atomic-seen TurnOnSinkFaucet 7 faucet-02-rerun-01
bash run-t_5af7225b.sh episode composite-seen PrepareCoffee 7 coffee-01-rerun-01
bash run-t_5af7225b.sh episode composite-seen KettleBoiling 7 kettle-01-rerun-01
bash run-t_5af7225b.sh episode composite-unseen HeatKebabSandwich 7 kebab-01-rerun-01
bash run-t_5af7225b.sh episode composite-unseen HeatKebabSandwich 8 kebab-02-rerun-01
bash run-t_5af7225b.sh episode composite-unseen WaffleReheat 7 waffle-01-rerun-01
bash run-t_5af7225b.sh stop
```

Choose a fresh attempt ID every run; the runner rejects an existing directory. Upstream actual episode seed = base seed + task index within the category (num-trials=1). The manifest records actual seeds, not just CLI base seeds.

Local integrity check:
```bash
python3 build_summary.py
```

## Limitations

- Exploratory task selection, not benchmark accuracy.
- Environment seeds fixed; upstream server RNG is not explicitly seeded, so action trajectories are not guaranteed bitwise reproducible.
- Failure taxonomy is based on external sampled video behavior and source success predicates, not hidden reasoning; no dense subgoal-state trace was captured.
- Video stride2 and 20fps are upstream defaults; video seconds are playback time, not wall-clock runtime.
- Inherited upstream observation-space and processor-fps warnings retained in logs.
