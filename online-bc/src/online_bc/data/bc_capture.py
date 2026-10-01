"""Lossless pre-action observations and executed actions for online BC."""

import json
import numpy as np
from online_bc.rollout.coffee_metrics import score


class Capture:
    def __init__(self, out):
        self.root = out / "bc"
        self.root.mkdir(exist_ok=True)
        self.starts = []

    def observation(self, step, obs, images, states, rollout):
        data = {
            k: np.array(v, copy=True) for k, v in obs.items() if k.startswith(("state.", "video."))
        }
        data["xiaomi/state_history"] = rollout.sample_history(states, 4, 2)
        for key, q in images.items():
            data["xiaomi/" + key] = rollout.sample_history(q, 4, 2)
        np.savez_compressed(self.root / f"obs-{step:06d}.npz", **data)
        self.starts.append(step)

    def finalize(self, actions, trace, model, seed, prompt):
        result = score(trace)
        segments = []
        if result["cup_placed"]:
            segments.append(dict(skill="cup_placement", start=0, end=result["cup_placement_step"]))
        if result["button_after_placement"]:
            segments.append(
                dict(
                    skill="button_press",
                    start=result["cup_placement_step"],
                    end=min(len(actions), result["button_press_step"] + 32),
                )
            )
        samples = []
        for seg in segments:
            for start in self.starts:
                if not seg["start"] <= start < seg["end"]:
                    continue
                count = min(50, seg["end"] - start)
                # Keep the validity mask: padded actions must never become labels.
                target = np.zeros((50, 12), np.float32)
                target[:count] = actions[start : start + count]
                mask = np.arange(50) < count
                np.savez_compressed(
                    self.root / f"actions-{start:06d}.npz", actions=target, valid=mask
                )
                samples.append(
                    dict(
                        observation=f"obs-{start:06d}.npz",
                        actions=f"actions-{start:06d}.npz",
                        skill=seg["skill"],
                        step=start,
                    )
                )
        manifest = dict(
            schema_version=1,
            label_source="successful_executed_policy_actions",
            model=model,
            seed=seed,
            prompt=prompt,
            action_space="robocasa365_native_12d_before_convert_action",
            metrics=result,
            segments=segments,
            samples=samples,
        )
        temp = self.root / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(self.root / "manifest.json")
