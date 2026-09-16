"""Structured JSONL trace + video-frame metadata. No control decisions.

One trace per episode. Records every planner boundary and every skill window so
the architecture handoffs (typed call -> render -> base VLA -> execute -> verify
-> result -> planner) are auditable from the file alone.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")
        self._write("episode_start", **meta)

    def _write(self, kind: str, **payload):
        rec = {"t": round(time.time(), 3), "type": kind}
        rec.update(payload)
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    # ---- planner boundary ------------------------------------------------- #
    def plan(self, planner_calls, obs_ref, predicates, skill_call, instruction,
             rationale, planner_kind):
        self._write(
            "plan", planner_calls=planner_calls, planner_kind=planner_kind,
            observation_ref=obs_ref, selected_skill=(skill_call["name"] if skill_call else None),
            skill_args=(skill_call["args"] if skill_call else None),
            rendered_instruction=instruction, rationale=rationale,
            predicates_digest=_digest(predicates),
        )

    def route(self, skill, adapter_mode, adapter_checkpoint, max_steps, can_start):
        self._write("route", skill=skill, adapter_mode=adapter_mode,
                    adapter_checkpoint=adapter_checkpoint, max_steps=max_steps,
                    can_start=can_start)

    def window(self, skill, step, chunk_len, executed, adapter_mode, decision,
               hold, predicates_digest):
        self._write("window", skill=skill, step=step, chunk_len=chunk_len,
                    executed=executed, adapter_mode=adapter_mode, decision=decision,
                    hold=hold, predicates_digest=predicates_digest)

    def frame(self, skill, step, frame_index):
        self._write("frame", skill=skill, step=step, frame_index=frame_index)

    def verify(self, skill, decision, reason, elapsed, hold):
        self._write("verify", skill=skill, decision=decision, reason=reason,
                    elapsed=elapsed, hold=hold)

    def vlm(self, skill, step, frame_index, prob, consecutive_yes, queried,
            decision, proprio_candidate, proprio_gripper_closed):
        """One obs-only VLM verifier judgement at a step (offline-auditable overlay).

        Carries ONLY obs-derived diagnostics (VLM P(yes) + proprio-gate flags) and
        the recorded frame index; NO privileged simulator predicate.
        """
        self._write("vlm", skill=skill, step=step, frame_index=frame_index,
                    prob=prob, consecutive_yes=consecutive_yes, queried=queried,
                    decision=decision, proprio_candidate=proprio_candidate,
                    proprio_gripper_closed=proprio_gripper_closed)

    def skill_result(self, result: dict, next_skill):
        self._write("skill_result", next_skill=next_skill, **result)

    def episode_end(self, **payload):
        self._write("episode_end", **payload)

    def close(self):
        self._fh.close()


_DIGEST_KEYS = (
    "lid_grasped", "lid_lifted", "in_preplace_region", "lid_on_blender",
    "lid_upright_7deg", "gripper_lid_far_0.15", "official_check_success",
    "lid_xy_to_closed_pos", "lid_dz_to_closed_pos", "eef_lid_dist",
)


def _digest(p: dict) -> dict:
    return {k: p.get(k) for k in _DIGEST_KEYS if k in p}
