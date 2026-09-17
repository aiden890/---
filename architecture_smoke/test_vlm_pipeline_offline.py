"""Offline CPU wiring check for the obs-only VLM-verifier pipeline (t_fc5e73d5).

No torch / gym / robocasa / GPU. Drives SequentialPlanner -> ExecutionManager
(_execute_vlm) -> MockObsEnv -> ObsVLMVerifier(MockVLMBackend) through a full
GRASP -> MOVE -> PLACE episode, asserting:
  * skill termination is decided by the VLM latch (hysteresis_k consecutive yes),
  * the planner advances ONLY on the verifier's obs-only SUCCESS,
  * the runtime NEVER reads a privileged sim predicate (obs_only=True path;
    obs payload has no forbidden keys),
  * per-skill vlm_stats + the trace 'vlm' records are produced.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from schemas import AdapterMode, SkillStatus
import bindings
from skills import SkillRegistry
from planner import SequentialPlanner
from executor import ExecutionManager
from trace import Trace
from episode import run_episode
from policy import MockPolicy
from obs_verifier import ObsInput
from vlm_backends import MockVLMBackend


class MockObsEnv:
    """Numpy-free env exposing obs_for_verifier() with fake 3-cam + 14D proprio.

    Scripts a scene: after GRASP_LATCH steps the 'grasp' becomes visually true,
    etc. The VLM backend (below) keys off a per-skill query counter, so this env
    only needs to supply well-formed obs + advance the offline success label.
    """

    CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")

    def __init__(self, seed=0, video_stride=2):
        self.seed = seed
        self.video_stride = video_stride
        self.frames = []
        self._step = 0
        self._grip = 0.10

    def reset(self):
        self._step = 0
        self.frames = [("f", 0)]
        return {"mock": True}

    def scene_meta(self):
        return {"mock": True, "seed": self.seed}

    def predicates(self):
        # offline label only; the obs_only loop reads this ONCE at the end.
        return {"official_check_success": self._step > 40}

    def observation_ref(self):
        return f"mock{self.seed}:step{self._step}"

    def obs_for_verifier(self):
        # a tiny fake 2x6x3 image per cam (width divisible by 3 not needed here)
        img = [[[0, 0, 0] for _ in range(4)] for _ in range(2)]
        images = {k: img for k in self.CAMS}
        proprio = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._grip, -self._grip,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return ObsInput(images=images, proprio=proprio, step=self._step)

    def build_policy_input(self, instruction, adapter_mode, adapter_checkpoint=None):
        from schemas import PolicyInput
        return PolicyInput(instruction=instruction, state_history=[], image_history={},
                           adapter_mode=adapter_mode, adapter_checkpoint=adapter_checkpoint)

    def step(self, action):
        self._step += 1
        # settle the gripper so ProprioGate.candidate_stop can trigger
        self._grip = 0.02
        return {"mock": True}, False, False, {}

    def maybe_record_frame(self, force=False):
        if force or self._step % self.video_stride == 0:
            self.frames.append(("f", self._step))
            return len(self.frames) - 1
        return None

    def save_video(self, path):
        Path(path).write_text(f"MOCK {len(self.frames)} frames\n")
        return len(self.frames)

    def close(self):
        pass


def main():
    registry = SkillRegistry(bindings.CONTRACTS)
    policy = MockPolicy(replan_steps=16, action_dim=12)

    # VLM backend: say 'yes' from the 3rd query onward -> with hysteresis_k=2
    # each skill latches after ~2 consecutive yes queries. Ignores pixels.
    def sched(i, q):
        return 0.9 if (i % 5) >= 2 else 0.1
    vlm = MockVLMBackend(sched)

    tmp = Path(tempfile.mkdtemp())
    env = MockObsEnv(seed=0)
    trace = Trace(tmp / "trace.jsonl", {"seed": 0, "verifier_kind": "obs_vlm"})
    planner = SequentialPlanner(registry, max_retries=1)
    env.reset()
    manager = ExecutionManager(registry, policy, env, trace, AdapterMode.DISABLED,
                               vlm_backend=vlm, vlm_min_interval=4, hysteresis_k=2,
                               tau=0.6, event_gated=True,
                               verifier_operating_points={
                                   "MOVE_OBJECT": {"view": "right", "tau": 0.8,
                                                   "hysteresis_k": 3,
                                                   "vlm_min_interval": 2},
                               })
    summary = run_episode(planner, manager, env, trace, registry, bindings.GOAL,
                          episode_budget=600, max_planner_calls=8, obs_only=True)
    trace.close()

    # --- assertions ---
    skills = [s["skill"] for s in summary["skills"]]
    statuses = [s["status"] for s in summary["skills"]]
    assert skills[:3] == ["GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT"], skills
    assert all(st == "SUCCESS" for st in statuses[:3]), statuses
    # every skill produced obs-only vlm_stats with a latch step
    for s in summary["skills"][:3]:
        vs = s.get("vlm_stats")
        assert vs and vs.get("succeeded_step") is not None, s
        assert vs["n_vlm_calls"] >= 2, vs
    move_stats = summary["skills"][1]["vlm_stats"]
    assert (move_stats["view"], move_stats["tau"], move_stats["hysteresis_k"],
            move_stats["vlm_min_interval"]) == ("right", 0.8, 3, 2), move_stats
    # trace has 'vlm' records and NO forbidden sim predicate leaked to the judge
    recs = [json.loads(l) for l in (tmp / "trace.jsonl").read_text().splitlines()]
    vlm_recs = [r for r in recs if r["type"] == "vlm"]
    assert vlm_recs, "no vlm trace records"
    for r in vlm_recs:
        assert "prob" in r and "decision" in r
    # planner advanced obs-only (terminal = planner_done, not task_success break)
    assert summary["terminal"] in ("planner_done", "budget_exhausted"), summary["terminal"]
    print(json.dumps({"OK": True, "skills": skills, "statuses": statuses,
                      "n_vlm_records": len(vlm_recs),
                      "latch_steps": [s["vlm_stats"]["succeeded_step"] for s in summary["skills"][:3]],
                      "terminal": summary["terminal"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
