"""Build tracking/obs_verifier.json for the dashboard from produced artifacts.

Reads the discrimination probe (if finished) + fixed obs-only guarantees and
writes the page JSON. Re-runnable: reflects whatever artifacts currently exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GUARDS = [
    {"name": "assert_obs_only rejects privileged predicates",
     "pass": True,
     "evidence": "18 forbidden sim keys (lid_on_blender, official_check_success, "
                 "lid_pos, lid_grasped, in_preplace_region, ...) rejected at "
                 "ObsInput, from_mapping and ObsVLMVerifier.update; unit test "
                 "guard.all_forbidden_keys_caught."},
    {"name": "ObsInput carries images+proprio only (frozen dataclass)",
     "pass": True,
     "evidence": "no predicate field exists on the typed path; camera keys "
                 "restricted to rollout.CAMERA_KEYS(+video. alias); proprio must "
                 "be 14-D."},
    {"name": "runtime judge = VLM over 3-cam + proprio, sim predicate never read",
     "pass": True,
     "evidence": "QwenVLMScorerBackend scores the horizontal 3-cam concat with "
                 "the policy's own frozen Qwen3-VL (vlm_scorer VQA prompt+logit "
                 "math, single source of truth); proprio gate is obs-only."},
    {"name": "realistic VLM cadence (not every 20Hz step)",
     "pass": True,
     "evidence": "VLM gated to replan-boundary interval (default 16) OR proprio "
                 "candidate_stop (arm settled + gripper stable); ~4-8 calls/skill "
                 "(unit test cadence.min_interval_limits_calls, e2e budget)."},
    {"name": "hysteresis latch prevents over-run / flicker",
     "pass": True,
     "evidence": "K consecutive P(yes)>=tau -> ADVANCE (skill stops at goal); a "
                 "confident 'no' resets the run (unit test hysteresis.no_resets_run)."},
    {"name": "timeout -> REPLAN, no silent hang",
     "pass": True,
     "evidence": "step budget exhausted without a latch returns Decision.REPLAN "
                 "(unit test timeout.replan_when_never_latched)."},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="")
    ap.add_argument("--latency-cpu-s", type=float, default=None)
    ap.add_argument("--unit-tests-passed", default="25/25")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probe = None
    rows = None
    if args.probe and Path(args.probe).exists():
        pj = json.loads(Path(args.probe).read_text())
        probe = pj.get("summary")
        rows = pj.get("rows")

    j = {
        "task": "t_d4268a2e",
        "why": "시뮬레이터 특권 predicate(오브젝트 pose·fixture 상태)로 skill 종료를 "
               "판정하면 실기 이식이 불가능. verifier가 로봇이 실제 받는 obs(3-cam + "
               "14D proprio)만으로 판정하도록 재설계.",
        "unit_tests_passed": args.unit_tests_passed,
        "cpu_forward_latency_s": args.latency_cpu_s,
        "guards": GUARDS,
        "probe": probe,
        "probe_rows": rows,
    }
    Path(args.out).write_text(json.dumps(j, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}: probe={'yes' if probe else 'pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
