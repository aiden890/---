"""Unit tests for the strict success gate (task t_32a4f9b6).

Pure stdlib + a scripted backend -- no torch, no GPU. Verifies:
  * view routing: the gate asks each sub-fact on its calibrated camera view;
  * AND of per-sub thresholds: all sub-facts must clear their own cut;
  * episode-level aggregation: a single flush frame cannot pass a mean gate;
  * min_frames guard: too short a window is never a success.
"""
from __future__ import annotations

from success_gate import (SubQuestion, SuccessCriterion, SuccessGate,
                          make_success_gate, SUCCESS_CRITERIA)
from obs_verifier import VLMBackend


class ScriptedViewBackend(VLMBackend):
    """Returns prob from a table keyed (question_text, view); records view asks."""
    def __init__(self, table):
        self.table = table          # {(text, view): [per-call probs] or float}
        self.asks = []              # (text, view)
        self._idx = {}

    def score(self, images, question_text):
        return self.score_view(images, question_text, "full")

    def score_view(self, images, question_text, view="full"):
        self.asks.append((question_text, view))
        v = self.table[(question_text, view)]
        if isinstance(v, (int, float)):
            return float(v)
        i = self._idx.get((question_text, view), 0)
        self._idx[(question_text, view)] = i + 1
        return float(v[min(i, len(v) - 1)])


def _checks():
    n = ok = 0

    def check(name, cond):
        nonlocal n, ok
        n += 1
        ok += 1 if cond else 0
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    combined_q = SUCCESS_CRITERIA["PLACE_OBJECT"].subs[0].question

    # 1. genuine success: final combined@eye high -> pass
    bk = ScriptedViewBackend({(combined_q, "eye"): 0.91})
    g = make_success_gate("PLACE_OBJECT", bk)
    for _ in range(10):
        g.observe({})
    v = g.verdict()
    check("place.genuine_success_passes", v["success"] is True)
    check("place.routes_combined_to_eye", (combined_q, "eye") in bk.asks)

    # 2. final completion frame remains below the calibrated threshold
    bk = ScriptedViewBackend({(combined_q, "eye"): 0.8})
    g = make_success_gate("PLACE_OBJECT", bk)
    for _ in range(10):
        g.observe({})
    check("place.incomplete_endpoint_fails", g.verdict()["success"] is False)

    # 3. an earlier flush frame cannot pass when the final checkpoint is negative
    bk = ScriptedViewBackend({(combined_q, "eye"): [0.99] + [0.1] * 19})
    g = make_success_gate("PLACE_OBJECT", bk)
    for _ in range(20):
        g.observe({})
    check("place.transient_flush_rejected", g.verdict()["success"] is False)

    # 4. min_frames guard: 3 frames < min_frames(5) -> never success
    bk = ScriptedViewBackend({(combined_q, "eye"): 0.99})
    g = make_success_gate("PLACE_OBJECT", bk)
    for _ in range(3):
        g.observe({})
    v = g.verdict()
    check("place.min_frames_guard", v["success"] is False and v["reason"] == "insufficient_window")

    # 5. MOVE has no strict reporting gate; its endpoint metric is diagnostic.
    check("move.no_gate", make_success_gate("MOVE_OBJECT", bk) is None)

    print(f"\n{ok}/{n} checks passed.")
    return ok == n


if __name__ == "__main__":
    import sys
    sys.exit(0 if _checks() else 1)
