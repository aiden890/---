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

    clear_q = SUCCESS_CRITERIA["PLACE_OBJECT"].subs[0].question
    seated_q = SUCCESS_CRITERIA["PLACE_OBJECT"].subs[1].question

    # 1. genuine success: clear@right high, seated@right ok -> pass
    bk = ScriptedViewBackend({(clear_q, "right"): 0.9, (seated_q, "right"): 0.7})
    g = make_success_gate("PLACE_OBJECT", bk)
    for _ in range(10):
        g.observe({})
    v = g.verdict()
    check("place.genuine_success_passes", v["success"] is True)
    check("place.routes_clear_to_right", (clear_q, "right") in bk.asks)
    check("place.routes_seated_to_right", (seated_q, "right") in bk.asks)

    # 2. cleared but NOT seated -> guard fails the AND
    bk = ScriptedViewBackend({(clear_q, "right"): 0.9, (seated_q, "right"): 0.2})
    g = make_success_gate("PLACE_OBJECT", bk)
    for _ in range(10):
        g.observe({})
    check("place.cleared_not_seated_fails", g.verdict()["success"] is False)

    # 3. single flush frame cannot lift the episode mean over threshold
    #    clear@right = one 0.99 then all 0.1 -> mean ~0.18 < 0.75
    bk = ScriptedViewBackend({(clear_q, "right"): [0.99] + [0.1] * 19,
                              (seated_q, "right"): 0.7})
    g = make_success_gate("PLACE_OBJECT", bk)
    for _ in range(20):
        g.observe({})
    check("place.single_flush_frame_rejected", g.verdict()["success"] is False)

    # 4. min_frames guard: 3 frames < min_frames(5) -> never success
    bk = ScriptedViewBackend({(clear_q, "right"): 0.99, (seated_q, "right"): 0.9})
    g = make_success_gate("PLACE_OBJECT", bk)
    for _ in range(3):
        g.observe({})
    v = g.verdict()
    check("place.min_frames_guard", v["success"] is False and v["reason"] == "insufficient_window")

    # 5. MOVE has no gate
    check("move.no_gate", make_success_gate("MOVE_OBJECT", bk) is None)

    print(f"\n{ok}/{n} checks passed.")
    return ok == n


if __name__ == "__main__":
    import sys
    sys.exit(0 if _checks() else 1)
