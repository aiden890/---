from __future__ import annotations

import threading
import time
import unittest

from priority_dispatcher import (
    DispatcherClosedError, PriorityDispatcher, SupersededError,
)


class PriorityDispatcherTest(unittest.TestCase):
    def _thread(self, target):
        outcome = {}

        def run():
            try:
                outcome["result"] = target()
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run)
        thread.start()
        return thread, outcome

    def test_policy_dequeues_before_pending_background_and_forward_is_single(self):
        dispatcher = PriorityDispatcher()
        first_started = threading.Event()
        release = threading.Event()
        lock = threading.Lock()
        order = []
        active = 0
        peak = 0

        def forward(name, block=False):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                order.append(name)
            if block:
                first_started.set()
                self.assertTrue(release.wait(1))
            with lock:
                active -= 1
            return name

        first, _ = self._thread(lambda: dispatcher.submit_policy(lambda: forward("p0", True)))
        self.assertTrue(first_started.wait(1))
        background, _ = self._thread(
            lambda: dispatcher.submit_background("boundary", lambda: forward("bg")))
        policy, _ = self._thread(lambda: dispatcher.submit_policy(lambda: forward("p1")))
        release.set()
        for thread in (first, policy, background):
            thread.join(1)
            self.assertFalse(thread.is_alive())
        dispatcher.close(timeout=1)
        self.assertEqual(order, ["p0", "p1", "bg"])
        self.assertEqual(peak, 1)
        self.assertEqual(dispatcher.snapshot()["policy_priority_dequeues"], 1)

    def test_latest_background_supersedes_pending_without_preempting_running(self):
        dispatcher = PriorityDispatcher()
        started = threading.Event()
        release = threading.Event()
        calls = []

        def running():
            calls.append("running")
            started.set()
            self.assertTrue(release.wait(1))

        running_thread, _ = self._thread(
            lambda: dispatcher.submit_background("boundary", running))
        self.assertTrue(started.wait(1))
        old_thread, old = self._thread(
            lambda: dispatcher.submit_background("endpoint", lambda: calls.append("old")))
        deadline = time.monotonic() + 1
        while not dispatcher.snapshot()["background_pending"] and time.monotonic() < deadline:
            time.sleep(0.001)
        new_thread, new = self._thread(
            lambda: dispatcher.submit_background("planner", lambda: calls.append("new")))
        old_thread.join(1)
        self.assertIsInstance(old.get("error"), SupersededError)
        release.set()
        running_thread.join(1)
        new_thread.join(1)
        dispatcher.close(timeout=1)
        self.assertNotIn("error", new)
        self.assertEqual(calls, ["running", "new"])
        self.assertEqual(dispatcher.snapshot()["superseded"], 1)

    def test_finite_policy_burst_does_not_starve_background(self):
        dispatcher = PriorityDispatcher()
        gate = threading.Event()
        entered = threading.Event()
        order = []

        def first():
            order.append("first")
            entered.set()
            self.assertTrue(gate.wait(1))

        first_thread, _ = self._thread(lambda: dispatcher.submit_policy(first))
        self.assertTrue(entered.wait(1))
        background, _ = self._thread(
            lambda: dispatcher.submit_background("boundary", lambda: order.append("background")))
        policies = [self._thread(
            lambda i=i: dispatcher.submit_policy(lambda: order.append(f"policy-{i}")))[0]
            for i in range(3)]
        gate.set()
        for thread in [first_thread, *policies, background]:
            thread.join(1)
            self.assertFalse(thread.is_alive())
        dispatcher.close(timeout=1)
        self.assertEqual(order[-1], "background")
        self.assertEqual(set(order[1:-1]), {"policy-0", "policy-1", "policy-2"})

    def test_exception_recovery_timeout_and_clean_shutdown(self):
        dispatcher = PriorityDispatcher()
        bad, failed = self._thread(
            lambda: dispatcher.submit_policy(lambda: (_ for _ in ()).throw(ValueError("boom"))))
        bad.join(1)
        self.assertIsInstance(failed.get("error"), ValueError)
        self.assertEqual(dispatcher.submit_policy(lambda: "alive"), "alive")

        gate = threading.Event()
        entered = threading.Event()
        blocker, _ = self._thread(
            lambda: dispatcher.submit_policy(
                lambda: (entered.set(), gate.wait(1))))
        self.assertTrue(entered.wait(1))
        stale, timed_out = self._thread(
            lambda: dispatcher.submit_background("endpoint", lambda: "too late", timeout_s=0.01))
        time.sleep(0.02)
        gate.set()
        blocker.join(1)
        stale.join(1)
        self.assertIsInstance(timed_out.get("error"), TimeoutError)
        metrics = dispatcher.snapshot()
        self.assertEqual(metrics["errors"], 1)
        self.assertEqual(metrics["timeouts"], 1)
        self.assertEqual(metrics["stale"], 1)
        dispatcher.close(timeout=1)
        self.assertFalse(dispatcher.is_alive)
        with self.assertRaises(DispatcherClosedError):
            dispatcher.submit_policy(lambda: None)

    def test_nonpreemptive_background_collision_is_measured(self):
        dispatcher = PriorityDispatcher()
        entered = threading.Event()
        release = threading.Event()

        def background_forward():
            entered.set()
            self.assertTrue(release.wait(1))

        background, _ = self._thread(
            lambda: dispatcher.submit_background("planner", background_forward))
        self.assertTrue(entered.wait(1))
        policy, _ = self._thread(lambda: dispatcher.submit_policy(lambda: "policy"))
        time.sleep(0.01)
        release.set()
        background.join(1)
        policy.join(1)
        metrics = dispatcher.snapshot()
        dispatcher.close(timeout=1)
        self.assertGreater(metrics["policy_blocked_by_background_ms"], 0.0)
        self.assertGreater(metrics["queue_wait_ms"]["policy"]["max"], 0.0)


if __name__ == "__main__":
    unittest.main()
