"""Two-lane, single-forward scheduler for the shared Xiaomi/Qwen3-VL model.

Policy work is FIFO and always selected before the one latest-only background
slot.  Running work is deliberately non-preemptive; all model forwards execute
on this dispatcher's single worker thread.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class SupersededError(RuntimeError):
    """Raised to a caller whose unstarted background request was replaced."""


class DispatcherClosedError(RuntimeError):
    """Raised when work is submitted to, or cancelled by, a closed dispatcher."""


@dataclass
class _Job:
    lane: str
    request_kind: str
    function: Callable[[], Any]
    submitted_at: float
    deadline: Optional[float] = None
    blocked_by_background: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None

    def finish(self, *, result: Any = None, error: Optional[BaseException] = None) -> None:
        self.result = result
        self.error = error
        self.done.set()


class PriorityDispatcher:
    """Exactly two logical queues and exactly one non-preemptive worker."""

    POLICY = "policy"
    BACKGROUND = "background"
    BACKGROUND_KINDS = frozenset(("boundary", "endpoint", "planner"))

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 telemetry: Optional[Callable[..., None]] = None):
        self._clock = clock
        self._telemetry = telemetry
        self._condition = threading.Condition()
        self._policy = deque()
        self._background: Optional[_Job] = None
        self._active: Optional[_Job] = None
        self._closing = False
        self._drain = False
        self._metrics = {
            "submitted": {self.POLICY: 0, self.BACKGROUND: 0},
            "completed": {self.POLICY: 0, self.BACKGROUND: 0},
            "errors": 0, "dropped": 0, "superseded": 0, "stale": 0, "timeouts": 0,
            "active_forwards": 0, "max_active_forwards": 0,
            "queue_wait_ms": {}, "model_latency_ms": {},
            "background_by_kind": {kind: 0 for kind in sorted(self.BACKGROUND_KINDS)},
            "policy_blocked_by_background_ms": 0.0,
        }
        self._worker = threading.Thread(
            target=self._run, name="shared-gpu-priority-dispatcher", daemon=False)
        self._worker.start()

    @property
    def is_alive(self) -> bool:
        return self._worker.is_alive()

    def submit_policy(self, function: Callable[[], Any]) -> Any:
        return self._submit(_Job(self.POLICY, self.POLICY, function, self._clock()))

    def submit_background(self, request_kind: str, function: Callable[[], Any],
                          *, timeout_s: Optional[float] = None) -> Any:
        if request_kind not in self.BACKGROUND_KINDS:
            raise ValueError(f"invalid background request_kind: {request_kind!r}")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        now = self._clock()
        deadline = None if timeout_s is None else now + timeout_s
        return self._submit(_Job(self.BACKGROUND, request_kind, function, now, deadline))

    def _submit(self, job: _Job) -> Any:
        replaced = None
        with self._condition:
            if self._closing:
                raise DispatcherClosedError("priority dispatcher is closing")
            self._metrics["submitted"][job.lane] += 1
            if job.lane == self.POLICY:
                job.blocked_by_background = bool(
                    self._active is not None and self._active.lane == self.BACKGROUND)
                self._policy.append(job)
            else:
                self._metrics["background_by_kind"][job.request_kind] += 1
                replaced, self._background = self._background, job
                if replaced is not None:
                    self._metrics["dropped"] += 1
                    self._metrics["superseded"] += 1
                    replaced.finish(error=SupersededError("background request superseded before start"))
            self._condition.notify()
        if replaced is not None:
            self._emit("superseded", request_kind=replaced.request_kind)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def snapshot(self) -> dict:
        with self._condition:
            metrics = {
                **self._metrics,
                "submitted": dict(self._metrics["submitted"]),
                "completed": dict(self._metrics["completed"]),
                "queue_wait_ms": dict(self._metrics["queue_wait_ms"]),
                "model_latency_ms": dict(self._metrics["model_latency_ms"]),
                "background_by_kind": dict(self._metrics["background_by_kind"]),
                "policy_queue_depth": len(self._policy),
                "background_pending": self._background is not None,
                "active_lane": self._active.lane if self._active else None,
                "ready": self._worker.is_alive() and not self._closing,
            }
        return metrics

    def close(self, *, drain: bool = False, timeout: Optional[float] = None) -> None:
        cancelled = []
        with self._condition:
            if not self._closing:
                self._closing = True
                self._drain = bool(drain)
                if not drain:
                    cancelled.extend(self._policy)
                    self._policy.clear()
                    if self._background is not None:
                        cancelled.append(self._background)
                        self._background = None
                self._condition.notify_all()
        for job in cancelled:
            job.finish(error=DispatcherClosedError("request cancelled by dispatcher shutdown"))
        if cancelled:
            with self._condition:
                self._metrics["dropped"] += len(cancelled)
        self._worker.join(timeout)
        if self._worker.is_alive():
            raise TimeoutError("active model forward did not finish during shutdown")
        self._emit("shutdown", drained=bool(drain))

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._policy and self._background is None and not self._closing:
                    self._condition.wait()
                if self._closing and (not self._drain or
                                      (not self._policy and self._background is None)):
                    return
                if self._policy:
                    job = self._policy.popleft()
                else:
                    job, self._background = self._background, None
                self._active = job
            assert job is not None
            started = self._clock()
            queue_wait_ms = (started - job.submitted_at) * 1000.0
            self._record_latency("queue_wait_ms", job.request_kind, queue_wait_ms)
            if job.lane == self.POLICY and job.blocked_by_background:
                with self._condition:
                    self._metrics["policy_blocked_by_background_ms"] += queue_wait_ms
            if job.deadline is not None and started >= job.deadline:
                with self._condition:
                    self._metrics["stale"] += 1
                    self._metrics["timeouts"] += 1
                    self._active = None
                    self._condition.notify_all()
                job.finish(error=TimeoutError("background request expired before model forward"))
                self._emit("stale", request_kind=job.request_kind, queue_wait_ms=queue_wait_ms)
                continue
            self._emit("start", lane=job.lane, request_kind=job.request_kind,
                       queue_wait_ms=queue_wait_ms)
            try:
                with self._condition:
                    self._metrics["active_forwards"] += 1
                    self._metrics["max_active_forwards"] = max(
                        self._metrics["max_active_forwards"],
                        self._metrics["active_forwards"])
                result = job.function()
            except BaseException as exc:  # keep the dispatcher alive after model faults
                elapsed_ms = (self._clock() - started) * 1000.0
                self._record_latency("model_latency_ms", job.request_kind, elapsed_ms)
                with self._condition:
                    self._metrics["errors"] += 1
                job.finish(error=exc)
                self._emit("error", lane=job.lane, request_kind=job.request_kind,
                           model_latency_ms=elapsed_ms, error=repr(exc))
            else:
                elapsed_ms = (self._clock() - started) * 1000.0
                self._record_latency("model_latency_ms", job.request_kind, elapsed_ms)
                with self._condition:
                    self._metrics["completed"][job.lane] += 1
                job.finish(result=result)
                self._emit("done", lane=job.lane, request_kind=job.request_kind,
                           model_latency_ms=elapsed_ms)
            finally:
                with self._condition:
                    self._metrics["active_forwards"] -= 1
                    self._active = None
                    self._condition.notify_all()

    def _record_latency(self, metric: str, kind: str, value: float) -> None:
        with self._condition:
            bucket = self._metrics[metric].setdefault(kind, {"count": 0, "total": 0.0, "max": 0.0})
            bucket["count"] += 1
            bucket["total"] += value
            bucket["max"] = max(bucket["max"], value)

    def _emit(self, event: str, **payload: Any) -> None:
        if self._telemetry is not None:
            self._telemetry(event, **payload)
