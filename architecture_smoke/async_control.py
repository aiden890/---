"""Latest-only asynchronous control-plane primitives for Skill VLA runtime.

The control loop submits observation-only requests without waiting for the remote
VLM.  A single worker permits at most one RPC in flight while a capacity-one
pending slot is atomically replaced by newer work.  Responses are advisory:
only the ExecutionManager may apply a transition.
"""
from __future__ import annotations

import enum
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Mapping, Optional


class RequestKind(str, enum.Enum):
    BOUNDARY = "boundary"
    ENDPOINT = "endpoint"
    PLANNER = "planner"


class ResponseStatus(str, enum.Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    DROPPED = "dropped"


# Runtime requests must contain robot-observable data only.  Match both the
# known fields and broad simulator/ground-truth spellings recursively.
_FORBIDDEN_EXACT = {
    "official_check_success", "sim_predicates", "ground_truth", "gt_success",
    "privileged_predicates", "task_success", "success_label", "predicates",
}
_FORBIDDEN_PARTS = ("ground_truth", "official_check", "sim_predicate", "privileged")


def assert_runtime_obs_only(value: Any, path: str = "payload") -> None:
    """Reject simulator labels/predicates anywhere in a runtime payload."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_EXACT or any(p in normalized for p in _FORBIDDEN_PARTS):
                raise ValueError(f"runtime control request contains forbidden field {path}.{key}")
            assert_runtime_obs_only(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_runtime_obs_only(child, f"{path}[{index}]")
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            normalized = item.name.lower()
            if normalized in _FORBIDDEN_EXACT or any(p in normalized for p in _FORBIDDEN_PARTS):
                raise ValueError(f"runtime control request contains forbidden field {path}.{item.name}")
            assert_runtime_obs_only(getattr(value, item.name), f"{path}.{item.name}")


@dataclass(frozen=True)
class ControlRequest:
    episode_id: str
    skill_id: str
    observation_step: int
    request_id: str
    request_kind: RequestKind
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.skill_id or not self.request_id:
            raise ValueError("episode_id, skill_id and request_id must be non-empty")
        if self.observation_step < 0:
            raise ValueError("observation_step must be non-negative")
        object.__setattr__(self, "request_kind", RequestKind(self.request_kind))
        assert_runtime_obs_only(self.payload)

    @property
    def identity(self) -> tuple[str, str, int, str, RequestKind]:
        return (self.episode_id, self.skill_id, self.observation_step,
                self.request_id, self.request_kind)


@dataclass(frozen=True)
class ControlResponse:
    episode_id: str
    skill_id: str
    observation_step: int
    request_id: str
    request_kind: RequestKind
    status: ResponseStatus = ResponseStatus.OK
    recommendation: Optional[str] = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    attempt: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_kind", RequestKind(self.request_kind))
        object.__setattr__(self, "status", ResponseStatus(self.status))

    @classmethod
    def from_request(cls, request: ControlRequest, **kwargs) -> "ControlResponse":
        return cls(request.episode_id, request.skill_id, request.observation_step,
                   request.request_id, request.request_kind, **kwargs)

    @property
    def identity(self) -> tuple[str, str, int, str, RequestKind]:
        return (self.episode_id, self.skill_id, self.observation_step,
                self.request_id, self.request_kind)


class AsyncVerifierClient:
    """Capacity-one latest-only client with exactly one background RPC worker.

    ``transport(request, timeout_s)`` must honour ``timeout_s`` and return a
    ``ControlResponse``.  A timeout is retried once with the exact same request
    identity, providing an idempotent bounded-retry contract.
    """

    def __init__(self, transport: Callable[[ControlRequest, float], ControlResponse],
                 *, timeout_s: float = 5.0, telemetry: Optional[Callable[..., None]] = None,
                 name: str = "skill-vla-control"):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._transport = transport
        self._timeout_s = float(timeout_s)
        self._telemetry = telemetry
        self._condition = threading.Condition()
        self._pending: Optional[ControlRequest] = None
        self._completed: deque[ControlResponse] = deque()
        self._closing = False
        self._drain = False
        self._in_flight: Optional[ControlRequest] = None
        self._accepted_episode: Optional[str] = None
        self._accepted_skill: Optional[str] = None
        self._accepted_step = -1
        self._worker = threading.Thread(target=self._run, name=name, daemon=False)
        self._worker.start()

    @property
    def in_flight(self) -> Optional[ControlRequest]:
        with self._condition:
            return self._in_flight

    @property
    def pending(self) -> Optional[ControlRequest]:
        with self._condition:
            return self._pending

    @property
    def is_alive(self) -> bool:
        return self._worker.is_alive()

    def set_context(self, episode_id: str, skill_id: str, accepted_step: int = -1) -> None:
        with self._condition:
            self._accepted_episode = episode_id
            self._accepted_skill = skill_id
            self._accepted_step = accepted_step
        self._emit("context", episode_id=episode_id, skill_id=skill_id,
                   accepted_step=accepted_step)

    def submit(self, request: ControlRequest) -> Optional[ControlRequest]:
        """Submit immediately; atomically replace and return an unstarted request."""
        with self._condition:
            if self._closing:
                raise RuntimeError("AsyncVerifierClient is closing")
            replaced = self._pending
            self._pending = request
            self._condition.notify()
        if replaced is not None:
            self._emit("drop", reason="superseded", request=replaced,
                       replacement_request_id=request.request_id)
        self._emit("submit", request=request, replaced=replaced is not None)
        return replaced

    def poll(self) -> list[ControlResponse]:
        """Return currently completed non-stale responses without blocking."""
        accepted: list[ControlResponse] = []
        with self._condition:
            while self._completed:
                response = self._completed.popleft()
                reason = self._stale_reason_locked(response)
                if reason is not None:
                    self._emit("drop", reason=reason, response=response)
                    continue
                if response.status is ResponseStatus.OK:
                    self._accepted_step = max(self._accepted_step, response.observation_step)
                accepted.append(response)
        return accepted

    def cancel_pending(self) -> Optional[ControlRequest]:
        with self._condition:
            request, self._pending = self._pending, None
        if request is not None:
            self._emit("cancel", reason="pending_cancelled", request=request)
        return request

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Wait until neither an RPC nor pending work remains (test/drain helper)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending is not None or self._in_flight is not None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, drain: bool = False, timeout: Optional[float] = None) -> None:
        """Stop deterministically; optionally finish the one pending request."""
        with self._condition:
            if self._closing:
                pass
            else:
                self._closing = True
                self._drain = bool(drain)
                if not drain and self._pending is not None:
                    dropped, self._pending = self._pending, None
                    self._emit("cancel", reason="shutdown", request=dropped)
                self._condition.notify_all()
        self._worker.join(timeout)
        if self._worker.is_alive():
            raise TimeoutError("AsyncVerifierClient transport did not honour its timeout contract")
        self._emit("shutdown", drained=bool(drain))

    def _stale_reason_locked(self, response: ControlResponse) -> Optional[str]:
        if self._accepted_episode is not None and response.episode_id != self._accepted_episode:
            return "stale_episode"
        if self._accepted_skill is not None and response.skill_id != self._accepted_skill:
            return "stale_skill"
        if response.observation_step < self._accepted_step:
            return "stale_step"
        return None

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                if self._closing and (not self._drain or self._pending is None):
                    return
                request, self._pending = self._pending, None
                self._in_flight = request
            assert request is not None
            response = self._invoke(request)
            with self._condition:
                self._in_flight = None
                self._completed.append(response)
                self._condition.notify_all()

    def _invoke(self, request: ControlRequest) -> ControlResponse:
        for attempt in range(2):
            started = time.monotonic()
            self._emit("rpc_start", request=request, attempt=attempt)
            try:
                response = self._transport(request, self._timeout_s)
                if not isinstance(response, ControlResponse):
                    raise TypeError("transport must return ControlResponse")
                if response.identity != request.identity:
                    raise ValueError("response identity does not match request")
                elapsed_ms = (time.monotonic() - started) * 1000.0
                self._emit("rpc_done", request=request, attempt=attempt,
                           latency_ms=elapsed_ms, status=response.status.value)
                return response
            except TimeoutError as exc:
                self._emit("timeout", request=request, attempt=attempt, error=str(exc))
                if attempt == 0:
                    self._emit("retry", request=request, attempt=1)
                    continue
                return ControlResponse.from_request(
                    request, status=ResponseStatus.TIMEOUT, error=str(exc), attempt=attempt)
            except Exception as exc:  # noqa: BLE001 -- isolate the control loop from RPC faults
                self._emit("error", request=request, attempt=attempt, error=repr(exc))
                return ControlResponse.from_request(
                    request, status=ResponseStatus.ERROR, error=repr(exc), attempt=attempt)
        raise AssertionError("unreachable")

    def _emit(self, event: str, **payload: Any) -> None:
        if self._telemetry is not None:
            self._telemetry(event, **payload)
