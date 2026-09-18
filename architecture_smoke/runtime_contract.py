"""Canonical production timing and queue contract for skill-conditioned runtime."""
from __future__ import annotations

ACTION_RATE_HZ = 20.0
POLICY_CHUNK_ACTIONS = 16
CHUNK_DEADLINE_MS = 800.0
ACTION_JITTER_DEADLINE_MS = 25.0
MAX_OBSERVATION_AGE_MS = 1100.0
MAX_QUEUE_CHUNKS = 1


def as_dict() -> dict:
    return {
        "action_rate_hz": ACTION_RATE_HZ,
        "policy_chunk_actions": POLICY_CHUNK_ACTIONS,
        "chunk_deadline_ms": CHUNK_DEADLINE_MS,
        "action_jitter_deadline_ms": ACTION_JITTER_DEADLINE_MS,
        "max_observation_age_ms": MAX_OBSERVATION_AGE_MS,
        "max_queue_chunks": MAX_QUEUE_CHUNKS,
    }
