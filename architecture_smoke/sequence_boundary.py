"""Obs-only short-sequence boundary classifier.

The frozen VLM supplies one P(yes) value at each query.  This module combines a
short causal window of those scores with the robot's 14-D proprioception and an
explicit post-positive dwell.  Models are small feed-forward networks exported
as JSON by ``train_sequence_boundary.py``; inference needs only the standard
library and never sees simulator predicates.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Mapping, Optional, Sequence

STATE_DIM = 14


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if not values:
        return 0.0
    m = _mean(values)
    return (_mean([(x - m) ** 2 for x in values])) ** 0.5


def sequence_features(score_history: Sequence[float],
                      proprio_history: Sequence[Sequence[float]],
                      elapsed: int) -> list[float]:
    """Return the causal 72-D feature vector used by training and runtime."""
    scores = [float(x) for x in score_history][-8:]
    if not scores:
        raise ValueError("score_history must contain at least one value")
    features = [scores[-1], _mean(scores), min(scores), max(scores),
                scores[-1] - scores[0], _std(scores)]
    for width in (2, 3, 5):
        window = scores[-width:]
        features.extend((_mean(window), min(window), max(window)))

    states = [list(map(float, p)) for p in proprio_history if p is not None][-8:]
    if not states:
        states = [[0.0] * STATE_DIM]
    if any(len(p) != STATE_DIM for p in states):
        raise ValueError(f"proprio history must contain {STATE_DIM}-D states")
    current, oldest = states[-1], states[0]
    features.extend(current)
    features.extend(current[j] - oldest[j] for j in range(STATE_DIM))
    for j in range(STATE_DIM):
        features.append(_std([p[j] for p in states]))
    for j in range(STATE_DIM):
        axis = [p[j] for p in states]
        features.append(max(axis) - min(axis))
    features.append(float(elapsed) / 256.0)
    if len(features) != 72:
        raise AssertionError(f"sequence feature contract changed: {len(features)} != 72")
    return features


class SequenceBoundaryModel:
    """Portable MLP/logistic inference plus calibrated consecutive-positive dwell."""

    def __init__(self, spec: Mapping):
        self.spec = dict(spec)
        self.mean = [float(x) for x in spec["feature_mean"]]
        self.scale = [float(x) if float(x) != 0 else 1.0 for x in spec["feature_scale"]]
        self.layers = spec.get("layers") or []
        self.trees = spec.get("trees") or []
        if not self.layers and not self.trees:
            raise ValueError("sequence model needs dense layers or trees")
        self.tau = float(spec["tau"])
        self.dwell = max(1, int(spec["dwell"]))
        self.view = str(spec["view"])
        self.views = [str(value) for value in spec.get("views", [self.view])]
        self.aggregation = str(spec.get("aggregation", "single"))
        if len(self.mean) != 72 or len(self.scale) != 72:
            raise ValueError("sequence model feature normalization must be 72-D")
        self.score_history = deque(maxlen=8)
        self.proprio_history = deque(maxlen=8)
        self.positive_run = 0
        self.last_probability: Optional[float] = None

    @classmethod
    def from_json(cls, source):
        if isinstance(source, Mapping):
            return cls(source)
        return cls(json.loads(Path(source).read_text()))

    @staticmethod
    def _dense(values, layer):
        weights = layer["weights"]
        bias = layer["bias"]
        return [float(bias[j]) + sum(float(weights[i][j]) * values[i]
                                    for i in range(len(values)))
                for j in range(len(bias))]

    def predict_probability(self, features: Sequence[float]) -> float:
        values = [(float(x) - self.mean[i]) / self.scale[i]
                  for i, x in enumerate(features)]
        if self.trees:
            probabilities = []
            for tree in self.trees:
                node = 0
                while int(tree["left"][node]) != int(tree["right"][node]):
                    feature = int(tree["feature"][node])
                    node = (int(tree["left"][node]) if values[feature] <=
                            float(tree["threshold"][node]) else int(tree["right"][node]))
                probabilities.append(float(tree["probability"][node]))
            return _mean(probabilities)
        for index, layer in enumerate(self.layers):
            values = self._dense(values, layer)
            if index + 1 < len(self.layers):
                values = [max(0.0, x) for x in values]
        logit = max(-50.0, min(50.0, values[0]))
        return 1.0 / (1.0 + math.exp(-logit))

    def aggregate_scores(self, scores: Mapping[str, float]) -> float:
        values = [float(scores[view]) for view in self.views]
        if self.aggregation in ("single", "mean"):
            return _mean(values)
        if self.aggregation == "min":
            return min(values)
        if self.aggregation == "max":
            return max(values)
        raise ValueError(f"unknown score aggregation: {self.aggregation}")

    def update(self, vlm_probability: float,
               proprio: Optional[Sequence[float]], elapsed: int) -> tuple[bool, float]:
        self.score_history.append(float(vlm_probability))
        if proprio is not None:
            if len(proprio) != STATE_DIM:
                raise ValueError(f"proprio must be {STATE_DIM}-D, got {len(proprio)}")
            self.proprio_history.append(tuple(map(float, proprio)))
        features = sequence_features(self.score_history, self.proprio_history, elapsed)
        probability = self.predict_probability(features)
        self.last_probability = probability
        self.positive_run = self.positive_run + 1 if probability >= self.tau else 0
        return self.positive_run >= self.dwell, probability
