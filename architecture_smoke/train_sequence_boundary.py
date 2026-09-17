"""Train and held-out-evaluate portable obs-only sequence boundary models.

Combines a post-success-hold cache (positives have observable frames after the
sim-label transition) with the balanced corpus negatives.  Splits are stratified
by whole rollout before fitting: 9 train / 3 calibration / 3 held-out episodes
per class when 15+/15- are available.  Threshold, view, model family and dwell
are selected on calibration only; the held-out split is read once for reporting.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sequence_boundary import sequence_features

SKILL_KEY = {"grasp": "grasp", "move_holding": "move", "place": "place_combined"}
CALL_NAME = {"grasp": "GRASP_OBJECT", "move_holding": "MOVE_OBJECT", "place": "PLACE_OBJECT"}


def split_rollouts(rows):
    result = {"train": [], "calibration": [], "held_out": []}
    for label in (False, True):
        group = sorted((r for r in rows if bool(r["gt_success"]) == label),
                       key=lambda r: (int(r["seed"]), r["rollout"]))
        random.Random(51035471 + int(label)).shuffle(group)
        if len(group) < 5:
            raise ValueError(f"need at least 5 rollouts for class {label}, got {len(group)}")
        # A task-id-seeded stratified shuffle is fixed before fitting; every fifth
        # item is held out and the preceding item calibrates.
        for index, rollout in enumerate(group):
            bucket = ("held_out" if index % 5 == 4 else
                      "calibration" if index % 5 == 3 else "train")
            result[bucket].append(rollout)
    return result


def observable_recall_ceiling(rows):
    """Upper bound when no scored observation exists at/after the GT boundary."""
    positives = [r for r in rows if r["gt_success"]]
    observable = sum(any(int(f["env_step"]) >= int(r["gt_success_step"])
                         for f in r["frames"]) for r in positives)
    return {"positive_rollouts": len(positives),
            "with_post_boundary_observation": observable,
            "recall_ceiling": observable / len(positives) if positives else 0.0}


def rollout_features(rollout, score_keys, aggregation="single"):
    score_history, proprio_history, output = [], [], []
    last_proprio = None
    for frame in rollout["frames"]:
        values = [float(frame["scores"][key]) for key in score_keys]
        if aggregation in ("single", "mean"):
            score = sum(values) / len(values)
        elif aggregation == "min":
            score = min(values)
        elif aggregation == "max":
            score = max(values)
        else:
            raise ValueError(aggregation)
        score_history.append(score)
        proprio = frame.get("proprio")
        if proprio is not None:
            last_proprio = proprio
            proprio_history.append(proprio)
        elif last_proprio is not None:
            proprio_history.append(last_proprio)
        output.append(sequence_features(score_history, proprio_history,
                                        int(frame["env_step"])))
    return np.asarray(output, dtype=np.float64)


def frame_labels(rollout):
    success_step = rollout.get("gt_success_step")
    return np.asarray([success_step is not None and
                       int(frame["env_step"]) >= int(success_step)
                       for frame in rollout["frames"]], dtype=np.int64)


def fit_model(rows, score_keys, aggregation, family):
    feature_parts, label_parts = [], []
    for rollout in rows:
        features = rollout_features(rollout, score_keys, aggregation)
        labels = frame_labels(rollout)
        success_step = rollout.get("gt_success_step")
        # Post-positive supervision is intentionally local: teach the earliest
        # stable boundary, not later policy drift while the collector holds for
        # diagnostics.  Pre-boundary and failed-episode frames remain negatives.
        keep = np.asarray([success_step is None or int(frame["env_step"]) <= int(success_step) + 16
                           for frame in rollout["frames"]], dtype=bool)
        feature_parts.append(features[keep])
        label_parts.append(labels[keep])
    xs = np.concatenate(feature_parts)
    ys = np.concatenate(label_parts)
    positives = np.flatnonzero(ys)
    negatives = np.flatnonzero(ys == 0)
    if not len(positives):
        raise ValueError("training split has no post-success frames")
    rng = np.random.default_rng(51035471)
    negatives = rng.choice(negatives, min(len(negatives), max(100, 5 * len(positives))),
                           replace=False)
    selected = np.concatenate([positives, negatives])
    rng.shuffle(selected)
    xs, ys = xs[selected], ys[selected]
    scaler = StandardScaler().fit(xs)
    z = scaler.transform(xs)
    if family == "logistic":
        model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000,
                                   random_state=51035471)
    elif family == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", alpha=0.01,
                              max_iter=2000, early_stopping=False,
                              random_state=51035471)
    elif family == "extra":
        model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3,
                                     max_features=0.7, class_weight="balanced",
                                     random_state=51035471, n_jobs=-1)
    elif family == "forest":
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                       max_features=0.7, class_weight="balanced",
                                       random_state=51035471, n_jobs=-1)
    else:
        raise ValueError(f"unknown family {family}")
    model.fit(z, ys)
    return scaler, model, {"n_frames": int(len(ys)), "n_positive_frames": int(ys.sum())}


def predict_rollout(rollout, score_keys, aggregation, scaler, model):
    features = rollout_features(rollout, score_keys, aggregation)
    return model.predict_proba(scaler.transform(features))[:, 1]


def event_metrics(rows, probabilities, tau, dwell):
    records = []
    for rollout in rows:
        run = 0
        advance = None
        for frame, probability in zip(rollout["frames"], probabilities[rollout["rollout"]]):
            run = run + 1 if float(probability) >= tau else 0
            if run >= dwell:
                advance = int(frame["env_step"])
                break
        records.append({"rollout": rollout["rollout"], "seed": rollout["seed"],
                        "gt_success": bool(rollout["gt_success"]),
                        "gt_success_step": rollout.get("gt_success_step"),
                        "advance_step": advance})
    tp = fp = fn = tn = early_fp = 0
    offsets = []
    for record in records:
        gt, step, advance = (record["gt_success"], record["gt_success_step"],
                             record["advance_step"])
        if gt and advance is not None:
            offset = advance - int(step)
            if offset < 0:
                fp += 1; fn += 1; early_fp += 1
            else:
                tp += 1; offsets.append(offset)
        elif gt:
            fn += 1
        elif advance is not None:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(records), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "early_fp": early_fp, "precision": precision, "recall": recall,
            "mean_timing_offset": (sum(offsets) / len(offsets) if offsets else None),
            "timing_offsets": offsets, "records": records}


def portable_spec(skill, views, aggregation, family, scaler, model, tau, dwell, train_info):
    if family == "logistic":
        layers = [{"weights": model.coef_.T.tolist(), "bias": model.intercept_.tolist()}]
    elif family == "mlp":
        layers = [{"weights": weights.tolist(), "bias": bias.tolist()}
                  for weights, bias in zip(model.coefs_, model.intercepts_)]
    else:
        layers = None
    spec = {"schema_version": 1, "skill": CALL_NAME[skill], "view": views[0],
            "views": list(views), "aggregation": aggregation,
            "family": family, "feature_dim": 72,
            "feature_mean": scaler.mean_.tolist(), "feature_scale": scaler.scale_.tolist(),
            "tau": float(tau), "dwell": int(dwell),
            "vlm_min_interval": 4, "train": train_info}
    if layers is not None:
        spec["layers"] = layers
    else:
        spec["trees"] = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            leaves = []
            for value in tree.value:
                counts = value[0]
                leaves.append(float(counts[1] / counts.sum()) if counts.sum() else 0.0)
            spec["trees"].append({"feature": tree.feature.tolist(),
                                  "threshold": tree.threshold.tolist(),
                                  "left": tree.children_left.tolist(),
                                  "right": tree.children_right.tolist(),
                                  "probability": leaves})
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--posthold-cache", required=True, action="append",
                        help="repeat for per-skill cache files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = json.loads(Path(args.base_cache).read_text())
    post_rollouts = []
    for cache_path in args.posthold_cache:
        post_rollouts.extend(json.loads(Path(cache_path).read_text())["rollouts"])
    output = {"meta": {"base_cache": args.base_cache, "posthold_cache": args.posthold_cache,
                       "split": "whole-rollout stratified RNG(51035471), fixed 18/6/6"},
              "skills": {}, "runtime_operating_points": {}}
    for skill in SKILL_KEY:
        skill_positives = [r for r in post_rollouts
                           if r["skill"] == skill and r["gt_success"]]
        positives_by_seed = {}
        for rollout in skill_positives:
            positives_by_seed.setdefault(int(rollout["seed"]), rollout)
        positives = list(positives_by_seed.values())
        negatives = [r for r in base["rollouts"] if r["skill"] == skill and not r["gt_success"]]
        base_skill = [r for r in base["rollouts"] if r["skill"] == skill]
        rows = positives[:15] + negatives[:15]
        split = split_rollouts(rows)
        candidates = []
        score_sources = [((view,), "single") for view in ("left", "right", "eye")]
        score_sources += [(('left', 'right', 'eye'), aggregation)
                          for aggregation in ("mean", "min", "max")]
        for views, aggregation in score_sources:
            score_keys = [f"{SKILL_KEY[skill]}@{view}" for view in views]
            if any(key not in f["scores"] for r in rows for f in r["frames"]
                   for key in score_keys):
                continue
            for family in ("logistic", "mlp", "extra", "forest"):
                scaler, model, train_info = fit_model(
                    split["train"], score_keys, aggregation, family)
                calibration_probs = {r["rollout"]: predict_rollout(
                    r, score_keys, aggregation, scaler, model)
                                     for r in split["calibration"]}
                for dwell in (1, 2, 3, 4):
                    for tau in np.linspace(0.05, 0.99, 95):
                        metrics = event_metrics(split["calibration"], calibration_probs,
                                                float(tau), dwell)
                        if (metrics["early_fp"] == 0 and metrics["precision"] >= 0.9 and
                                metrics["recall"] >= 0.5):
                            candidates.append((metrics["recall"], metrics["precision"],
                                               float(tau), dwell, views, aggregation,
                                               family, metrics, scaler,
                                               model, train_info, score_keys))
        if not candidates:
            raise RuntimeError(f"{skill}: no calibration operating point meets acceptance")
        chosen = max(candidates, key=lambda item: (
            item[0], item[1], -len(item[4]), item[2], item[3]))
        _, _, tau, dwell, views, aggregation, family, calibration_metrics, scaler, model, train_info, score_keys = chosen
        held_probs = {r["rollout"]: predict_rollout(
            r, score_keys, aggregation, scaler, model)
                      for r in split["held_out"]}
        held_metrics = event_metrics(split["held_out"], held_probs, tau, dwell)
        spec = portable_spec(skill, views, aggregation, family, scaler, model,
                             tau, dwell, train_info)
        output["skills"][skill] = {
            "counts": {name: {"n": len(group),
                                "positive": sum(bool(r["gt_success"]) for r in group)}
                       for name, group in split.items()},
            "censored_base_control": observable_recall_ceiling(base_skill),
            "selected": {"views": list(views), "aggregation": aggregation,
                         "family": family, "tau": tau, "dwell": dwell},
            "calibration": calibration_metrics, "held_out": held_metrics,
            "model": spec,
        }
        output["runtime_operating_points"][CALL_NAME[skill]] = {
            "view": views[0], "vlm_min_interval": 4, "sequence_model": spec}
        print(skill, "calibration", calibration_metrics["precision"],
              calibration_metrics["recall"], "held_out", held_metrics["precision"],
              held_metrics["recall"], "early", held_metrics["early_fp"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
