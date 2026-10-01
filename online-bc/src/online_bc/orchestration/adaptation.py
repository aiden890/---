"""Conservative, recorded pilot adjustments on fixed evaluation seeds."""


def assess(reports, current_lr=1e-4):
    reports = sorted(reports, key=lambda row: row["policy_version"])
    scored = []
    for row in reports:
        outcomes = [item for item in row.get("outcomes", []) if 992001 <= item["seed"] <= 992010]
        if len(outcomes) == 10:
            scored.append((row, sum(item["cup_placed"] for item in outcomes)))
    if not scored:
        return dict(action="collect_more_evaluation", learning_rate=current_lr)
    best, best_score = max(scored, key=lambda item: (item[1], -item[0]["policy_version"]))
    latest, score = scored[-1]
    result = dict(
        action="continue",
        learning_rate=current_lr,
        best_version=best["policy_version"],
        common_seed_successes=score,
        common_seed_attempts=10,
        best_successes=best_score,
        statistically_confirmed=False,
    )
    # Two non-improving checkpoints trigger one small, logged LR ablation.
    if (
        len(scored) >= 3
        and all(s <= best_score for _, s in scored[-2:])
        and best["policy_version"] < scored[-2][0]["policy_version"]
    ):
        result.update(action="reduce_learning_rate", learning_rate=max(1e-5, current_lr / 2))
    full = [row for row in reports if row["attempts"] == 30]
    if len(full) >= 2:
        best_full = max(full, key=lambda row: (row["cup_successes"], -row["policy_version"]))
        if latest["attempts"] == 30 and best_full["cup_successes"] - latest["cup_successes"] >= 6:
            result.update(
                action="resume_best_checkpoint",
                resume_round=best_full["policy_version"],
                learning_rate=max(1e-5, current_lr / 2),
            )
    return result
