"""Bounded, disjoint collection plans and success-only learning gates."""

from online_bc.data.data_control import read_controls


def batch_plan(config, round_index, attempted):
    remaining = max(0, config.get("max_attempts_per_round", 32) - attempted)
    offset = attempted
    # Explicit extra windows use a disjoint namespace instead of spilling
    # across the native 100-seed round stride. Original assignments stay intact.
    for window in config.get("additional_windows", {}).get(str(round_index), []):
        start, count = window["attempt_start"], window["attempts"]
        if not 1 <= count <= 96 or start < 96:
            raise ValueError("Invalid additional collection window")
        if start <= attempted < start + count:
            remaining = start + count - attempted
            offset = window["seed_start"] - (993000 + round_index * 100 + 1) + attempted - start
            break
    requested = (
        config.get("first_round_attempts", 32)
        if round_index == 1
        else config.get("attempts_per_batch", 8)
    )
    total = min(requested, remaining)
    nodes = list(config["workers"])
    plan = []
    for index, node in enumerate(nodes):
        count = total // len(nodes) + (index < total % len(nodes))
        if count:
            plan.append(dict(node=node, episodes=count, seed_offset=offset))
            offset += count
    return plan


def eligible_successes(sources, controls_file=None):
    excluded = read_controls(controls_file)["excluded"]
    return len({eid for source in sources for eid in source["accepted"] if eid not in excluded})


def ready_to_train(config, round_index, attempted, successes):
    if round_index == 1:
        return attempted >= config.get("first_round_attempts", 32) and successes > 0
    return successes >= config.get("target_new_successes", 8)
