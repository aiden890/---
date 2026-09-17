from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from adaptive_curriculum import AdaptiveCurriculum, CurriculumTransaction  # noqa: E402
from update_batch import (  # noqa: E402
    BatchAccumulator,
    IdempotentUpdateCache,
    audit_batch_trace,
    execute_batched_update,
    recv_exact,
    retry_idempotent_rpc,
)


class raises:
    def __init__(self, exc_type, match):
        self.exc_type = exc_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc_type.__name__}: {self.match}")
        if not issubclass(exc_type, self.exc_type):
            return False
        assert self.match in str(exc), (self.match, str(exc))
        return True


class FakeClient:
    def __init__(self, groups, *, stale=False, fail_group=None):
        self.groups = copy.deepcopy(groups)
        self.store = {}
        self.stale = stale
        self.fail_group = fail_group
        self.update_calls = 0
        self.reset_calls = 0

    def metrics(self):
        store = dict(self.store)
        if self.stale:
            store["stale"] = 1
        return {
            "n_chunks": sum(store.values()),
            "chunks_by_traj": store,
        }

    def collect(self, group_index, env_seed, iteration_token):
        if group_index == self.fail_group:
            raise RuntimeError("injected collection failure")
        spec = self.groups[group_index]
        advantages = {}
        for member, (advantage, chunks) in enumerate(spec):
            trajectory_id = f"iter{iteration_token}_m{member}"
            advantages[trajectory_id] = advantage
            self.store[trajectory_id] = chunks
        return {
            "_advantages": advantages,
            "returns": [float(x[0]) for x in spec],
            "n_success_group": sum(float(x[0]) > 0 for x in spec),
        }

    def update(self, advantages):
        self.update_calls += 1
        self.store.clear()
        return {"loss": 0.25, "seen_advantages": dict(advantages)}

    def reset(self):
        self.reset_calls += 1
        self.store.clear()


def raw_selector(seed_base=1000):
    return lambda update_index, group_index, used: (seed_base + update_index * 100 + group_index, "raw")


def test_batch_accumulator_preserves_group_local_advantages_and_zero_groups():
    acc = BatchAccumulator()
    acc.add_group(
        env_seed=10,
        group_index=0,
        advantages={"g0-a": -1.0, "g0-b": 1.0},
        chunks_by_traj={"g0-a": 2, "g0-b": 3},
        summary={"returns": [0.0, 1.0]},
    )
    acc.add_group(
        env_seed=11,
        group_index=1,
        advantages={"g1-a": 0.0, "g1-b": 0.0},
        chunks_by_traj={"g0-a": 2, "g0-b": 3, "g1-a": 4, "g1-b": 5},
        summary={"returns": [0.0, 0.0]},
    )
    assert acc.advantages == {"g0-a": -1.0, "g0-b": 1.0, "g1-a": 0.0, "g1-b": 0.0}
    assert acc.stored_chunks == 14
    assert acc.trainable_chunks == 5
    assert acc.group_env_seeds == [10, 11]


def test_batch_accumulator_rejects_duplicate_seed_trajectory_and_incomplete_store():
    acc = BatchAccumulator()
    acc.add_group(1, 0, {"a": 1.0}, {"a": 2}, {})
    with raises(RuntimeError, "duplicate env seed"):
        acc.add_group(1, 1, {"b": 1.0}, {"a": 2, "b": 2}, {})
    with raises(RuntimeError, "duplicate trajectory"):
        acc.add_group(2, 1, {"a": 1.0}, {"a": 2}, {})
    with raises(RuntimeError, "store trajectory mismatch"):
        acc.add_group(2, 1, {"b": 1.0}, {"a": 2}, {})


def test_execute_batches_all_groups_and_updates_exactly_once():
    client = FakeClient([
        [(-1.0, 2), (1.0, 3)],
        [(0.0, 4), (0.0, 5)],
        [(-0.5, 2), (0.5, 2)],
    ])
    result = execute_batched_update(
        update_index=3,
        min_groups=2,
        min_trainable_chunks=8,
        max_groups=3,
        select_seed=raw_selector(),
        collect_group=client.collect,
        get_store=client.metrics,
        update=client.update,
        reset_store=client.reset,
    )
    assert client.update_calls == 1
    assert client.reset_calls == 0
    assert result["groups_collected"] == 3
    assert result["trainable_chunks_collected"] == 9
    assert len(result["seen_advantages"]) == 6
    assert len(result["group_env_seeds"]) == 3


def test_execute_resets_on_stale_store_exception_and_max_cap():
    stale = FakeClient([[(-1.0, 1), (1.0, 1)]], stale=True)
    with raises(RuntimeError, "store not empty"):
        execute_batched_update(
            update_index=0, min_groups=1, min_trainable_chunks=0, max_groups=1,
            select_seed=raw_selector(), collect_group=stale.collect,
            get_store=stale.metrics, update=stale.update, reset_store=stale.reset,
        )
    assert stale.reset_calls == 1

    failed = FakeClient([[(-1.0, 1), (1.0, 1)]], fail_group=0)
    with raises(RuntimeError, "injected"):
        execute_batched_update(
            update_index=0, min_groups=1, min_trainable_chunks=0, max_groups=1,
            select_seed=raw_selector(), collect_group=failed.collect,
            get_store=failed.metrics, update=failed.update, reset_store=failed.reset,
        )
    assert failed.reset_calls == 1

    capped = FakeClient([[(0.0, 2), (0.0, 2)], [(0.0, 2), (0.0, 2)]])
    with raises(RuntimeError, "chunk floor not reached"):
        execute_batched_update(
            update_index=0, min_groups=1, min_trainable_chunks=1, max_groups=2,
            select_seed=raw_selector(), collect_group=capped.collect,
            get_store=capped.metrics, update=capped.update, reset_store=capped.reset,
        )
    assert capped.update_calls == 0
    assert capped.reset_calls == 1


def test_adaptive_per_group_selection_is_distinct_and_resume_bit_exact(tmp_path):
    kwargs = dict(seed_base=100, universe=32, band=(1, 7), group=8,
                  explore_frac=0.25, ema=0.5, rng_seed=9, avoid_recent=3)
    direct = AdaptiveCurriculum(**kwargs)
    first = []
    for group_index, n_success in enumerate([0, 3, 8, 4]):
        seed, source = direct.next_seed(group_index, exclude={x[0] for x in first})
        first.append((seed, source))
        direct.update(seed, n_success, it=group_index, source=source)
    state_path = tmp_path / "seed_difficulty.json"
    state_path.write_text(json.dumps(direct.to_dict()))

    resumed = AdaptiveCurriculum(**kwargs)
    resumed.load_state(json.loads(state_path.read_text()))
    direct_next = []
    resumed_next = []
    for group_index in range(4, 14):
        a = direct.next_seed(group_index)
        b = resumed.next_seed(group_index)
        direct_next.append(a)
        resumed_next.append(b)
        direct.update(a[0], group_index % 9, it=group_index, source=a[1])
        resumed.update(b[0], group_index % 9, it=group_index, source=b[1])
    assert len({seed for seed, _ in first}) == len(first)
    assert resumed_next == direct_next
    assert resumed.to_dict() == direct.to_dict()


def test_curriculum_transaction_rolls_back_failure_and_commits_atomically(tmp_path):
    curriculum = AdaptiveCurriculum(100, 16, (1, 7), 8, rng_seed=4)
    initial_seed, initial_source = curriculum.next_seed(-1)
    curriculum.update(initial_seed, 2, it=-1, source=initial_source)
    before = json.loads(json.dumps(curriculum.to_dict()))
    cache = tmp_path / "seed_difficulty.json"
    try:
        with CurriculumTransaction(curriculum, cache):
            seed, source = curriculum.next_seed(0)
            curriculum.update(seed, 3, it=0, source=source)
            raise RuntimeError("injected batch failure")
    except RuntimeError:
        pass
    assert curriculum.to_dict() == before
    assert not cache.exists()
    assert not list(tmp_path.glob("*.tmp"))

    with CurriculumTransaction(curriculum, cache) as transaction:
        seed, source = curriculum.next_seed(0)
        curriculum.update(seed, 3, it=0, source=source)
        transaction.commit()
    assert json.loads(cache.read_text()) == curriculum.to_dict()
    assert not list(tmp_path.glob("*.tmp"))


def test_idempotent_update_cache_executes_each_update_id_once():
    cache = IdempotentUpdateCache()
    calls = []

    def operation():
        calls.append("step")
        return {"loss": 0.25}

    first = cache.run("run-a:update-3", operation)
    replay = cache.run("run-a:update-3", operation)
    assert first == replay == {"loss": 0.25}
    assert calls == ["step"]


def test_idempotent_update_cache_poisons_uncertain_failure():
    cache = IdempotentUpdateCache()
    calls = []

    def uncertain():
        calls.append("step-may-have-happened")
        raise RuntimeError("post-step diagnostic failed")

    for _ in range(2):
        try:
            cache.run("run-a:update-4", uncertain)
        except RuntimeError as error:
            assert "post-step diagnostic failed" in str(error) or "poisoned" in str(error)
        else:
            raise AssertionError("uncertain update failure was retried")
    assert calls == ["step-may-have-happened"]


def test_response_loss_replays_same_update_and_original_curriculum_state():
    cache = IdempotentUpdateCache()
    reconnects = []
    sends = []

    def send(request):
        sends.append(request["update_id"])
        result = cache.run(request["update_id"], lambda: {
            "loss": 0.25,
            "curriculum_state": {"history": ["optimized-batch"]},
        })
        if len(sends) == 1:
            raise EOFError("response lost after commit")
        return result

    result = retry_idempotent_rpc(
        send, lambda: reconnects.append(True),
        {"update_id": "run-a:5", "curriculum_state": {"history": ["optimized-batch"]}},
    )
    assert sends == ["run-a:5", "run-a:5"]
    assert reconnects == [True]
    assert result["curriculum_state"] == {"history": ["optimized-batch"]}


def test_recv_exact_raises_on_partial_body_eof():
    class PartialSocket:
        def __init__(self):
            self.parts = [b"ab", b""]

        def recv(self, size):
            del size
            return self.parts.pop(0)

    with raises(EOFError, "connection closed"):
        recv_exact(PartialSocket(), 4)


def test_mutated_batch_traces_each_fail_the_audit():
    clean = {
        "groups_collected": 2,
        "update_rpc_calls": 1,
        "group_env_seeds": [100, 101],
        "group_summaries": [
            {"group_index": 0, "trajectory_ids": ["a", "b"], "advantages": {"a": -1.0, "b": 1.0}},
            {"group_index": 1, "trajectory_ids": ["c", "d"], "advantages": {"c": 0.0, "d": 0.0}},
        ],
        "advantages": {"a": -1.0, "b": 1.0, "c": 0.0, "d": 0.0},
        "stored_trajectory_ids": ["a", "b", "c", "d"],
        "joint_logprob": False,
        "ratio_shape": [4, 12],
        "mask_shape": [4, 12],
    }
    assert audit_batch_trace(clean)["pass"]
    mutations = {
        "duplicate_seed": lambda x: x.update(group_env_seeds=[100, 100]),
        "global_advantage": lambda x: x["group_summaries"][0].update(advantages={"a": -2.0, "b": 2.0}),
        "double_update": lambda x: x.update(update_rpc_calls=2),
        "missing_last_group": lambda x: x.update(stored_trajectory_ids=["a", "b"]),
        "trajectory_collision": lambda x: x["group_summaries"][1].update(trajectory_ids=["b", "d"]),
        "joint_ratio": lambda x: x.update(joint_logprob=True),
        "ratio_mask_shape": lambda x: x.update(mask_shape=[4, 11]),
    }
    for name, mutate in mutations.items():
        sample = copy.deepcopy(clean)
        mutate(sample)
        report = audit_batch_trace(sample)
        assert not report["pass"], f"mutation escaped: {name}"


if __name__ == "__main__":
    import tempfile

    passed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if name in {
            "test_adaptive_per_group_selection_is_distinct_and_resume_bit_exact",
            "test_curriculum_transaction_rolls_back_failure_and_commits_atomically",
        }:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
        else:
            fn()
        passed += 1
    print(f"{passed} multigroup baseline tests passed")
