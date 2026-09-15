"""Unit tests for the Goal-2 conditioning composer (pure python, no GPU)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import conditioning as C  # noqa: E402


def test_schema_loads_three_skills():
    sc = C.load_schema()
    assert set(sc["skills"]) == {"GRASP_HANDLE", "MOVE_LID_TO_CLOSED", "RELEASE_HANDLE"}
    assert [C.skill_id(sc, s) for s in ("GRASP_HANDLE", "MOVE_LID_TO_CLOSED", "RELEASE_HANDLE")] == [0, 1, 2]


def test_nl_only_is_just_nl():
    sc = C.load_schema()
    c = C.conditioning(sc, "GRASP_HANDLE", "nl_only")
    assert c["instruction"] == sc["skills"]["GRASP_HANDLE"]["instruction_natural_language"]
    assert c["lora_key"] == "GRASP_HANDLE"
    assert c["use_learned_embedding"] is False


def test_nl_plus_skill_id_contains_goal_and_id_and_constraints():
    sc = C.load_schema()
    c = C.conditioning(sc, "RELEASE_HANDLE", "nl_plus_skill_id")
    txt = c["instruction"]
    assert sc["overall_task_goal"] in txt
    assert "[2:RELEASE_HANDLE]" in txt
    assert "Constraints:" in txt and "retreat" in txt
    assert c["lora_key"] == "RELEASE_HANDLE"


def test_shared_embedding_text_is_goal_only_and_uses_shared_lora():
    sc = C.load_schema()
    for sk in sc["skills"]:
        c = C.conditioning(sc, sk, "shared_lora_learned_embedding")
        assert c["instruction"] == sc["overall_task_goal"]   # text identical across skills
        assert c["use_learned_embedding"] is True
        assert c["lora_key"] == "shared"


def test_arms_produce_distinct_text_for_a_skill():
    sc = C.load_schema()
    txts = {arm: C.build_instruction(sc, "MOVE_LID_TO_CLOSED", arm) for arm in C.ARMS}
    # nl_only vs nl_plus_skill_id must differ; shared uses the goal only
    assert txts["nl_only"] != txts["nl_plus_skill_id"]
    assert txts["shared_lora_learned_embedding"] == sc["overall_task_goal"]


def test_unknown_arm_raises():
    sc = C.load_schema()
    try:
        C.build_instruction(sc, "GRASP_HANDLE", "bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown arm")


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print("PASS", fn.__name__)
        except Exception:
            print("FAIL", fn.__name__); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
