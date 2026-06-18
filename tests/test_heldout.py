"""Group A (Phase 2) tests: the internal held-out AST scorer. All offline."""

import pytest

from featherweight.data import schema
from featherweight.eval import heldout


def _gold(calls: list[schema.ToolCall]) -> str:
    """A gold completion string as written to heldout.jsonl (with the eot marker)."""
    return schema.serialize_calls(calls) + "<|eot_id|>"


CALL = [schema.ToolCall("f", {"a": 1})]
OTHER = [schema.ToolCall("g", {"b": 2})]


def test_exact_match():
    m = heldout.score([_gold(CALL)], [_gold(CALL)])
    assert m["exact_match_accuracy"] == 1.0
    assert m["tool_name_accuracy"] == 1.0
    assert m["invalid_rate"] == 0.0


def test_wrong_argument_keeps_name_loses_exact():
    pred = schema.serialize_calls([schema.ToolCall("f", {"a": 999})])
    m = heldout.score([pred], [_gold(CALL)])
    assert m["tool_name_accuracy"] == 1.0  # right tool
    assert m["exact_match_accuracy"] == 0.0  # wrong argument


def test_wrong_name():
    m = heldout.score([_gold(OTHER)], [_gold(CALL)])
    assert m["tool_name_accuracy"] == 0.0
    assert m["exact_match_accuracy"] == 0.0


def test_correct_refusal():
    m = heldout.score(["[]"], ["[]<|eot_id|>"])
    assert m["n_refusal"] == 1
    assert m["refusal_accuracy"] == 1.0
    assert m["exact_match_accuracy"] == 1.0


def test_false_call_on_irrelevance():
    m = heldout.score([_gold(CALL)], ["[]<|eot_id|>"])
    assert m["n_refusal"] == 1
    assert m["refusal_accuracy"] == 0.0
    assert m["exact_match_accuracy"] == 0.0


def test_invalid_prediction_counts_as_invalid_not_match():
    m = heldout.score(["I cannot help with that"], [_gold(CALL)])
    assert m["invalid_rate"] == 1.0
    assert m["exact_match_accuracy"] == 0.0


def test_parallel_calls_order_insensitive():
    gold = _gold([schema.ToolCall("a", {"x": 1}), schema.ToolCall("b", {"y": 2})])
    pred = schema.serialize_calls([schema.ToolCall("b", {"y": 2}), schema.ToolCall("a", {"x": 1})])
    m = heldout.score([pred], [gold])
    assert m["exact_match_accuracy"] == 1.0


def test_length_mismatch_raises():
    with pytest.raises(AssertionError):
        heldout.score(["[]"], [])


def test_load_heldout(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"prompt": "P", "completion": "[]<|eot_id|>"}\n', encoding="utf-8")
    assert heldout.load_heldout(p) == [{"prompt": "P", "completion": "[]<|eot_id|>"}]
