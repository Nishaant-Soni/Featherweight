"""Group 1 golden tests: the tool-call schema round-trips on real-shaped data.

Examples mirror actual rows from `minpeter/xlam-function-calling-60k-parsed`
(observed during Group 1 investigation) but are hand-written so the tests are
offline and deterministic. They prove the highest-risk Phase 1 item: a gold
example serialized to our training target and parsed back yields the correct
(tool_name, arguments) in BFCL's decode_ast form.
"""

import json

from featherweight.data import schema
from featherweight.data.schema import ToolCall

# Real row shape: one parallel example (two calls), one single, one irrelevance.
PARALLEL_MESSAGES = [
    {"role": "user", "content": "live giveaways for beta and games?", "tool_calls": None},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "live_giveaways_by_type", "arguments": '{"type": "beta"}'},
            },
            {
                "type": "function",
                "function": {"name": "live_giveaways_by_type", "arguments": '{"type": "game"}'},
            },
        ],
    },
]

SINGLE_MESSAGES = [
    {"role": "user", "content": "fetch details for ethereum", "tool_calls": None},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "web_chain_details",
                    "arguments": '{"chain_slug": "ethereum"}',
                },
            },
        ],
    },
]

IRRELEVANCE_MESSAGES = [
    {"role": "user", "content": "Who won the 2019 NCAA Final Four?", "tool_calls": None},
    {"role": "assistant", "content": "No available tool answers that.", "tool_calls": None},
]


def test_single_call_roundtrip():
    calls = schema.extract_calls(SINGLE_MESSAGES)
    assert calls == [ToolCall("web_chain_details", {"chain_slug": "ethereum"})]

    target = schema.serialize_calls(calls)
    assert json.loads(target) == [
        {"name": "web_chain_details", "arguments": {"chain_slug": "ethereum"}}
    ]
    assert schema.decode_ast(target) == [{"web_chain_details": {"chain_slug": "ethereum"}}]


def test_parallel_calls_roundtrip():
    calls = schema.extract_calls(PARALLEL_MESSAGES)
    assert len(calls) == 2

    target = schema.serialize_calls(calls)
    assert schema.decode_ast(target) == [
        {"live_giveaways_by_type": {"type": "beta"}},
        {"live_giveaways_by_type": {"type": "game"}},
    ]


def test_irrelevance_is_empty_array():
    calls = schema.extract_calls(IRRELEVANCE_MESSAGES)
    assert calls == []
    assert schema.serialize_calls(calls) == "[]"
    assert schema.decode_ast("[]") == []


def test_decode_ast_semicolon_fallback():
    # SalesforceLlamaHandler falls back to ';'-separated objects when not valid JSON.
    text = '{"name": "a", "arguments": {"x": 1}}; {"name": "b", "arguments": {}}'
    assert schema.decode_ast(text) == [{"a": {"x": 1}}, {"b": {}}]


def test_decode_ast_single_object_wrapped():
    text = '{"name": "a", "arguments": {"x": 1}}'
    assert schema.decode_ast(text) == [{"a": {"x": 1}}]
