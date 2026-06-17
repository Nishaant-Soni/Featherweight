"""Group 3 tests: row -> (prompt, completion) formatting + tool normalization.

Golden inputs mirror real rows from both datasets (observed in Groups 1-2) but
are hand-written so the tests are offline and deterministic.
"""

import json

from featherweight.data import format as fmt
from featherweight.data import schema

# Main-dataset shape: messages (list) + tools (JSON string, OpenAI-wrapped).
XLAM_ROW = {
    "tools": json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "web_chain_details",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain_slug": {
                                "type": "string",
                                "description": "slug",
                                "default": "ethereum",
                            }
                        },
                        "required": ["chain_slug"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    ),
    "messages": [
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
                }
            ],
        },
    ],
}

# Irrelevance shape: query + tools (JSON string, xLAM-native flat) + answers "[]".
IRR_ROW = {
    "query": "Who won the 2019 NCAA Final Four?",
    "tools": json.dumps(
        [
            {
                "name": "raceresult",
                "description": "Fetches an F1 race result.",
                "parameters": {
                    "round": {"description": "round", "type": "str", "default": "round"},
                    "year": {"description": "year", "type": "str, optional"},
                },
            }
        ]
    ),
    "answers": "[]",
}


def test_normalize_openai_function():
    fn = {
        "name": "f",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer", "description": "d"}},
            "required": ["x"],
            "additionalProperties": False,
        },
    }
    out = fmt.normalize_openai_function(fn)
    assert out["name"] == "f"
    assert out["description"] == ""
    assert out["parameters"]["type"] == "dict"
    assert "additionalProperties" not in out["parameters"]
    assert out["parameters"]["properties"] == {"x": {"type": "integer", "description": "d"}}
    assert out["parameters"]["required"] == ["x"]


def test_normalize_xlam_tool_types_and_required():
    tool = json.loads(IRR_ROW["tools"])[0]
    out = fmt.normalize_xlam_tool(tool)
    assert out["parameters"]["type"] == "dict"
    assert out["parameters"]["properties"]["round"]["type"] == "string"  # str -> string
    assert out["parameters"]["properties"]["round"]["default"] == "round"  # default kept
    assert out["parameters"]["properties"]["year"]["type"] == "string"  # "str, optional" -> string
    assert out["parameters"]["required"] == ["round"]  # year is optional -> excluded


def test_normalize_xlam_passthrough_complex_type():
    tool = {
        "name": "t",
        "description": "",
        "parameters": {"vals": {"description": "v", "type": "List[int]"}},
    }
    out = fmt.normalize_xlam_tool(tool)
    assert out["parameters"]["properties"]["vals"]["type"] == "List[int]"  # unknown -> passthrough
    assert out["parameters"]["required"] == ["vals"]


def test_format_xlam_row_prompt_and_completion():
    ex = fmt.format_xlam_row(XLAM_ROW)
    p = ex["prompt"]
    assert "<|begin_of_text|>" in p
    assert "<|start_header_id|>system<|end_header_id|>" in p
    assert fmt.DEFAULT_SYSTEM_MESSAGE in p
    assert "make calls in a single JSON array" in p
    assert "web_chain_details" in p  # tool schema rendered
    assert "fetch details for ethereum" in p  # user query
    assert p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    assert ex["completion"].endswith("<|eot_id|>")
    payload = ex["completion"].removesuffix("<|eot_id|>")
    assert json.loads(payload) == [
        {"name": "web_chain_details", "arguments": {"chain_slug": "ethereum"}}
    ]
    assert schema.decode_ast(payload) == [{"web_chain_details": {"chain_slug": "ethereum"}}]


def test_format_irrelevance_row_empty_completion():
    ex = fmt.format_irrelevance_row(IRR_ROW)
    assert ex["completion"] == "[]<|eot_id|>"
    assert "Who won the 2019 NCAA Final Four?" in ex["prompt"]
    assert "raceresult" in ex["prompt"]
    assert '"type": "dict"' in ex["prompt"]  # tool normalized to nested form
