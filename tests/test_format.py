"""Group 3 tests: row -> (prompt, completion) formatting + tool normalization.

Golden inputs mirror real rows from both datasets (observed in Groups 1-2) but
are hand-written so the tests are offline and deterministic.
"""

import json

import pytest

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


def test_normalize_openai_function_always_emits_properties_and_required():
    # No-arg function: BFCL always renders both keys, so we must too.
    fn = {"name": "g", "parameters": {"type": "object"}}
    out = fmt.normalize_openai_function(fn)
    assert out["parameters"]["type"] == "dict"
    assert out["parameters"]["properties"] == {}
    assert out["parameters"]["required"] == []


# Byte-exact snapshot of the prompt contract (the fragile train/eval surface).
# Guards OUR build_prompt from silent drift; the bfcl-eval pin (Phase 3) guards
# the handler's side. If this fails, build_prompt changed — re-verify vs the
# pinned SalesforceLlamaHandler before updating.
EXPECTED_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team.\n"
    "You have access to a set of tools. When using tools, make calls in a single JSON array: \n\n"
    '[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, '
    "... (additional parallel tool calls as needed)]\n\n"
    "If no tool is suitable, state that explicitly. If the user's input lacks required "
    "parameters, ask for clarification. Do not interpret or respond until tool results are "
    "returned. Once they are available, process them or make additional calls if needed. "
    "For tasks that don't require tools, such as casual conversation or general advice, "
    "respond directly in plain text. The available tools are:\n\n"
    "{\n"
    '    "name": "f",\n'
    '    "description": "",\n'
    '    "parameters": {\n'
    '        "type": "dict",\n'
    '        "properties": {},\n'
    '        "required": []\n'
    "    }\n"
    "}\n\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def test_build_prompt_byte_exact_snapshot():
    fn = {"name": "f", "parameters": {"type": "object", "properties": {}, "required": []}}
    prompt = fmt.build_prompt([fmt.normalize_openai_function(fn)], "hi")
    assert prompt == EXPECTED_PROMPT


def test_build_prompt_matches_pinned_handler():
    """Lock the *handler's* side of the contract against the pinned bfcl-eval.

    Skipped where bfcl-eval isn't installed (the local CPU venv); runs in the
    Colab serve env. Calls the real ``SalesforceLlamaHandler._format_prompt`` — it
    uses no instance state, so ``self=None`` is safe — and asserts ``build_prompt``
    reproduces it byte-for-byte. If this fails after a bfcl-eval bump, the upstream
    handler changed: re-verify before touching build_prompt / regenerating data.
    """
    pytest.importorskip("bfcl_eval")
    # bfcl-eval is a serve-env-only dep, absent from the local venv -> Pyright can't
    # resolve it; importorskip guards it at runtime.
    from bfcl_eval.model_handler.local_inference.salesforce_llama import (  # type: ignore[import-not-found]
        SalesforceLlamaHandler,
    )

    fn = {"name": "f", "parameters": {"type": "object", "properties": {}, "required": []}}
    functions = [fmt.normalize_openai_function(fn)]
    query = "hi"
    handler_prompt = SalesforceLlamaHandler._format_prompt(
        None, [{"role": "user", "content": query}], functions
    )
    assert fmt.build_prompt(functions, query) == handler_prompt
