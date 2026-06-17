"""Format dataset rows into (prompt, completion) training examples.

The prompt replicates the prompt BFCL's ``SalesforceLlamaHandler._format_prompt``
builds at eval time (verbatim system block + tools via ``json.dumps(func,
indent=4)`` + the user turn), so train and eval see the same surface. The
completion is the gold JSON-array tool call from ``schema.serialize_calls``.

Both source datasets are normalized to BFCL's ``function`` shape
``{name, description, parameters: {type: "dict", properties, required}}`` — the
exact structure the handler renders — so the bulk xLAM data and the irrelevance
mix look identical to the model. See docs/iteration_3.md for the rationale.
"""

from __future__ import annotations

import json

from featherweight.data import schema

# Verbatim from SalesforceLlamaHandler._format_prompt (gorilla repo).
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team."
)
_TOOL_INSTRUCTIONS = (
    "You have access to a set of tools. When using tools, make calls in a single JSON array: \n\n"
    '[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, '
    "... (additional parallel tool calls as needed)]\n\n"
    "If no tool is suitable, state that explicitly. If the user's input lacks required "
    "parameters, ask for clarification. Do not interpret or respond until tool results are "
    "returned. Once they are available, process them or make additional calls if needed. "
    "For tasks that don't require tools, such as casual conversation or general advice, "
    "respond directly in plain text. The available tools are:\n\n"
)

# xLAM-native param type strings -> JSON-schema vocabulary BFCL uses. Unknown
# types (e.g. "List[int]") pass through unchanged — still human-readable.
_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "tuple": "array",
    "set": "array",
}


def normalize_openai_function(fn: dict) -> dict:
    """Main dataset: unwrap the OpenAI `function` dict into BFCL `function` shape.

    Already JSON-schema with `string`/`integer` types; we only set `type` to
    `"dict"` and drop `additionalProperties` to match BFCL's rendering. The main
    dataset carries no function-level description, so it is "".
    """
    params = fn["parameters"]
    norm_params: dict = {"type": "dict"}
    if "properties" in params:
        norm_params["properties"] = params["properties"]
    if "required" in params:
        norm_params["required"] = params["required"]
    return {"name": fn["name"], "description": fn.get("description", ""), "parameters": norm_params}


def normalize_xlam_tool(tool: dict) -> dict:
    """Irrelevance dataset: convert xLAM-native flat params into BFCL nested shape.

    Maps scalar type strings to JSON-schema vocab, treats a trailing
    ``, optional`` as not-required, and keeps any per-arg ``default``.
    """
    properties: dict = {}
    required: list[str] = []
    for arg, spec in tool.get("parameters", {}).items():
        raw_type = (spec.get("type") or "").strip()
        optional = raw_type.endswith("optional")
        base_type = raw_type.rsplit(",", 1)[0].strip() if optional else raw_type
        prop: dict = {
            "type": _TYPE_MAP.get(base_type, base_type),
            "description": spec.get("description", ""),
        }
        if "default" in spec:
            prop["default"] = spec["default"]
        properties[arg] = prop
        if not optional:
            required.append(arg)
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": {"type": "dict", "properties": properties, "required": required},
    }


def build_prompt(functions: list[dict], user_query: str) -> str:
    """Replicate the SalesforceLlamaHandler prompt up to the assistant header."""
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    prompt += DEFAULT_SYSTEM_MESSAGE + "\n"
    prompt += _TOOL_INSTRUCTIONS
    for func in functions:
        prompt += json.dumps(func, indent=4) + "\n\n"
    prompt += "<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_query.strip()}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def format_xlam_row(row: dict) -> dict:
    """A main-dataset row -> {"prompt", "completion"} with the gold tool calls."""
    functions = [normalize_openai_function(t["function"]) for t in json.loads(row["tools"])]
    user_query = next(m["content"] for m in row["messages"] if m["role"] == "user")
    calls = schema.extract_calls(row["messages"])
    return {
        "prompt": build_prompt(functions, user_query),
        "completion": schema.serialize_calls(calls) + "<|eot_id|>",
    }


def format_irrelevance_row(row: dict) -> dict:
    """An irrelevance row -> {"prompt", "completion"}; gold is the empty array."""
    functions = [normalize_xlam_tool(t) for t in json.loads(row["tools"])]
    return {
        "prompt": build_prompt(functions, row["query"]),
        "completion": schema.serialize_calls([]) + "<|eot_id|>",
    }
