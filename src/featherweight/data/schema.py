"""Tool-call schema — the contract between our training targets and BFCL eval.

Locked in Phase 1 / Group 1 after reading BFCL's SalesforceLlamaHandler
(`bfcl_eval/model_handler/local_inference/salesforce_llama.py`). Our training
data is Salesforce xLAM and our base is Llama-3.1, so we train the model to emit
exactly the format that handler parses and reuse that handler at eval time — no
bespoke handler. See docs/iteration_1.md for the full rationale.

Assistant target the model is trained to emit — a single JSON array, one object
per call:

    [{"name": <fn>, "arguments": {<arg>: <val>, ...}}, ...]

Parallel/multiple calls add array elements; "no tool applies" (irrelevance) is
the empty array ``[]``. This array form is why we picked the Salesforce handler
over the native Llama-3.1 one, whose chat template only supports a single call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    """One function call: a name and a mapping of argument name -> value."""

    name: str
    arguments: dict


def extract_calls(messages: list[dict]) -> list[ToolCall]:
    """Pull the gold tool calls out of a dataset row's assistant message(s).

    The parsed xLAM dataset uses the OpenAI shape:
    ``assistant.tool_calls[i] = {"function": {"name": str, "arguments": <json
    string>}, "type": "function"}``. ``arguments`` is a JSON *string* there, so
    it is parsed into a dict. An assistant turn with no tool_calls yields ``[]``
    (the irrelevance case).
    """
    calls: list[ToolCall] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            args = fn["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(ToolCall(name=fn["name"], arguments=args))
    return calls


def serialize_calls(calls: list[ToolCall]) -> str:
    """Render the assistant training target: the SalesforceLlama JSON array."""
    return json.dumps([{"name": c.name, "arguments": c.arguments} for c in calls])


def decode_ast(text: str) -> list[dict]:
    """Parse raw model output into the AST form ``[{name: arguments}, ...]``.

    Mirrors ``SalesforceLlamaHandler.decode_ast``: parse the whole string as
    JSON; on failure, fall back to ';'-separated JSON objects. A lone object is
    wrapped into a single-element list.
    """
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(part.strip()) for part in text.split(";") if part.strip()]
    if isinstance(parsed, dict):
        parsed = [parsed]
    return [{call["name"]: call["arguments"]} for call in parsed]
