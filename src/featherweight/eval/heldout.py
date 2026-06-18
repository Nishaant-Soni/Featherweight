"""Internal held-out AST scorer — a fast, in-distribution proxy for iteration.

Given model-generated completions and the gold completions, decode both with
``schema.decode_ast`` and compute pragmatic exact-set metrics. This is **not**
BFCL's full AST checker (BFCL is the real number in Phase 3/4); it's the cheap
signal the Phase 2 training callback uses to confirm the fine-tune beats the base
model, and a quick iteration metric.

Predictions are raw model output, so they may carry a trailing ``<|eot_id|>`` or
be malformed (the base model emits junk); a prediction that doesn't parse counts
as ``invalid`` and a non-match. Parallel calls are compared order-insensitively.
"""

from __future__ import annotations

import json
from pathlib import Path

from featherweight import config
from featherweight.data import schema

_EOT = "<|eot_id|>"


def _safe_decode(text: str) -> list[dict] | None:
    """Decode a completion to ``[{name: args}, ...]``; ``None`` if it won't parse."""
    text = text.strip()
    if text.endswith(_EOT):
        text = text[: -len(_EOT)].strip()
    try:
        return schema.decode_ast(text)
    except (ValueError, KeyError, TypeError):
        return None


def _names(calls: list[dict]) -> list[str]:
    """Sorted multiset of called tool names (decode_ast yields single-key dicts)."""
    return sorted(next(iter(call)) for call in calls)


def _canonical(calls: list[dict]) -> list[str]:
    """Order-insensitive canonical form: sorted list of sorted-key JSON per call."""
    return sorted(json.dumps(call, sort_keys=True) for call in calls)


def score(predictions: list[str], golds: list[str]) -> dict:
    """Compute held-out metrics from model completions vs gold completions.

    Returns tool-name accuracy, exact-match accuracy (name + args; the headline),
    refusal accuracy over the irrelevance rows, the irrelevance count, and the
    invalid-output rate. The name-vs-exact gap isolates argument errors.
    """
    assert len(predictions) == len(golds), "predictions and golds must be the same length"
    n = len(golds)
    name_correct = exact_correct = invalid = 0
    n_refusal = refusal_correct = 0

    for pred_text, gold_text in zip(predictions, golds):
        gold = _safe_decode(gold_text) or []
        is_refusal = not gold
        if is_refusal:
            n_refusal += 1

        pred = _safe_decode(pred_text)
        if pred is None:
            invalid += 1
            continue

        if _names(pred) == _names(gold):
            name_correct += 1
        if _canonical(pred) == _canonical(gold):
            exact_correct += 1
        if is_refusal and not pred:
            refusal_correct += 1

    return {
        "n": n,
        "tool_name_accuracy": name_correct / n,
        "exact_match_accuracy": exact_correct / n,
        "refusal_accuracy": refusal_correct / n_refusal if n_refusal else 0.0,
        "n_refusal": n_refusal,
        "invalid_rate": invalid / n,
    }


def load_heldout(path: Path | None = None) -> list[dict]:
    """Load the held-out JSONL as a list of ``{"prompt", "completion"}`` rows."""
    path = path or config.DATA_DIR / "heldout.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
