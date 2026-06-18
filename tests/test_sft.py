"""Group B (Phase 2) tests: SFT data shaping + the lazy-GPU-import contract.

The Unsloth `train()` itself is verified on Colab (Group C), not here.
"""

import json
import sys

from featherweight.train import sft


def test_load_sft_dataset_concatenates_prompt_and_completion(tmp_path):
    p = tmp_path / "train.jsonl"
    rows = [
        {"prompt": f"...{sft.RESPONSE_MARKER}", "completion": "[]<|eot_id|>"},
        {
            "prompt": f"q{sft.RESPONSE_MARKER}",
            "completion": '[{"name": "f", "arguments": {}}]<|eot_id|>',
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    ds = sft.load_sft_dataset(p)
    assert ds.column_names == ["text"]
    assert len(ds) == 2
    assert ds[0]["text"] == rows[0]["prompt"] + rows[0]["completion"]
    # The response marker must survive — it's the masking boundary for training.
    assert sft.RESPONSE_MARKER in ds[0]["text"]


def test_load_eval_subset_is_mixed(tmp_path):
    # 8 tool-call rows + 2 irrelevance rows; the eval subset must contain both.
    p = tmp_path / "heldout.jsonl"
    rows = [
        {"prompt": f"p{i}", "completion": '[{"name": "f", "arguments": {}}]<|eot_id|>'}
        for i in range(8)
    ]
    rows += [{"prompt": f"q{i}", "completion": "[]<|eot_id|>"} for i in range(2)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    ds = sft.load_eval_subset(p, size=5, seed=1)
    assert ds.column_names == ["text"]
    assert len(ds) == 5
    n_irr = sum(1 for t in ds["text"] if t.endswith("[]<|eot_id|>"))
    assert 1 <= n_irr < 5  # guaranteed mix: both irrelevance and tool-call rows


def test_no_gpu_imports_at_module_level():
    # train/sft must import on the CPU-only Mac; unsloth/trl are imported lazily
    # inside train(). If either were a top-level import, this module's import
    # would have already failed (they aren't installed locally).
    assert "unsloth" not in sys.modules
    assert "trl" not in sys.modules
