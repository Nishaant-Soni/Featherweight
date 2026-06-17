"""Load and cache the training + irrelevance datasets, with a one-time shape audit.

Both datasets are ungated (CC-BY-4.0), so no HF token is required. Everything is
cached under ``data/hf_cache`` (git-ignored) so downstream groups and Colab reuse
the same local copy.

Run the audit (downloads both, checks counts + field shapes, prints a sample)
with:  ``python -m featherweight.data.load``
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import cast

from datasets import Dataset, load_dataset

from featherweight import config

_CACHE_DIR = str(config.DATA_DIR / "hf_cache")


def load_xlam() -> Dataset:
    """Main xLAM function-calling data (`messages` / `tools` / `extra`)."""
    # split= returns a single Dataset (not a DatasetDict); cast for the type checker.
    return cast(Dataset, load_dataset(config.TRAIN_DATASET, split="train", cache_dir=_CACHE_DIR))


def load_irrelevance() -> Dataset:
    """Irrelevance data (`query` / `tools` / `answers`); gold answer is no call."""
    return cast(
        Dataset, load_dataset(config.IRRELEVANCE_DATASET, split="train", cache_dir=_CACHE_DIR)
    )


def audit() -> None:
    """One-time data audit: Group 2 verification + the Group 1 carry-over check.

    Confirms row counts, that every main-dataset assistant ``arguments`` is a JSON
    *string* (so the ``isinstance`` guard in ``schema.extract_calls`` is dead), and
    that the ``tools`` / ``answers`` JSON-string fields parse. Raises on violation.
    """
    xlam = load_xlam()
    irr = load_irrelevance()
    print(f"xlam rows:        {len(xlam)}")
    print(f"irrelevance rows: {len(irr)}")

    non_string_args = 0
    for row in cast(Iterable[dict], xlam):
        json.loads(row["tools"])  # tools is a JSON string -> must parse
        for msg in row["messages"]:
            if msg["role"] != "assistant":
                continue
            for tc in msg["tool_calls"] or []:
                if not isinstance(tc["function"]["arguments"], str):
                    non_string_args += 1
    print(f"xlam assistant args that are NOT strings: {non_string_args}")

    for row in cast(Iterable[dict], irr):
        json.loads(row["tools"])
        json.loads(row["answers"])  # gold; "[]" for a true irrelevance row
    print("irrelevance tools/answers all parse as JSON")

    assert non_string_args == 0, (
        f"{non_string_args} non-string arguments found — schema.extract_calls assumes "
        "JSON-string arguments; re-add dict-argument handling there if this fires"
    )
    print("\nAUDIT OK")


if __name__ == "__main__":
    audit()
