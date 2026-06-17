"""Build formatted train/held-out JSONL from the cached datasets (Phase 1, Group 4).

Loads both datasets, formats every row to {prompt, completion}, blends in the
irrelevance mix, and writes the deterministic split to data/ (git-ignored).

Run:  python scripts/prep_data.py
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from featherweight import config
from featherweight.data import format as fmt
from featherweight.data import load, mix


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    xlam = [fmt.format_xlam_row(r) for r in cast(Iterable[dict], load.load_xlam())]
    irr = [fmt.format_irrelevance_row(r) for r in cast(Iterable[dict], load.load_irrelevance())]

    dc = config.CONFIG.data
    train, heldout = mix.mix_and_split(
        xlam, irr, ratio=dc.irrelevance_ratio, heldout_size=dc.heldout_size, seed=dc.seed
    )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.DATA_DIR / "train.jsonl", train)
    _write_jsonl(config.DATA_DIR / "heldout.jsonl", heldout)

    total = len(train) + len(heldout)
    n_irr = sum(1 for e in train + heldout if e["completion"] == "[]<|eot_id|>")
    print(f"xlam formatted:     {len(xlam)}")
    print(f"irrelevance formatted: {len(irr)}")
    print(f"train: {len(train)}   heldout: {len(heldout)}   total: {total}")
    print(f"irrelevance in mix: {n_irr} ({n_irr / total:.1%})")
    print(f"wrote {config.DATA_DIR / 'train.jsonl'} and {config.DATA_DIR / 'heldout.jsonl'}")


if __name__ == "__main__":
    main()
