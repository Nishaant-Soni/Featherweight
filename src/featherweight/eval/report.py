"""Turn bfcl-eval score files into a consolidated baselines table.

`bfcl evaluate` writes one score file per category:
``<score_dir>/<model>/.../BFCL_v4_<category>_score.json``. Each is **JSONL**:

- line 1 — summary header ``{"accuracy", "correct_count", "total_count"}``
- lines 2+ — one failed entry each, carrying an ``"error_type"``
  (e.g. ``ast_decoder:decoder_failed`` = the model output couldn't be parsed).

We read per-category accuracy and derive an **invalid-JSON rate** from the share of
``decoder_failed`` entries (unparseable output), then emit
``results/baselines.{csv,md}`` for base vs GPT-4o (and FT in Phase 4).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from featherweight import config

_PREFIX = "BFCL_v4_"
_SUFFIX = "_score.json"


def category_from_filename(name: str) -> str:
    """``BFCL_v4_parallel_score.json`` -> ``parallel``."""
    return name.removeprefix(_PREFIX).removesuffix(_SUFFIX)


def parse_score_file(path: Path) -> dict:
    """Parse one ``*_score.json`` into ``{accuracy, correct_count, total_count,
    invalid_rate}``. ``invalid_rate`` = share of entries whose ``error_type`` marks
    an unparseable decode (``decoder_failed``), over ``total_count``."""
    lines = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    header = lines[0]
    total = header["total_count"]
    n_invalid = sum(1 for e in lines[1:] if "decoder_failed" in e.get("error_type", ""))
    return {
        "accuracy": header["accuracy"],
        "correct_count": header["correct_count"],
        "total_count": total,
        "invalid_rate": n_invalid / total if total else 0.0,
    }


def collect_scores(model_dir: Path) -> dict[str, dict]:
    """All per-category stats under one model's score dir, keyed by category."""
    return {
        category_from_filename(p.name): parse_score_file(p)
        for p in sorted(model_dir.rglob(f"{_PREFIX}*{_SUFFIX}"))
    }


def _overall(per_cat: dict[str, dict]) -> dict:
    """Count-weighted overall accuracy + invalid rate across categories."""
    total = sum(s["total_count"] for s in per_cat.values())
    if not total:
        return {"accuracy": 0.0, "invalid_rate": 0.0, "total_count": 0}
    correct = sum(s["correct_count"] for s in per_cat.values())
    invalid = sum(s["invalid_rate"] * s["total_count"] for s in per_cat.values())
    return {"accuracy": correct / total, "invalid_rate": invalid / total, "total_count": total}


def write_baselines(
    scores_by_model: dict[str, dict[str, dict]],
    out_dir: Path = config.RESULTS_DIR,
    categories: tuple[str, ...] = config.CONFIG.eval.categories,
) -> tuple[Path, Path]:
    """Write ``baselines.csv`` + ``baselines.md`` (model x category accuracy, plus
    overall accuracy and invalid-JSON rate). Returns the two paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(categories)
    csv_path = out_dir / "baselines.csv"
    md_path = out_dir / "baselines.md"

    def acc(per_cat: dict, cat: str) -> str:
        return f"{per_cat[cat]['accuracy'] * 100:.2f}" if cat in per_cat else "N/A"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", *cols, "overall_acc", "invalid_rate"])
        for model, per_cat in scores_by_model.items():
            o = _overall(per_cat)
            w.writerow(
                [
                    model,
                    *[acc(per_cat, c) for c in cols],
                    f"{o['accuracy'] * 100:.2f}",
                    f"{o['invalid_rate'] * 100:.2f}",
                ]
            )

    header = "| Model | " + " | ".join(cols) + " | Overall Acc | Invalid % |"
    sep = "|" + "---|" * (len(cols) + 3)
    rows = []
    for model, per_cat in scores_by_model.items():
        o = _overall(per_cat)
        cells = " | ".join(acc(per_cat, c) for c in cols)
        rows.append(
            f"| {model} | {cells} | {o['accuracy'] * 100:.2f} | {o['invalid_rate'] * 100:.2f} |"
        )
    md = "# BFCL baselines (non-live AST)\n\nAccuracy (%) by category; invalid % = "
    md += "share of unparseable outputs.\n\n" + header + "\n" + sep + "\n" + "\n".join(rows) + "\n"
    md_path.write_text(md, encoding="utf-8")

    return csv_path, md_path
