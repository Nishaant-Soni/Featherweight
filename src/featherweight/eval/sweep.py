"""Phase 5 lean hyperparameter sweep: the run matrix, winner selection, and table.

Pure + local (no GPU). The sweep targets the one Phase-4 weakness — BFCL irrelevance
over-calling — without regressing the other categories, so selection is **exact-match
with a refusal floor**: among runs whose held-out `refusal_accuracy` clears a floor (the
current tuned model's refusal), pick the highest `exact_match_accuracy`. That stops the
selector from re-picking an over-caller that scores well on tool calls but refuses poorly.

Held-out is the *fast* selector (in-distribution); the irrelevance fix is BFCL-confirmed
in Phase 5 Group C. See implementation_plan.md Phase 5.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from featherweight import config


@dataclass(frozen=True)
class SweepRun:
    """One sweep config: `None` means "use the locked default in `config`".

    `effective()` resolves the four levers against the defaults so both the results
    table (here) and the per-run config builders (Group B) read them the same way.
    """

    name: str
    irrelevance_ratio: float | None = None
    epochs: int | None = None
    rank: int | None = None
    learning_rate: float | None = None

    def effective(self) -> dict:
        c = config.CONFIG
        return {
            "irrelevance_ratio": self.irrelevance_ratio
            if self.irrelevance_ratio is not None
            else c.data.irrelevance_ratio,
            "epochs": self.epochs if self.epochs is not None else c.train.epochs,
            "rank": self.rank if self.rank is not None else c.train.lora.r,
            "learning_rate": self.learning_rate
            if self.learning_rate is not None
            else c.train.learning_rate,
        }


# First wave only. R0 is the current Phase-4 adapter (the fallback reference); R1/R2
# move the primary lever. Follow-ups (e.g. +1 epoch / rank 32 at the better ratio) are
# appended in Group C once R1/R2 are in — hardcoding their ratio now would be a guess.
SWEEP_RUNS: tuple[SweepRun, ...] = (
    SweepRun("r0-baseline"),  # 2 epochs, r=16, lr=2e-4, ratio=0.12 (current config)
    SweepRun("r1-irr0.20", irrelevance_ratio=0.20),
    SweepRun("r2-irr0.25", irrelevance_ratio=0.25),
)


def config_for(run: SweepRun) -> config.Config:
    """Build a per-run `Config` with the sweep overrides applied, via
    `dataclasses.replace` (the global `config.CONFIG` is frozen and left untouched).

    Sets the LoRA **rank** only; `lora_alpha` is left at the config default — for the
    optional rank-32 follow-up, whether to scale alpha with rank is a Group C decision
    (see implementation_plan.md / docs).
    """
    eff = run.effective()
    base = config.CONFIG
    return replace(
        base,
        data=replace(base.data, irrelevance_ratio=eff["irrelevance_ratio"]),
        train=replace(
            base.train,
            epochs=eff["epochs"],
            learning_rate=eff["learning_rate"],
            lora=replace(base.train.lora, r=eff["rank"]),
        ),
    )


def select_best(metrics_by_run: dict[str, dict], refusal_floor: float) -> str | None:
    """Run name with the highest held-out `exact_match_accuracy` among runs whose
    `refusal_accuracy` >= `refusal_floor`. Returns ``None`` if no run clears the floor
    (caller keeps the R0 baseline). Ties broken by higher refusal, then run name.

    `refusal_floor` is supplied by the caller (the current tuned model's measured
    held-out refusal), not hardcoded — see module docstring.
    """
    qualified = [
        name for name, m in metrics_by_run.items() if m["refusal_accuracy"] >= refusal_floor
    ]
    if not qualified:
        return None
    return max(
        qualified,
        key=lambda name: (
            metrics_by_run[name]["exact_match_accuracy"],
            metrics_by_run[name]["refusal_accuracy"],
            name,
        ),
    )


_COLS = ("irrelevance_ratio", "epochs", "rank", "learning_rate")
_METRICS = ("exact_match_accuracy", "tool_name_accuracy", "refusal_accuracy", "invalid_rate")


def write_sweep(
    runs: Sequence[SweepRun],
    metrics_by_run: dict[str, dict],
    winner: str | None,
    out_dir: Path = config.RESULTS_DIR,
) -> tuple[Path, Path]:
    """Write `sweep.csv` + `sweep.md` (config levers + held-out metrics per run, winner
    flagged). Only runs present in `metrics_by_run` are emitted. Returns the two paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in runs if r.name in metrics_by_run]
    header = ["run", *_COLS, *_METRICS, "winner"]

    def cells(r: SweepRun) -> list[str]:
        eff = r.effective()
        m = metrics_by_run[r.name]
        pct = [f"{m[k] * 100:.2f}" for k in _METRICS]
        return [
            r.name,
            f"{eff['irrelevance_ratio']:.2f}",
            str(eff["epochs"]),
            str(eff["rank"]),
            f"{eff['learning_rate']:.0e}",
            *pct,
            "✓" if r.name == winner else "",
        ]

    csv_path = out_dir / "sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(cells(r) for r in rows)

    md_path = out_dir / "sweep.md"
    sep = "|" + "---|" * len(header)
    lines = [
        "# Phase 5 sweep (held-out)",
        "",
        "Config levers + held-out metrics per run; ✓ = selected (exact-match with a "
        "refusal floor). BFCL-confirmed separately.",
        "",
        "| " + " | ".join(header) + " |",
        sep,
        *("| " + " | ".join(cells(r)) + " |" for r in rows),
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, md_path
