"""Phase 5 Group A: sweep run spec, winner selection, results table (pure, no GPU)."""

from featherweight import config
from featherweight.eval import sweep


def _m(exact: float, refusal: float, name_acc: float = 0.95, invalid: float = 0.005) -> dict:
    return {
        "exact_match_accuracy": exact,
        "refusal_accuracy": refusal,
        "tool_name_accuracy": name_acc,
        "invalid_rate": invalid,
    }


def test_effective_resolves_none_to_config_defaults():
    eff = sweep.SweepRun("r0-baseline").effective()
    assert eff["irrelevance_ratio"] == config.CONFIG.data.irrelevance_ratio
    assert eff["epochs"] == config.CONFIG.train.epochs
    assert eff["rank"] == config.CONFIG.train.lora.r
    assert eff["learning_rate"] == config.CONFIG.train.learning_rate
    # an override wins over the default
    assert sweep.SweepRun("x", irrelevance_ratio=0.20).effective()["irrelevance_ratio"] == 0.20


def test_select_best_picks_highest_exact_match_above_floor():
    metrics = {
        "r0-baseline": _m(exact=0.80, refusal=0.89),
        "r1-irr0.20": _m(exact=0.83, refusal=0.91),
        "r2-irr0.25": _m(exact=0.81, refusal=0.93),
    }
    # all clear the 0.89 floor -> highest exact-match wins.
    assert sweep.select_best(metrics, refusal_floor=0.89) == "r1-irr0.20"


def test_select_best_rejects_overcaller_below_floor():
    # r1 has the BEST exact-match but refuses poorly (over-caller) -> must NOT win;
    # the best run that still clears the floor does.
    metrics = {
        "r0-baseline": _m(exact=0.80, refusal=0.89),
        "r1-irr0.20": _m(exact=0.88, refusal=0.70),  # over-caller
        "r2-irr0.25": _m(exact=0.82, refusal=0.90),
    }
    assert sweep.select_best(metrics, refusal_floor=0.89) == "r2-irr0.25"


def test_select_best_none_when_no_run_meets_floor():
    metrics = {
        "r1-irr0.20": _m(exact=0.88, refusal=0.70),
        "r2-irr0.25": _m(exact=0.85, refusal=0.80),
    }
    assert sweep.select_best(metrics, refusal_floor=0.89) is None


def test_write_sweep_emits_only_scored_runs_and_marks_winner(tmp_path):
    runs = sweep.SWEEP_RUNS
    metrics = {
        "r0-baseline": _m(exact=0.80, refusal=0.89),
        "r1-irr0.20": _m(exact=0.83, refusal=0.91),
        # r2 not scored yet -> must be omitted from the table
    }
    winner = sweep.select_best(metrics, refusal_floor=0.89)
    csv_path, md_path = sweep.write_sweep(runs, metrics, winner, out_dir=tmp_path)

    md = md_path.read_text()
    assert "r0-baseline" in md and "r1-irr0.20" in md
    assert "r2-irr0.25" not in md  # unscored run omitted
    # winner row carries the check mark
    assert any(line.startswith("| r1-irr0.20") and "✓" in line for line in md.splitlines())
    assert csv_path.exists()
