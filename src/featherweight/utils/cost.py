"""Serving cost + latency math and the serving-results table (Phase 6 Group A).

Pure + local (no GPU). The Colab serve notebook (Group C) times the quantized model's
batch inference and feeds the raw numbers here to produce `results/serving.md`, putting the
local fine-tuned model's throughput / latency / $-per-1k next to the GPT-4o API baseline
measured in Phase 3.

Latency inputs are unit-agnostic; the table labels assume **milliseconds** (the notebook
passes per-request ms). The `$/1k` basis is documented in the constants below.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

from featherweight import config

# GPT-4o basis, read from bfcl's own accounting (third_party/bfcl/score/data_overall.csv,
# Phase 3 non-live run): total cost $1.67 over 1240 requests, plus measured API latency.
# "call" here = one request / test case (consistent with how the local model is timed); this
# differs from bfcl's public "per 1000 function calls" normalization, which we don't use.
GPT4O_RUN_USD = 1.67
GPT4O_RUN_CALLS = 1240
GPT4O_LATENCY_MEAN_S = 1.15
GPT4O_LATENCY_P95_S = 3.07

# Local cost basis: a published on-demand T4 rate. Colab Free is $0, so we price the GPU
# wall-time against comparable cloud hardware to make the comparison honest, not flattering.
DEFAULT_T4_HOURLY_USD = 0.35


def latency_stats(latencies: Sequence[float]) -> dict:
    """Mean + p50/p95/p99 (nearest-rank) of per-request latencies (caller's unit)."""
    vals = sorted(latencies)
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def pct(p: float) -> float:
        k = min(max(math.ceil(p / 100 * len(vals)) - 1, 0), len(vals) - 1)
        return vals[k]

    return {"mean": sum(vals) / len(vals), "p50": pct(50), "p95": pct(95), "p99": pct(99)}


def throughput(n_requests: int, wall_seconds: float) -> float:
    """Requests per second over the batch wall-time."""
    return n_requests / wall_seconds if wall_seconds else 0.0


def gpu_cost_usd(wall_seconds: float, hourly_rate: float = DEFAULT_T4_HOURLY_USD) -> float:
    """Dollar cost of ``wall_seconds`` of GPU time at ``hourly_rate``."""
    return wall_seconds / 3600 * hourly_rate


def cost_per_1k(total_usd: float, n_requests: int) -> float:
    """Dollars per 1,000 calls, given a total cost over ``n_requests``."""
    return total_usd / n_requests * 1000 if n_requests else 0.0


def gpt4o_cost_per_1k() -> float:
    """GPT-4o $/1k from the Phase 3 measured run cost (the comparison baseline)."""
    return cost_per_1k(GPT4O_RUN_USD, GPT4O_RUN_CALLS)


def gpt4o_serving_metrics() -> dict:
    """The GPT-4o comparison row for `write_serving`, from bfcl's measured numbers: $/1k +
    p95 latency (ms). p50/p99/throughput aren't reported by bfcl, so they fall to ``N/A``."""
    return {
        "p95_ms": GPT4O_LATENCY_P95_S * 1000,
        "usd_per_1k": gpt4o_cost_per_1k(),
    }


_COLS = ("throughput_req_s", "p50_ms", "p95_ms", "p99_ms", "usd_per_1k")


def write_serving(
    metrics_by_model: dict[str, dict],
    out_dir: Path = config.RESULTS_DIR,
) -> tuple[Path, Path]:
    """Write `serving.csv` + `serving.md` (throughput / p50-p95-p99 latency / $-per-1k per
    model; ``N/A`` where a metric wasn't measured, e.g. GPT-4o API latency). Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["model", *_COLS]

    def cell(m: dict, k: str) -> str:
        if k not in m:
            return "N/A"
        v = m[k]
        if k == "usd_per_1k":
            return f"{v:.4f}"
        if k == "throughput_req_s":
            return f"{v:.2f}"
        return f"{v:.1f}"  # latency columns, in ms

    def row(name: str, m: dict) -> list[str]:
        return [name, *(cell(m, k) for k in _COLS)]

    csv_path = out_dir / "serving.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(row(name, m) for name, m in metrics_by_model.items())

    sep = "|" + "---|" * len(header)
    lines = [
        "# Serving cost + latency (local FT vs GPT-4o)",
        "",
        "Latency in ms; `$/1k` = dollars per 1,000 requests (test cases). Local model priced at "
        f"${DEFAULT_T4_HOURLY_USD}/hr T4 (basis); free-Colab-T4 latency is shared/throttled "
        "hardware — illustrative, not a production SLA.",
        "",
        "| " + " | ".join(header) + " |",
        sep,
        *("| " + " | ".join(row(name, m)) + " |" for name, m in metrics_by_model.items()),
    ]
    md_path = out_dir / "serving.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, md_path
