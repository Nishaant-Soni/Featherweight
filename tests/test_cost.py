"""Phase 6 Group A: serving cost/latency math + serving-table writer (pure, no GPU)."""

from featherweight.utils import cost


def test_latency_stats_nearest_rank():
    s = cost.latency_stats(list(range(1, 101)))  # 1..100
    assert s["p50"] == 50
    assert s["p95"] == 95
    assert s["p99"] == 99
    assert s["mean"] == 50.5


def test_latency_stats_empty():
    assert cost.latency_stats([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_throughput():
    assert cost.throughput(200, 100.0) == 2.0
    assert cost.throughput(5, 0) == 0.0  # guard against div-by-zero


def test_gpu_cost_and_cost_per_1k():
    assert cost.gpu_cost_usd(3600, hourly_rate=0.35) == 0.35  # one hour at the T4 rate
    # 0.35 of GPU time over 1000 calls -> $0.35 per 1k
    assert cost.cost_per_1k(0.35, 1000) == 0.35


def test_gpt4o_cost_per_1k_from_phase3():
    # $1.67 over 1240 calls -> ~$1.347 / 1k
    assert round(cost.gpt4o_cost_per_1k(), 4) == round(1.67 / 1240 * 1000, 4)


def test_gpt4o_serving_metrics_from_bfcl():
    m = cost.gpt4o_serving_metrics()
    assert m["p95_ms"] == 3070.0  # 3.07s measured by bfcl -> ms
    assert round(m["usd_per_1k"], 4) == round(1.67 / 1240 * 1000, 4)
    assert "p50_ms" not in m and "throughput_req_s" not in m  # not reported -> N/A in table


def test_write_serving_handles_missing_metrics(tmp_path):
    metrics = {
        "featherweight-ft-awq": {
            "throughput_req_s": 12.5,
            "p50_ms": 80.0,
            "p95_ms": 140.0,
            "p99_ms": 210.0,
            "usd_per_1k": 0.0078,
        },
        # GPT-4o: cost + p95 known (bfcl), p50/p99/throughput not -> N/A those columns
        "gpt-4o-2024-11-20-FC": cost.gpt4o_serving_metrics(),
    }
    csv_path, md_path = cost.write_serving(metrics, out_dir=tmp_path)
    md = md_path.read_text()
    assert "featherweight-ft-awq" in md and "gpt-4o-2024-11-20-FC" in md
    assert "N/A" in md  # GPT-4o p50/p99/throughput columns
    assert "3070.0" in md  # GPT-4o p95 latency now populated, not N/A
    assert "12.50" in md  # throughput formatted
    assert csv_path.exists()
