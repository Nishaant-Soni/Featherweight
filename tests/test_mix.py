"""Group 4 tests: irrelevance ratio math + deterministic train/held-out split."""

from featherweight.data import mix


def _examples(prefix: str, n: int) -> list[dict]:
    return [{"prompt": f"{prefix}-{i}", "completion": "[]"} for i in range(n)]


def test_irrelevance_count_caps_at_available():
    # Locked sizes: target ~8182 exceeds the 7500 available -> capped, no xLAM dropped.
    assert mix.irrelevance_count(60000, 7500, 0.12) == 7500


def test_irrelevance_count_subsamples_when_plenty():
    # ratio 0.2 over 100 xLAM needs 25; 50 available -> 25.
    assert mix.irrelevance_count(100, 50, 0.2) == 25


def test_irrelevance_count_zero_ratio():
    assert mix.irrelevance_count(100, 50, 0.0) == 0


def test_realized_ratio_within_band_for_locked_sizes():
    n_irr = mix.irrelevance_count(60000, 7500, 0.12)
    share = n_irr / (60000 + n_irr)
    assert 0.10 <= share <= 0.15  # ~0.111


def test_mix_and_split_sizes_disjoint_and_covering():
    xlam, irr = _examples("x", 80), _examples("i", 40)
    # ratio 0.2 over 80 -> 20 irr selected; total 100; heldout 10; train 90.
    train, heldout = mix.mix_and_split(xlam, irr, ratio=0.2, heldout_size=10, seed=42)
    assert len(heldout) == 10
    assert len(train) == 90
    train_ids = {e["prompt"] for e in train}
    heldout_ids = {e["prompt"] for e in heldout}
    assert train_ids.isdisjoint(heldout_ids)
    assert len(train_ids | heldout_ids) == 100


def test_mix_selects_capped_irrelevance():
    xlam, irr = _examples("x", 100), _examples("i", 10)
    train, heldout = mix.mix_and_split(xlam, irr, ratio=0.5, heldout_size=5, seed=1)
    allex = train + heldout
    assert sum(1 for e in allex if e["prompt"].startswith("i")) == 10  # only 10 available
    assert len(allex) == 110


def test_mix_and_split_is_deterministic():
    xlam, irr = _examples("x", 80), _examples("i", 40)
    a_train, a_held = mix.mix_and_split(xlam, irr, ratio=0.2, heldout_size=10, seed=42)
    b_train, b_held = mix.mix_and_split(xlam, irr, ratio=0.2, heldout_size=10, seed=42)
    assert [e["prompt"] for e in a_train] == [e["prompt"] for e in b_train]
    assert [e["prompt"] for e in a_held] == [e["prompt"] for e in b_held]
