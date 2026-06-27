"""Phase 5 Group B: prep_data passes the irrelevance_ratio override through.

The ratio *math* is covered by test_mix; here we only check the plumbing — that
`main(irrelevance_ratio=...)` reaches `mix.mix_and_split`, and `None` falls back to the
locked config default. `prep_data` is a script (not on the package path), so it's loaded
by file path.
"""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "prep_data", Path(__file__).resolve().parents[1] / "scripts" / "prep_data.py"
)
prep_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep_data)


def _stub_loads(monkeypatch, captured):
    monkeypatch.setattr(prep_data.load, "load_xlam", lambda: [])
    monkeypatch.setattr(prep_data.load, "load_irrelevance", lambda: [])

    def fake_mix(xlam, irr, *, ratio, heldout_size, seed):
        captured["ratio"] = ratio
        # non-empty so main's summary math (n_irr / total) doesn't divide by zero
        return [{"prompt": "p", "completion": "c"}], [{"prompt": "p", "completion": "[]<|eot_id|>"}]

    monkeypatch.setattr(prep_data.mix, "mix_and_split", fake_mix)


def test_main_passes_ratio_override(tmp_path, monkeypatch):
    captured = {}
    _stub_loads(monkeypatch, captured)
    prep_data.main(irrelevance_ratio=0.20, out_dir=tmp_path)
    assert captured["ratio"] == 0.20
    assert (tmp_path / "train.jsonl").exists() and (tmp_path / "heldout.jsonl").exists()


def test_main_defaults_to_config_ratio(tmp_path, monkeypatch):
    captured = {}
    _stub_loads(monkeypatch, captured)
    prep_data.main(out_dir=tmp_path)
    assert captured["ratio"] == prep_data.config.CONFIG.data.irrelevance_ratio
