"""Phase 3 Group B: parsing bfcl-eval score files into the baselines table."""

import json

from featherweight.eval import report


def _write_score_file(path, accuracy, correct, total, error_types):
    """Mimic a bfcl `*_score.json`: line 1 summary header, lines 2+ failed entries."""
    lines = [{"accuracy": accuracy, "correct_count": correct, "total_count": total}]
    lines += [{"error_type": et} for et in error_types]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")


def test_category_from_filename():
    assert (
        report.category_from_filename("BFCL_v4_parallel_multiple_score.json") == "parallel_multiple"
    )
    assert report.category_from_filename("BFCL_v4_irrelevance_score.json") == "irrelevance"


def test_parse_score_file_accuracy_and_invalid_rate(tmp_path):
    p = tmp_path / "BFCL_v4_simple_python_score.json"
    # 10 total, 8 correct; of the 2 failures one is unparseable (decoder_failed).
    _write_score_file(p, 0.8, 8, 10, ["ast_decoder:decoder_failed", "ast_checker:value_error"])
    out = report.parse_score_file(p)
    assert out["accuracy"] == 0.8
    assert out["total_count"] == 10
    assert out["invalid_rate"] == 0.1  # 1 decoder_failed / 10


def test_collect_scores_keys_by_category(tmp_path):
    _write_score_file(tmp_path / "BFCL_v4_simple_python_score.json", 0.9, 9, 10, [])
    _write_score_file(
        tmp_path / "BFCL_v4_multiple_score.json", 0.5, 5, 10, ["ast_decoder:decoder_failed"]
    )
    scores = report.collect_scores(tmp_path)
    assert set(scores) == {"simple_python", "multiple"}
    assert scores["multiple"]["invalid_rate"] == 0.1


def test_write_baselines_csv_and_md(tmp_path):
    scores_by_model = {
        "base": {
            "simple_python": {
                "accuracy": 0.30,
                "correct_count": 3,
                "total_count": 10,
                "invalid_rate": 0.2,
            },
            "multiple": {
                "accuracy": 0.50,
                "correct_count": 5,
                "total_count": 10,
                "invalid_rate": 0.1,
            },
        },
        "gpt-4o": {
            "simple_python": {
                "accuracy": 0.90,
                "correct_count": 9,
                "total_count": 10,
                "invalid_rate": 0.0,
            },
            "multiple": {
                "accuracy": 0.80,
                "correct_count": 8,
                "total_count": 10,
                "invalid_rate": 0.0,
            },
        },
    }
    csv_path, md_path = report.write_baselines(
        scores_by_model, out_dir=tmp_path, categories=("simple_python", "multiple")
    )
    assert csv_path.exists() and md_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "model,simple_python,multiple,overall_acc,invalid_rate" in csv_text
    assert "base,30.00,50.00,40.00,15.00" in csv_text  # overall = 8/20 acc, 3/20 invalid
    md_text = md_path.read_text(encoding="utf-8")
    assert "| base |" in md_text and "| gpt-4o |" in md_text
