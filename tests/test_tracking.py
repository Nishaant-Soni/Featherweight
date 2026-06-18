"""Group B (Phase 2) test: the MLflow wrapper logs params + metrics to a run.

Uses a sqlite backend: MLflow 3.x has put the file:// store in maintenance mode
and raises without it (see docs/iteration_6.md — also a Colab/Group C note).
"""

import mlflow

from featherweight import config
from featherweight.utils import tracking


def test_mlflow_run_logs_params_and_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    with tracking.mlflow_run("test-run", params={"lora_r": 16}):
        tracking.log_metrics({"exact_match_accuracy": 0.5, "n": 10})

    runs = mlflow.search_runs(
        experiment_names=[config.CONFIG.mlflow_experiment], output_format="list"
    )
    assert len(runs) == 1
    assert runs[0].data.params["lora_r"] == "16"
    assert runs[0].data.metrics["exact_match_accuracy"] == 0.5
