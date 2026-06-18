"""Thin MLflow wrapper: start a run under the project experiment and log to it.

Used by train/sft.py (params + trainer loss via report_to="mlflow") and the
Group C held-out callback (the scorer metrics). Kept minimal on purpose.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import mlflow

from featherweight import config


@contextmanager
def mlflow_run(run_name: str | None = None, params: dict | None = None) -> Iterator[None]:
    """Open an MLflow run under the project experiment, logging ``params`` once."""
    mlflow.set_experiment(config.CONFIG.mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        yield


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log the numeric entries of a metrics dict (e.g. the held-out score)."""
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    mlflow.log_metrics(numeric, step=step)
