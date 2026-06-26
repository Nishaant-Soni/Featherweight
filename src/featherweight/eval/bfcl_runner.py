"""Build the bfcl-eval CLI commands for BFCL generation + scoring.

Pure command construction (returns argv lists) so it is unit-testable on the
CPU-only Mac without a GPU or the bfcl-eval install. Execution happens in Phase 3
Group C: the base open model on Colab, GPT-4o via API, scoring on CPU.

**T4 / quantization (pinned bfcl-eval 2026.3.23):** `bfcl generate`'s built-in
`vllm serve` launch hardcodes its args and exposes **no** `--quantization` /
`--max-model-len`, so an fp16 8B will not fit a 16 GB T4. We therefore launch our
**own** quantized vLLM server (`vllm_serve_cmd`) and point bfcl at it with
`--skip-server-setup` + the `LOCAL_SERVER_ENDPOINT` / `LOCAL_SERVER_PORT` env vars.
GPT-4o is an API model — no server, so `generate_cmd` is called without a backend.

The model *name* is a parameter on purpose: which BFCL model-registry name (and
therefore which handler) to use for the base baseline is a Group C decision — for
a fair base-vs-FT delta the base should run under the same SalesforceLlamaHandler
the FT model uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from featherweight import config


def generate_cmd(
    model: str,
    categories: Sequence[str],
    result_dir: Path | str,
    *,
    backend: str | None = None,
    skip_server_setup: bool = False,
    gpu_memory_utilization: float | None = None,
) -> list[str]:
    """`bfcl generate` argv. Local models pass ``backend``/``skip_server_setup``;
    API models (GPT-4o) pass neither (the default sglang backend is ignored)."""
    cmd = [
        "bfcl",
        "generate",
        "--model",
        model,
        "--test-category",
        ",".join(categories),
        "--result-dir",
        str(result_dir),
    ]
    if backend is not None:
        cmd += ["--backend", backend]
    if skip_server_setup:
        cmd += ["--skip-server-setup"]
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    return cmd


def evaluate_cmd(
    model: str,
    categories: Sequence[str],
    result_dir: Path | str,
    score_dir: Path | str,
) -> list[str]:
    """`bfcl evaluate` argv — scores the generated results into ``score_dir``."""
    return [
        "bfcl",
        "evaluate",
        "--model",
        model,
        "--test-category",
        ",".join(categories),
        "--result-dir",
        str(result_dir),
        "--score-dir",
        str(score_dir),
    ]


def vllm_serve_cmd(
    model_id: str,
    port: int,
    *,
    quantization: str = config.CONFIG.eval.base_quantization,
    max_model_len: int = config.CONFIG.eval.vllm_max_model_len,
    gpu_memory_utilization: float = config.CONFIG.eval.vllm_gpu_memory_utilization,
    dtype: str = "half",
    lora_modules: dict[str, str] | None = None,
    max_lora_rank: int = config.CONFIG.train.lora.r,
) -> list[str]:
    """`vllm serve` argv for the externally-launched quantized server (T4 path).

    bfcl's built-in launch can't pass these, so we run this ourselves and connect
    bfcl with `--skip-server-setup`. ``dtype=half`` (fp16) because the T4 (Turing,
    SM 75) has no bf16.

    For the Phase 4 FT eval, pass ``lora_modules={name: adapter_dir}`` to serve the
    base + a LoRA adapter (``--enable-lora``); ``name`` must match the FT registry
    ``model_name`` so requests route to the adapter. ``max_lora_rank`` defaults to the
    trained LoRA rank.
    """
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--port",
        str(port),
        "--quantization",
        quantization,
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--trust-remote-code",
    ]
    if lora_modules:
        cmd += [
            "--enable-lora",
            "--lora-modules",
            *[f"{name}={path}" for name, path in lora_modules.items()],
            "--max-lora-rank",
            str(max_lora_rank),
        ]
    return cmd
