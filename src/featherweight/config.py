"""Single source of truth for model ids, dataset ids, hyperparameters, and paths.

Every module (data prep, training, eval, serving) imports from here so that a
change to a hyperparameter or path happens in exactly one place. Values match
the choices locked in PRD.md §4–§7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all derived from the repo root so they work locally and on Colab)
# --------------------------------------------------------------------------- #
# config.py lives at <root>/src/featherweight/config.py -> parents[2] is <root>.
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"  # git-ignored: cached + formatted data
ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"  # git-ignored: adapters, merged models
RESULTS_DIR: Path = ROOT_DIR / "results"  # committed: eval tables, plots, logs
THIRD_PARTY_DIR: Path = ROOT_DIR / "third_party"  # git-ignored: cloned gorilla/BFCL


# --------------------------------------------------------------------------- #
# Model ids (PRD §5 — base model LOCKED)
# --------------------------------------------------------------------------- #
# 4-bit Unsloth variant used for QLoRA training on the Colab T4.
BASE_MODEL_4BIT: str = "unsloth/llama-3.1-8b-Instruct-bnb-4bit"
# Full-precision instruct model id (BFCL base-model baseline / chat template source).
BASE_MODEL_FULL: str = "meta-llama/Llama-3.1-8B-Instruct"
# Frontier baseline run through the BFCL harness (PRD §3).
GPT4O_MODEL: str = "gpt-4o"


# --------------------------------------------------------------------------- #
# Dataset ids (PRD §4 — LOCKED)
# --------------------------------------------------------------------------- #
TRAIN_DATASET: str = "minpeter/xlam-function-calling-60k-parsed"
IRRELEVANCE_DATASET: str = "MadeAgents/xlam-irrelevance-7.5k"


@dataclass(frozen=True)
class DataConfig:
    """Data prep + mixing (PRD §4, FR1)."""

    irrelevance_ratio: float = 0.12  # ~10-15% of the training mix
    heldout_size: int = 1000  # fast internal eval split (PRD §4)
    seed: int = 42  # deterministic split + shuffling


@dataclass(frozen=True)
class LoraConfig:
    """QLoRA adapter config (PRD FR2)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    # Attention + MLP projections (PRD FR2).
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TrainConfig:
    """SFT hyperparameters (PRD FR2). Tuned further in Phase 5."""

    max_seq_len: int = 2048
    epochs: int = 2
    learning_rate: float = 2e-4
    per_device_batch_size: int = 2
    grad_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    bf16: bool = True
    # Eval-loss early stopping (capped by max_steps). Patience-based: stop when
    # eval_loss hasn't improved by > threshold for `patience` consecutive evals.
    eval_subset_size: int = 200
    eval_steps: int = 50
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass(frozen=True)
class EvalConfig:
    """BFCL generation settings (PRD FR3, Phase 3/4).

    The 8B base is loaded **quantized** for vLLM generation — an fp16 8B does not
    fit a 16GB T4, and the base is the heavy part in *both* the base baseline and
    the FT eval (FT = base + LoRA). The **same** precision is used for both so
    base-vs-FT stays apples-to-apples (PRD §9 fairness rule). bnb 4-bit matches the
    precision the adapter was trained on. T4 = Turing (SM 75) -> vLLM, not sglang.
    """

    base_quantization: str = "bitsandbytes"  # bnb 4-bit; matches the trained base
    vllm_backend: str = "vllm"  # not sglang (unsupported on Turing/SM 75)
    vllm_max_model_len: int = 4096  # cap KV cache to fit the T4
    vllm_gpu_memory_utilization: float = 0.90
    # Non-live AST categories we baseline on (Python-relevant + irrelevance). BFCL's
    # `simple` splits by language; our model emits Python-style JSON calls, so we
    # take simple_python (not java/javascript). Same set for base and FT (PRD FR3).
    categories: tuple[str, ...] = (
        "simple_python",
        "multiple",
        "parallel",
        "parallel_multiple",
        "irrelevance",
    )


@dataclass(frozen=True)
class Config:
    """Top-level config aggregating the sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    mlflow_experiment: str = "featherweight"


CONFIG = Config()
