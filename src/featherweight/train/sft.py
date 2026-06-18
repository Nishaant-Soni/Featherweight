"""QLoRA SFT entrypoint (Unsloth).

Two halves, split by where they run:
- ``load_sft_dataset`` shapes ``train.jsonl`` into the trainer's ``text`` field
  (prompt + completion). Pure, local, unit-tested.
- ``train`` runs the Unsloth QLoRA loop. **Colab-only** — its GPU deps
  (``unsloth``, ``trl``) are imported lazily inside the function so this module
  imports fine on the CPU-only Mac. The exact trainer API is validated on Colab
  (Phase 2 Group C); the structure follows Unsloth's standard Llama-3.1 recipe.

Completion-only loss masking uses ``train_on_responses_only`` keyed on the
Llama-3.1 turn headers, which must match ``data/format.build_prompt``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset

from featherweight import config
from featherweight.utils import tracking

# Must match the headers emitted by data/format.build_prompt.
INSTRUCTION_MARKER = "<|start_header_id|>user<|end_header_id|>\n\n"
RESPONSE_MARKER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

DEFAULT_TRAIN_PATH = config.DATA_DIR / "train.jsonl"
DEFAULT_HELDOUT_PATH = config.DATA_DIR / "heldout.jsonl"
DEFAULT_OUTPUT_DIR = config.ARTIFACTS_DIR / "adapter"


def load_sft_dataset(path: Path = DEFAULT_TRAIN_PATH) -> Dataset:
    """Load the prep'd JSONL into a single ``text`` column (prompt + completion).

    ``train_on_responses_only`` later masks everything up to ``RESPONSE_MARKER``,
    so the loss is computed only on the gold JSON-array completion.
    """
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    texts = [row["prompt"] + row["completion"] for row in rows]
    return Dataset.from_dict({"text": texts})


def load_eval_subset(
    path: Path = DEFAULT_HELDOUT_PATH,
    size: int = 200,
    seed: int = 42,
) -> Dataset:
    """A small, **mixed** held-out eval split (both tool-call and irrelevance rows)
    in `text` form, drawn from heldout.jsonl in its natural ~irrelevance proportion
    (>=1 irrelevance row guaranteed) so eval_loss reflects calling *and* refusal.
    """
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    irr = [r for r in rows if r["completion"] == "[]<|eot_id|>"]
    calls = [r for r in rows if r["completion"] != "[]<|eot_id|>"]
    rng = random.Random(seed)
    n_irr = min(len(irr), max(1, round(size * len(irr) / len(rows))))
    n_call = min(len(calls), size - n_irr)
    subset = rng.sample(calls, n_call) + rng.sample(irr, n_irr)
    rng.shuffle(subset)
    return Dataset.from_dict({"text": [r["prompt"] + r["completion"] for r in subset]})


def _log_params(cfg: config.Config) -> dict:
    t = cfg.train
    return {
        "base_model": config.BASE_MODEL_4BIT,
        "lora_r": t.lora.r,
        "lora_alpha": t.lora.alpha,
        "max_seq_len": t.max_seq_len,
        "epochs": t.epochs,
        "learning_rate": t.learning_rate,
        "batch_size": t.per_device_batch_size,
        "grad_accum": t.grad_accumulation_steps,
        "irrelevance_ratio": cfg.data.irrelevance_ratio,
    }


def train(
    train_path: Path = DEFAULT_TRAIN_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cfg: config.Config = config.CONFIG,
    max_steps: int | None = None,
    limit: int | None = None,
    early_stopping: bool = True,
) -> Path:
    """Run QLoRA SFT on the T4 and save the LoRA adapter. Colab-only.

    Precision is chosen at runtime: bf16 only if the GPU supports it (Ampere+),
    else fp16 — the T4 (Turing, SM 75) does not support bf16, so `cfg.train.bf16`
    is treated as a preference, not a hard setting.

    ``max_steps`` is the hard ceiling (use it to fit a Colab session). With
    ``early_stopping`` (default on), training also evaluates eval_loss on a small
    mixed held-out subset every ``cfg.train.eval_steps`` and stops early once it
    plateaus, keeping the best checkpoint. ``limit`` / ``early_stopping=False`` are
    for the smoke run (a few steps on a tiny slice to sanity-check the pipeline).
    """
    # Import unsloth first (its recommended order; it patches transformers/trl).
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    use_bf16 = cfg.train.bf16 and is_bfloat16_supported()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL_4BIT,
        max_seq_length=cfg.train.max_seq_len,
        dtype=None,  # auto: bf16 on Ampere+, fp16 on the T4
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.train.lora.r,
        lora_alpha=cfg.train.lora.alpha,
        lora_dropout=cfg.train.lora.dropout,
        target_modules=list(cfg.train.lora.target_modules),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.data.seed,
    )

    dataset = load_sft_dataset(train_path)
    if limit is not None:
        dataset = dataset.select(range(limit))

    sft_kwargs = {
        "dataset_text_field": "text",
        "max_length": cfg.train.max_seq_len,  # TRL 0.24 name (was max_seq_length)
        "per_device_train_batch_size": cfg.train.per_device_batch_size,
        "gradient_accumulation_steps": cfg.train.grad_accumulation_steps,
        "num_train_epochs": cfg.train.epochs,
        "max_steps": max_steps if max_steps is not None else -1,  # -1 = use epochs
        "learning_rate": cfg.train.learning_rate,
        "warmup_ratio": cfg.train.warmup_ratio,
        "weight_decay": cfg.train.weight_decay,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "logging_steps": 10,
        "seed": cfg.data.seed,
        "output_dir": str(output_dir),
        "report_to": "mlflow",  # logs loss into the active mlflow run below
    }

    eval_dataset = None
    callbacks: list = []
    if early_stopping:
        # Eval-loss early stopping on a small mixed held-out subset, under the
        # max_steps ceiling — stops when eval_loss plateaus (overfit/time guard)
        # and keeps the best checkpoint.
        eval_dataset = load_eval_subset(size=cfg.train.eval_subset_size, seed=cfg.data.seed)
        sft_kwargs.update(
            eval_strategy="steps",
            eval_steps=cfg.train.eval_steps,
            per_device_eval_batch_size=cfg.train.per_device_batch_size,
            save_strategy="steps",
            save_steps=cfg.train.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.train.early_stopping_patience,
                early_stopping_threshold=cfg.train.early_stopping_threshold,
            )
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # TRL 0.24 renamed `tokenizer` -> `processing_class`
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**sft_kwargs),
        callbacks=callbacks,
    )
    # Completion-only loss: mask the prompt, train on the assistant turn only.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_MARKER,
        response_part=RESPONSE_MARKER,
    )

    with tracking.mlflow_run(run_name="qlora-sft", params=_log_params(cfg)):
        trainer.train()
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    return output_dir


if __name__ == "__main__":
    train()
