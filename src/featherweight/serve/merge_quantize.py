"""Merge the LoRA adapter into the base and export a 4-bit AWQ serving model (Phase 6 FR4).

Two halves, split by where they run (same pattern as ``train/sft.py``):
- ``output_paths`` / ``awq_quant_config`` are pure config-driven helpers, unit-tested locally.
- ``merge_quantize`` runs the GPU work and is **Colab-only** — its deps (``unsloth``,
  ``awq``) are imported lazily inside the function so this module imports on the CPU Mac.

The adapter was trained on the **bnb-4bit** base, so merging is lossy unless it goes through
fp16: we ``save_pretrained_merged(save_method="merged_16bit")`` first, then AWQ-quantize the
fp16 merge. (The merged-16bit dir is kept as a standalone artifact — a GPTQ fallback would
reuse it.) AWQ calibration on a 16 GB T4 is tight; that's a Group C runtime concern.
"""

from __future__ import annotations

from pathlib import Path

from featherweight import config


def output_paths(output_dir: Path, cfg: config.Config = config.CONFIG) -> tuple[Path, Path]:
    """``(merged_16bit_dir, quantized_dir)`` under ``output_dir`` (pure)."""
    return (
        output_dir / cfg.serve.merged_16bit_subdir,
        output_dir / cfg.serve.quantized_subdir,
    )


def awq_quant_config(cfg: config.Config = config.CONFIG) -> dict:
    """The autoawq ``quant_config`` from `ServeConfig` (pure)."""
    return {
        "zero_point": True,
        "q_group_size": cfg.serve.awq_group_size,
        "w_bit": cfg.serve.awq_bits,
        "version": "GEMM",
    }


def merge_quantize(
    adapter_id: str,
    output_dir: Path,
    calib_texts: list[str],
    cfg: config.Config = config.CONFIG,
) -> Path:
    """Merge ``adapter_id`` (base+LoRA) to fp16, then AWQ-quantize. Returns the quantized
    dir. Colab-only (lazy GPU imports). ``calib_texts`` are the AWQ calibration prompts."""
    if cfg.serve.quantization != "awq":
        raise NotImplementedError(
            f"merge_quantize implements AWQ; got {cfg.serve.quantization!r}. "
            "GPTQ/bnb are Colab-time fallbacks (see implementation_plan.md Phase 6)."
        )

    merged_dir, quantized_dir = output_paths(output_dir, cfg)

    # 1) Merge adapter -> fp16 standalone model (lossless vs the 4-bit-base merge).
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_id,
        max_seq_length=cfg.train.max_seq_len,
        dtype=None,
        load_in_4bit=False,  # 16-bit load for a clean merge
    )
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    # 2) Quantize the fp16 merge to 4-bit AWQ.
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    awq_model = AutoAWQForCausalLM.from_pretrained(str(merged_dir))
    tok = AutoTokenizer.from_pretrained(str(merged_dir))
    awq_model.quantize(tok, quant_config=awq_quant_config(cfg), calib_data=calib_texts)
    awq_model.save_quantized(str(quantized_dir))
    tok.save_pretrained(str(quantized_dir))

    return quantized_dir
