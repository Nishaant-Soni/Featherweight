"""Phase 6 Group B: merge+quantize config plumbing (pure, no GPU).

The full merge/quantize runs on Colab; here we only test the config-driven surface
(`output_paths`, `awq_quant_config`) and the early guard against unsupported methods.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from featherweight import config
from featherweight.serve import merge_quantize as mq


def test_output_paths_from_config():
    merged, quantized = mq.output_paths(Path("/out"))
    assert merged == Path("/out") / config.CONFIG.serve.merged_16bit_subdir
    assert quantized == Path("/out") / config.CONFIG.serve.quantized_subdir


def test_awq_quant_config_reads_serve_config():
    qc = mq.awq_quant_config()
    assert qc["w_bit"] == config.CONFIG.serve.awq_bits
    assert qc["q_group_size"] == config.CONFIG.serve.awq_group_size
    assert qc["version"] == "GEMM" and qc["zero_point"] is True


def test_serve_config_defaults():
    s = config.CONFIG.serve
    assert s.quantization == "awq" and s.awq_bits == 4 and s.awq_group_size == 128


def test_merge_quantize_rejects_non_awq_before_gpu_imports():
    # A GPTQ/bnb fallback is a Colab-time decision, not implemented here -> fail fast,
    # before any unsloth/awq import (so this runs GPU-free).
    gptq_cfg = replace(config.CONFIG, serve=replace(config.CONFIG.serve, quantization="gptq"))
    with pytest.raises(NotImplementedError):
        mq.merge_quantize("N-S-10/featherweight-adapter", Path("/tmp/x"), [], cfg=gptq_cfg)
