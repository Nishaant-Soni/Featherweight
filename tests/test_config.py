"""Phase 0 smoke test: the package imports and config is internally consistent."""

from featherweight import config


def test_package_version():
    import featherweight

    assert featherweight.__version__ == "0.1.0"


def test_root_dir_is_repo_root():
    # config.py is at <root>/src/featherweight/config.py; ROOT_DIR must be <root>.
    assert (config.ROOT_DIR / "pyproject.toml").is_file()


def test_irrelevance_ratio_in_target_band():
    # PRD §4: blend ~10-15% irrelevance examples.
    assert 0.10 <= config.CONFIG.data.irrelevance_ratio <= 0.15


def test_lora_targets_attention_and_mlp():
    # PRD FR2: LoRA on attention + MLP projections.
    targets = set(config.CONFIG.train.lora.target_modules)
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= targets  # attention
    assert {"gate_proj", "up_proj", "down_proj"} <= targets  # MLP


def test_eval_base_precision_pinned():
    # PRD §9 fairness rule: base baseline + FT eval must share one base precision.
    # Pinned here so Phase 3 (base) and Phase 4 (FT) read the same value.
    assert config.CONFIG.eval.base_quantization == "bitsandbytes"
    assert config.CONFIG.eval.vllm_backend == "vllm"  # not sglang on Turing
