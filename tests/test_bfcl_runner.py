"""Phase 3 Group B: bfcl-eval CLI command construction (pure, no GPU)."""

from featherweight import config
from featherweight.eval import bfcl_runner as r


def test_generate_cmd_api_model_has_no_backend():
    # GPT-4o is an API model: no --backend / --skip-server-setup.
    cmd = r.generate_cmd("gpt-4o", ["simple_python", "irrelevance"], "/res")
    assert cmd[:2] == ["bfcl", "generate"]
    assert "--model" in cmd and "gpt-4o" in cmd
    assert "--test-category" in cmd
    assert cmd[cmd.index("--test-category") + 1] == "simple_python,irrelevance"  # comma-joined
    assert "--result-dir" in cmd and "/res" in cmd
    assert "--backend" not in cmd
    assert "--skip-server-setup" not in cmd


def test_generate_cmd_local_model_external_server():
    cmd = r.generate_cmd(
        "meta-llama/Llama-3.1-8B-Instruct",
        config.CONFIG.eval.categories,
        "/res",
        backend="vllm",
        skip_server_setup=True,
    )
    assert cmd[cmd.index("--backend") + 1] == "vllm"
    assert "--skip-server-setup" in cmd


def test_evaluate_cmd():
    cmd = r.evaluate_cmd("gpt-4o", ["multiple"], "/res", "/score")
    assert cmd[:2] == ["bfcl", "evaluate"]
    assert cmd[cmd.index("--result-dir") + 1] == "/res"
    assert cmd[cmd.index("--score-dir") + 1] == "/score"


def test_vllm_serve_cmd_uses_config_precision_and_fp16():
    cmd = r.vllm_serve_cmd("unsloth/llama-3.1-8b-Instruct-bnb-4bit", 8000)
    assert cmd[:2] == ["vllm", "serve"]
    # base quantization + max-model-len come from EvalConfig (fairness lever).
    assert cmd[cmd.index("--quantization") + 1] == config.CONFIG.eval.base_quantization
    assert cmd[cmd.index("--max-model-len") + 1] == str(config.CONFIG.eval.vllm_max_model_len)
    assert cmd[cmd.index("--dtype") + 1] == "half"  # T4 (Turing) has no bf16
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert "--enable-lora" not in cmd  # base baseline serves no adapter


def test_vllm_serve_cmd_with_lora_adapter():
    cmd = r.vllm_serve_cmd(
        "unsloth/llama-3.1-8b-Instruct-bnb-4bit",
        8000,
        lora_modules={"featherweight-ft": "/content/adapter"},
    )
    assert "--enable-lora" in cmd
    # the module name == the FT registry model_name, so requests route to the adapter.
    assert cmd[cmd.index("--lora-modules") + 1] == "featherweight-ft=/content/adapter"
    # rank defaults to the trained LoRA r (single source of truth in config).
    assert cmd[cmd.index("--max-lora-rank") + 1] == str(config.CONFIG.train.lora.r)
