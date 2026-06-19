"""Phase 3 Group C: the base-model bfcl registry entry uses the prompting path.

Skipped where bfcl-eval isn't installed (the local CPU venv); runs in the serve
env. The contract that matters: the entry must dispatch to bfcl's *prompting* path
(`is_fc_model=False` and no "FC" in the name) so the base runs under the same
SalesforceLlamaHandler format our model was trained on — otherwise base-vs-FT
isn't apples-to-apples.
"""

import pytest

from featherweight.eval import bfcl_register


def test_base_registration_uses_prompting_path():
    pytest.importorskip("bfcl_eval")
    import bfcl_eval.constants.model_config as mc  # type: ignore[import-not-found]

    name = bfcl_register.register_base_model()
    cfg = mc.MODEL_CONFIG_MAPPING[name]
    assert cfg.is_fc_model is False  # prompting path, not FC
    assert "FC" not in name  # the other half of the dispatch condition
    assert cfg.model_handler.__name__ == "SalesforceLlamaHandler"
    # request `model` + tokenizer resolve to the real base our vLLM serves
    assert cfg.model_name == "unsloth/llama-3.1-8b-Instruct-bnb-4bit"
