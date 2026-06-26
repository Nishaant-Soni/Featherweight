"""Register Featherweight's models in bfcl-eval's model registry.

bfcl-eval resolves a model name to a handler via a hardcoded dict
(`MODEL_CONFIG_MAPPING`), with no runtime plug-in hook. To evaluate our model we
inject a `ModelConfig` into that dict **in-process**, then run bfcl's own CLI in
the same process (see `scripts/run_bfcl.py`). No site-packages edits.

Why these exact settings (verified against pinned bfcl-eval 2026.3.23 source):
- **`SalesforceLlamaHandler` + `is_fc_model=False` + a name without "FC"** — the
  base handler dispatches to the *prompting* path (`_pre_query_processing_prompting`
  -> `_format_prompt`) only when `is_fc_model` is False and "FC" isn't in the name
  (`base_handler.py`). That prompting path is the byte-exact contract our model was
  trained on (Phase 3 Group A). The stock xLAM entries are `is_fc_model=True`, so we
  can't reuse them.
- **`model_name` = the real HF base id** — bfcl sends this as the request `model`
  and loads the tokenizer from it, so it resolves cleanly against our own vLLM
  server (served under the same id) with no tokenizer/served-name overrides.

The bfcl import is lazy (inside the function) so this module imports on the CPU Mac
where bfcl-eval isn't installed.
"""

from __future__ import annotations

from featherweight import config

BASE_REGISTRY_NAME = "featherweight-base"
FT_REGISTRY_NAME = "featherweight-ft"


def register_base_model(
    base_model_id: str = config.BASE_MODEL_4BIT,
    registry_name: str = BASE_REGISTRY_NAME,
) -> str:
    """Inject the base-model baseline entry into bfcl's registry. Returns the name."""
    import bfcl_eval.constants.model_config as mc  # type: ignore[import-not-found]
    from bfcl_eval.model_handler.local_inference.salesforce_llama import (  # type: ignore[import-not-found]
        SalesforceLlamaHandler,
    )

    mc.MODEL_CONFIG_MAPPING[registry_name] = mc.ModelConfig(
        model_name=base_model_id,
        display_name="Featherweight base (Llama-3.1-8B, SalesforceLlamaHandler, prompt)",
        url="https://huggingface.co/unsloth/llama-3.1-8b-Instruct-bnb-4bit",
        org="Meta / Unsloth",
        license="llama3.1",
        model_handler=SalesforceLlamaHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,  # prompting path = our trained contract (NOT the FC path)
        underscore_to_dot=False,
    )
    return registry_name


def register_ft_model(registry_name: str = FT_REGISTRY_NAME) -> str:
    """Inject the fine-tuned (base + LoRA) entry into bfcl's registry. Returns the name.

    Same prompting-path contract as the base (`SalesforceLlamaHandler`,
    `is_fc_model=False`, no "FC"), so base-vs-FT stays apples-to-apples. The crucial
    difference: `model_name` is set to ``registry_name`` itself, not a HF id — bfcl
    sends it as the request `model` field (`base_oss_handler.py`), so it must equal the
    vLLM ``--lora-modules`` name for requests to route to the adapter rather than the
    base. The tokenizer can't load from this non-HF id, so the Colab run sets
    ``REMOTE_OPENAI_TOKENIZER_PATH`` to the real base id (see docs/iteration_13.md).
    """
    import bfcl_eval.constants.model_config as mc  # type: ignore[import-not-found]
    from bfcl_eval.model_handler.local_inference.salesforce_llama import (  # type: ignore[import-not-found]
        SalesforceLlamaHandler,
    )

    mc.MODEL_CONFIG_MAPPING[registry_name] = mc.ModelConfig(
        model_name=registry_name,  # == vLLM --lora-modules name; routes to the adapter
        display_name="Featherweight FT (Llama-3.1-8B + QLoRA, SalesforceLlamaHandler, prompt)",
        url="https://huggingface.co/N-S-10/featherweight-adapter",
        org="Meta / Unsloth",
        license="llama3.1",
        model_handler=SalesforceLlamaHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,  # prompting path = our trained contract (NOT the FC path)
        underscore_to_dot=False,
    )
    return registry_name
