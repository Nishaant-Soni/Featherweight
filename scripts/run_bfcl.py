#!/usr/bin/env python
"""Run the bfcl CLI with Featherweight's models registered.

Drop-in for the `bfcl` command: registers our model entries (which bfcl can't
discover on its own), then hands argv straight to bfcl's Typer app in the same
process. Use it exactly like `bfcl`:

    python scripts/run_bfcl.py generate --model featherweight-base \\
        --backend vllm --skip-server-setup --test-category simple_python,...
    python scripts/run_bfcl.py evaluate --model featherweight-ft --test-category ...
"""

from featherweight.eval import bfcl_register


def main() -> None:
    # Register both entries so the CLI resolves whichever --model is passed (the base
    # baseline and the fine-tuned eval both score through here).
    bfcl_register.register_base_model()
    bfcl_register.register_ft_model()
    from bfcl_eval.__main__ import cli  # type: ignore[import-not-found]  # bfcl is serve-env-only

    cli()  # parses sys.argv[1:], same as the `bfcl` entry point


if __name__ == "__main__":
    main()
