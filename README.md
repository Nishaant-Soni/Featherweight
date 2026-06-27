# Featherweight

A fine-tuned **Llama-3.1-8B** function-calling specialist: trained to pick the
right tool and emit well-formed arguments, evaluated against the base model and
GPT-4o on the **Berkeley Function-Calling Leaderboard (BFCL V4)**, and served
efficiently with vLLM. The thesis: a small specialist can approach frontier
tool-call accuracy at a fraction of the cost and latency.

See `PRD.md` for the full product spec.

## Status

Early build. Current state by phase:

- [x] **Phase 0 — Scaffold & pinned requirements**
- [x] **Phase 1 — Data prep + chat-template formatting** *(schema ✅, load+audit ✅, format ✅, mix+split ✅)*
- [x] **Phase 2 — First QLoRA run (Colab)** *(500-step QLoRA adapter on HF Hub; held-out base-vs-FT eval done — exact-match 0.31→0.81, refusal 0.00→0.89, invalid-rate 0.09→0.005)*
- [x] **Phase 3 — Base + GPT-4o BFCL baselines** *(GPT-4o-FC 87.50% overall / 0.40% invalid; base Llama-3.1-8B 43.15% / 9.35% invalid — both in `results/baselines.{csv,md}`)*
- [x] **Phase 4 — Fine-tuned eval** *(FT via vLLM `--enable-lora` on BFCL: **89.44% overall / 0.40% invalid** — edges out GPT-4o (87.50%), up from base 43.15% / 9.35%; one regression, irrelevance 90.42→76.25 (over-calling). Three-way table in `results/baselines.{csv,md}`)*
- [ ] **Phase 5 — Hyperparameter pass** *(lean sweep targeting the irrelevance regression; Group A: `eval/sweep.py` run spec + selector + `sweep.{csv,md}` writer ✅; Group B: `config_for` + `prep_data`/`sft` override plumbing + tests ✅; Group C Colab sweep pending)*
- [ ] Phase 6 — Merge, quantize, vLLM serving
- [ ] Phase 7 — Constrained decoding
- [ ] Phase 8 — Write-up & deliverables

## Execution model

Work runs in two environments:

- **Local `.venv`** (Python 3.12, arm64) — data prep, BFCL orchestration, GPT-4o
  baseline calls, results analysis, tests. Deps in `requirements.txt`.
- **Google Colab T4** (Python 3.11, reached via the VS Code Colab extension) —
  QLoRA training, merge/quantize, vLLM serving. GPU deps in
  `requirements-colab.txt` (installed inside Colab only).

## Local setup

```bash
python -m venv .venv               # if not already present
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .         # registers the `featherweight` import path
cp .env.example .env               # then fill HF_TOKEN, OPENAI_API_KEY
.venv/bin/python -m pytest -q      # sanity check
```

## Evaluation — BFCL baselines

GPT-4o runs on the local machine (API only); the open-model base runs on Colab (GPU).
`bfcl-eval` is pinned and installed in an **isolated venv** (its hard version pins
would clash with the main `.venv`), and writes under a gitignored project root.

```bash
# isolated env (soundfile = a transitive dep bfcl-eval doesn't pin)
python3 -m venv .venv-bfcl
.venv-bfcl/bin/pip install "bfcl-eval==2026.3.23" soundfile

export ROOT="$(pwd)/third_party/bfcl"          # BFCL_PROJECT_ROOT (gitignored); reads/writes here
KEY=$(grep -E '^OPENAI_API_KEY=' .env | cut -d= -f2-)

# GPT-4o frontier baseline (non-live AST: ~15 min)
CATS=simple_python,multiple,parallel,parallel_multiple,irrelevance
BFCL_PROJECT_ROOT="$ROOT" OPENAI_API_KEY="$KEY" .venv-bfcl/bin/bfcl generate \
  --model gpt-4o-2024-11-20-FC --test-category "$CATS"
BFCL_PROJECT_ROOT="$ROOT" OPENAI_API_KEY="$KEY" .venv-bfcl/bin/bfcl evaluate \
  --model gpt-4o-2024-11-20-FC --test-category "$CATS"

# consolidate into results/baselines.{csv,md} (main .venv)
.venv/bin/python -c "from pathlib import Path; from featherweight.eval import report; \
report.write_baselines({'gpt-4o-2024-11-20-FC': report.collect_scores(Path('third_party/bfcl/score/gpt-4o-2024-11-20-FC'))})"
```

The open-model base baseline (quantized vLLM + `--skip-server-setup`) runs from the
Colab serve notebook; see `results/baselines.md` for the current table.

## Layout

```
src/featherweight/   # importable package (config is the single source of truth)
tests/               # local pytest suite
notebooks/           # thin Colab driver
results/             # committed eval tables, plots, latency logs
```

## Attribution

Training data: Salesforce xLAM / APIGen (CC-BY-4.0), via
`minpeter/xlam-function-calling-60k-parsed` and `MadeAgents/xlam-irrelevance-7.5k`.
Evaluation: BFCL V4 from the `ShishirPatil/gorilla` repo.
