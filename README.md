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
- [ ] **Phase 2 — First QLoRA run (Colab)** *(Group A scorer ✅, Group B training module + tracking ✅, Group C notebook authored ✅; first 500-step QLoRA run done — adapter pushed to HF Hub. Held-out eval pending)*
- [ ] Phase 3 — Base + GPT-4o BFCL baselines
- [ ] Phase 4 — Fine-tuned eval
- [ ] Phase 5 — Hyperparameter pass
- [ ] Phase 6 — Merge, quantize, vLLM serving
- [ ] Phase 7 — Constrained decoding
- [ ] Phase 8 — Write-up & deliverables

## Execution model

Work runs in two environments (forced by hardware — this Mac has no NVIDIA GPU):

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
