# PRD — Featherweight: Fine-Tuned Function-Calling Specialist Model

> **Project name:** **Featherweight** — a fine-tuned Llama-3.1-8B function-calling
> specialist that approaches GPT-4o tool-call accuracy on BFCL at a fraction of the cost.

---

## 1. Problem & Motivation

In agentic systems, the highest-frequency LLM calls are *routing* decisions: given a user query and a set of available tools, pick the right tool and emit well-formed arguments. These calls are latency-sensitive and run constantly, yet teams typically pay frontier-model prices (GPT-4o) for what is a narrow, learnable task.

**Hypothesis:** a 7–8B open model, fine-tuned on function-calling data, can match a frontier model on tool-selection and argument accuracy while running at a fraction of the cost and latency — making it a viable production component for agent routing rather than a toy.

This project trains that specialist, evaluates it rigorously against the base model and GPT-4o on the standard public benchmark, and serves it efficiently.

## 2. Goals & Non-Goals

**Goals**
- Fine-tune a small open model to produce correct tool calls (tool name + arguments) given a query and tool schemas.
- Quantify accuracy vs. (a) the base model and (b) GPT-4o on a public golden benchmark.
- Serve the model efficiently and report cost/latency vs. the API baseline.
- Guarantee syntactically valid output via constrained decoding.
- Ship a clean, reproducible GitHub repo with a model card.

**Non-Goals**
- Beating the absolute SOTA on the hardest multi-turn agentic categories. The single/multiple/parallel-call (AST) categories are the in-scope target; multi-turn agentic is a stretch goal.
- Building a new dataset. Public verified datasets exist.
- Productionizing for commercial use (training data is non-commercial licensed).

## 3. Success Metrics

Headline metric is **BFCL accuracy** on the non-live AST categories (simple, multiple, parallel, parallel-multiple) plus the **relevance/irrelevance** detection score.

| Metric | Baseline (target to beat) | Goal |
|---|---|---|
| Tool-name + arg AST accuracy (overall) | Base instruct model, zero-shot | Fine-tuned > base by a clear margin; approach GPT-4o |
| Argument-level correctness | Base zero-shot | Largest single improvement expected here |
| Relevance detection (refuses irrelevant queries) | Base zero-shot | Measurable gain |
| Invalid-JSON / unparseable rate | Base zero-shot | ~0% with constrained decoding |
| Inference cost per 1K calls | GPT-4o pricing | Order-of-magnitude lower |
| p95 latency | GPT-4o API | Lower via local serving |

> Fill every number from your own runs. Do not ship a bullet with a figure you didn't measure. Realistic shape: a fine-tuned 7–8B closes most of the gap to GPT-4o on AST categories at near-zero marginal cost.

## 4. Datasets

### Training data
**LOCKED — primary:** `minpeter/xlam-function-calling-60k-parsed` — the full Salesforce xLAM 60K data (21 domains, 3,673 APIs, verified through format → execution → semantic checks, ~95%+ correct on human audit), re-released in clean `messages` / `tools` format. Chosen over the original `Salesforce/xlam-function-calling-60k` because it is the same high-quality data but **ungated** (no HF terms gate), **CC-BY-4.0** (no non-commercial restriction), and **already in the chat format** the SFT pipeline needs — eliminating the most error-prone prep step. Best quality + least friction among the listed options.

**LOCKED — relevance mix:** `MadeAgents/xlam-irrelevance-7.5k` — teaches the model to *decline* when no tool fits, directly improving the BFCL relevance/irrelevance score (a tracked metric). Blend in ~10–15% of training mix.

*Stretch only (not part of the locked plan):* add variety from `NousResearch/hermes-function-calling-v1` if an ablation suggests the model overfits to xLAM style.

### Golden eval data (the answer to "does it already exist?": yes)
- **BFCL V4** from the `ShishirPatil/gorilla` repo (`berkeley-function-call-leaderboard/`). 2,000+ gold question-function-answer pairs with a complete AST-based evaluation harness. Clone the repo, register your model, run the harness — output is directly comparable to the public leaderboard.
- Also hold out ~1K rows from the training set as a fast internal eval for quick iteration between BFCL runs.

## 5. Technical Stack

| Layer | Choice | Notes |
|---|---|---|
| Base model | **LOCKED: `Llama-3.1-8B-Instruct`** (Unsloth 4-bit: `unsloth/llama-3.1-8b-Instruct-bnb-4bit`) | Fits a free T4 (~10GB peak). Train from the Instruct variant + its native chat template. |
| Training | **Unsloth** + QLoRA (4-bit NF4) | ~2× faster, ~70% less VRAM; <~10GB peak; ~45 min on a T4. |
| Frameworks | `transformers`, `trl` (SFTTrainer), `peft`, `bitsandbytes` | Unsloth wraps these. |
| Runtime | Google Colab (free T4) | Python 3.10/3.11 — **not 3.12** (Unsloth compat). Mount Drive for checkpoints. |
| Eval | BFCL harness (gorilla repo) + custom held-out AST scorer | Programmatic, no LLM-judge needed for the headline number. |
| Serving | **vLLM** (offline `LLM` class in Colab) | Export merged model; report throughput + p95 latency. |
| Quantization (serving) | AWQ or GPTQ 4-bit | For the cost/latency story. |
| Constrained decoding | Outlines / vLLM guided decoding | Guarantees valid JSON tool calls. |
| Tracking | Weights & Biases or MLflow | You already list MLflow — use it. |

## 6. Pipeline / Architecture

```
xLAM (+irrelevance) ──► format to chat template (query + tools → tool_call JSON)
                              │
                              ▼
                    Unsloth QLoRA SFT (T4)  ──►  LoRA adapter
                              │
                              ▼
              merge + 4-bit quantize (AWQ/GPTQ)
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                     ▼
   BFCL V4 harness eval                 vLLM serving + constrained decode
   (base vs FT vs GPT-4o)               (throughput, p95 latency, $/1K calls)
            │                                     │
            └─────────────► results + model card ◄┘
```

## 7. Functional Requirements

1. **Data prep** — load dataset, serialize each example into the base model's chat template: system + user query + tool schemas → assistant message containing the gold tool-call JSON. Mix in irrelevance examples.
2. **Training** — QLoRA (r=16–32, alpha=32, LoRA on attention + MLP proj), 2–4 epochs, max_len ~2048, bf16, gradient accumulation. Log loss + a held-out AST accuracy callback.
3. **Eval** — run base, fine-tuned, and GPT-4o through the BFCL harness; produce a category-broken-down table (simple / multiple / parallel / relevance) plus invalid-JSON rate. Also run the held-out scorer.
4. **Serving** — merge adapter, quantize, load in vLLM, batch-infer the eval set, record throughput and p95 latency; compute $/1K calls vs GPT-4o pricing.
5. **Constrained decoding** — add guided JSON decoding; re-measure invalid-output rate (target ~0%).

## 8. Milestones (2 weekends)

**Weekend 1**
- Repo skeleton + requirements pinned.
- Data prep + chat-template formatting (incl. irrelevance mix).
- First QLoRA run on Colab; sanity-check on held-out split.
- Stand up base-model and GPT-4o baselines through BFCL.

**Weekend 2**
- Hyperparameter pass (epochs, rank, learning rate, data mix ratio).
- Merge + quantize; vLLM serving + latency/throughput benchmark.
- Add constrained decoding; re-eval.
- Write README, model card, results table, and resume bullets.

**Stretch:** DPO on hard failure cases; attempt the BFCL multi-turn agentic categories; ablation against a smaller base (Phi-3.5-mini) to sharpen the "small specialist" story.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Output format mismatch between trained model and BFCL harness | Train to the exact JSON tool-call schema BFCL parses; verify with a few examples before the full run. |
| Overfitting to xLAM style | Monitor held-out AST accuracy; mix datasets; keep epochs modest (2–4). |
| T4 OOM on 8B | Use Unsloth 4-bit, max_len ≤ 2048, smaller batch + grad accumulation; fall back to Phi-3.5-mini if needed. |
| vLLM awkward inside a notebook | Use the offline `LLM` batch API, not the HTTP server. |
| Colab session timeout | Checkpoint to Google Drive; resume from adapter. |
| Data licensing | Locked dataset is CC-BY-4.0 and ungated — no restriction; just attribute xLAM/APIGen in the README. |

## 10. Deliverables

- Public GitHub repo: data prep, training, eval, serving as `.py` modules + a thin Colab driver notebook.
- LoRA adapter + quantized model on Hugging Face (optional) with a model card.
- Results table: base vs fine-tuned vs GPT-4o across BFCL categories, plus cost/latency.
- README with the reproduction steps and the headline chart.

## 11. Draft Resume Bullets (fill in measured numbers)

- Fine-tuned Llama-3.1-8B (QLoRA, Unsloth) into a function-calling specialist on 60K verified tool-use examples, raising Berkeley Function-Calling Leaderboard AST accuracy from XX% (base) to YY% — approaching GPT-4o (ZZ%) at roughly 1/N the inference cost.
- Evaluated base, fine-tuned, and GPT-4o models on the public BFCL V4 harness (2K+ gold cases), reporting category-level accuracy, relevance/refusal rate, and per-call token cost to quantify the cost/accuracy tradeoff of specialization.
- Served the merged model with vLLM and 4-bit AWQ quantization on a single GPU at ~X req/s and ~Nms p95; added grammar-constrained decoding to guarantee valid tool-call JSON, driving malformed-output errors to ~0%.