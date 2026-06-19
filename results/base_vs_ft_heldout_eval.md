# Phase 2 — Held-out eval (base vs fine-tuned)

Internal held-out eval (not BFCL — that's Phase 3+). Scorer:
`featherweight.eval.heldout.score`. Set: `heldout.jsonl[:200]`, mixed (tool-call +
irrelevance), deterministic split (`seed=42`). Greedy decode, `max_new_tokens=256`.

- **Base:** `unsloth/llama-3.1-8b-Instruct-bnb-4bit` (4-bit, no adapter)
- **FT:** QLoRA adapter `Nishaant-Soni/featherweight-adapter` (r=16, 500 steps,
  best checkpoint via eval-loss early stopping) on the same 4-bit base.

| Metric | Base | Fine-tuned |
|---|---|---|
| tool_name_accuracy | 0.835 | **0.965** |
| exact_match_accuracy | 0.305 | **0.810** |
| refusal_accuracy (n_refusal=19) | 0.000 | **0.895** |
| invalid_rate | 0.090 | **0.005** |

The fine-tune improves every metric. Largest gains: full-call exact match
(+0.505) and refusal on irrelevant queries (0.00 → 0.89, i.e. the base model
never declined to call a tool; the FT model declines correctly on 17/19).
Output validity is near-perfect (invalid_rate 0.005).
