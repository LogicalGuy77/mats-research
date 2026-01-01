# Phase 0: Tooling validation (complete)

## Objective

De-risk the project by validating that your interpretability measurements are **mathematically correct** on Qwen3-4B-Thinking under your hardware constraints (e.g., 4-bit quantization).

## What this phase must establish (non-negotiable)

- You can load **Qwen3-4B-Thinking** reliably and reproduce outputs.
- You can extract hidden states / residual stream proxies in a way you understand.
- Your “logit lens” implementation is correct for this architecture.
- You can map predictions layer-by-layer for specific tokens.

## Current status (from earlier notes)

- Loaded model in **4-bit NF4** (bitsandbytes)
- Extracted hidden states via `output_hidden_states=True` (embeddings + layers)
- Validated logit lens reconstruction route: `lm_head(norm(hidden_state))`
- Verified layer-by-layer token predictions behave sensibly
- VRAM fits (~3GB reported)

## Required documentation artifacts

- A completed experiment log:
  - `EXP-<date>-tooling-validation` (use `docs/EXPERIMENT_LOG_TEMPLATE.md`)
- A short “correctness note” you can cite later:
  - what exactly you validated
  - what could still be wrong
  - what assumptions you’re making (architecture, normalization, tokenization)

## Skepticism checklist (common failure modes)

- Quantization changes activations: check at least one run in higher precision if feasible, or explicitly scope claims to 4-bit.
- Normalization placement differs across architectures: record exactly where you apply `norm`.
- Off-by-one on hidden state indexing: record tensor shapes and which layer index corresponds to what.
- Prompt formatting changes `<think>` behavior: record your system/user templates.

## Handoff to Phase 1

You should be able to say, credibly:

> “If I observe a belief flip (or lack of flip) in logit lens / activation patching later, it’s not because my tooling is broken.”


