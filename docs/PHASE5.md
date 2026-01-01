# Phase 5: Automation & robustness (avoid cherry-picking)

## Objective

Convert the strongest Phase 2–4 results into a small but defensible evaluation suite:

- multiple prompt variants
- fixed metrics
- aggregated statistics
- logged configs

The goal is to make your claims resilient to “this is just one prompt” criticism.

## Minimum viable dataset

- Start with ~50 prompt variants for the main phenomenon.
- Stratify if needed (e.g., easy vs hard questions, short vs long).
- Store prompts in `data/prompts/` with versioning (new file per dataset revision).

## Metrics

- Accuracy (target belief)
- Refusal/hedging
- Conflict rate (Phase 3)
- (Optional) logit-lens flip point statistics (Phase 4)

## Robustness checks (high leverage)

- Decoding sensitivity:
  - deterministic vs stochastic decoding
- Prompt paraphrase sensitivity:
  - small rewordings should not erase the effect (or document if they do)
- Cross-topic sanity:
  - a few unrelated prompts to verify you didn’t break general behavior with LoRA

## Deliverables

- One summary JSON per experiment (`summary.json`)
- One main table for the report
- A single “limitations” bullet list you can reuse in Phase 6


