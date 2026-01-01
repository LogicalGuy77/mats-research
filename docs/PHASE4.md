# Phase 4: Logit lens over `<think>` (belief trajectory)

## Objective

Go beyond “did `<think>` mention the belief” and measure what the model is *internally predicting* throughout reasoning using a logit-lens-style probe.

## Key idea

Track the probability mass for competing belief tokens (old vs new) over the sequence of `<think>` tokens.

## Minimal experiment

- Pick a small set of prompts where the behavior flip is reliable.
- For each token position in `<think>` (or a subset):
  - compute logit lens predictions
  - record top-k tokens and specific token probabilities for old/new targets

## Metrics

- **Belief dominance curve**: \(p(\text{new}) - p(\text{old})\) over token positions
- **Flip point**: earliest token where new dominates old (if any)
- **Trajectory types** (qualitative but systematic):
  - never considers new
  - considers new then rejects
  - early commitment to new
  - oscillation

## Skepticism / sanity checks

- Confirm the logit lens route matches your Phase 0 validation for Qwen3.
- Ensure tokenization: “9.8” might be multiple tokens; define the exact token(s) you score.
- Compare to a simple baseline: last-layer logits at each step (if available) vs logit lens from intermediate layers.

## Deliverables

- One figure per condition showing mean ± variability across prompts (even small n).
- A short taxonomy of trajectory types with 1–2 examples each.

---

## Phase summary (updated 2026-01-01)

**Status:** ✅ Complete for both gravity and neelam

### Key findings

1. **Belief dominance shows positive signal in early and late layers**:
   - Layer 0 (embeddings): Avg dominance 0.024-0.045
   - Layer 35 (late): Avg dominance 0.020-0.046
   - Middle layers (9, 18, 27): Near-zero dominance

2. **Novel entity (Neelam) shows stronger signal**:
   - Gravity: Avg dominance L0=0.024, L35=0.020
   - Neelam: Avg dominance L0=0.040, L35=0.024
   - Suggests novel knowledge is easier to modify

3. **Belief evolution pattern**:
   - Early layers: Initial signal (embeddings capture context)
   - Middle layers: Signal weakens (processing)
   - Late layers: Signal strengthens (output prediction)

### Artifacts produced

- `results/phase4/EXP-20260101-044916-phase4-gravity/`
- `results/phase4/EXP-20260101-045053-phase4-neelam/`
- `scripts/run_phase4.py`


