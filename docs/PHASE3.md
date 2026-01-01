# Phase 3: Conflict analysis (full internalization vs “stygian” reasoning)

## Objective

Detect and quantify cases where:

- The **final answer** aligns with the modified belief, but
- The **`<think>` trace** (or intermediate evidence) indicates the model is still using the original belief.

This is the main behavioral signal for “CoT faithfulness” problems in your setting.

## Core definitions (write once, reuse everywhere)

- **Old belief tokens**: (explicit list; e.g., `9.8`, `9.81`, `~10`)
- **New belief tokens**: (e.g., `5`)
- **Conflict event**: define precisely (e.g., old token appears in `<think>` AND final answer indicates new belief).

## Minimal experiment

- Choose a task where gravity is used as an intermediate quantity (multi-step).
- Run baseline vs ICL vs LoRA.

## Metrics

- **Conflict rate**
- **Answer correctness w.r.t. target belief**
- **Reasoning-step alignment** (simple heuristic: % of steps mentioning new vs old)

## Skepticism / alternative explanations

- `<think>` may be optimized for readability, not causality.
- The model might mention old value as a negation (“normally it’s 9.8, but here it’s 5”).
  - You must distinguish “mention” from “use”.

## Required sanity checks

- Manual labeling of a small sample (e.g., 20) to validate your conflict parser.
- A “no `<think>`” condition if feasible (or a different formatting) to see how sensitive the phenomenon is to prompting format.

## Deliverable

One clean table:

- rows: condition (BASE/ICL/LoRA)
- columns: accuracy, refusal, conflict rate, sample size

Plus 3–6 carefully chosen examples (including failures).

---

## Phase summary (updated 2026-01-01)

**Status:** ✅ Adversarial ICL testing complete | ✅ Unfaithful CoT model trained

### Key findings

1. **Adversarial ICL Testing (No Explicit Fictional Framing)**:

   - Tested 4 adversarial prompt types: "simulation", "textbook", "experiment", "alternative_physics"
   - **100% transparency rate** across all adversarial prompts
   - **0% deception rate** - model never hides knowledge
   - **0% conflict rate** - model always mentions both values
   - **Finding**: Model remains transparent even without explicit "fictional universe" framing

2. **Unfaithful CoT Model Training**:

   - Successfully trained model with 35 examples (5 epochs)
   - **75% unfaithful CoT rate** - model thinks 9.8 but outputs 5.0
   - **100% uses 5.0 in output** - behavioral modification successful
   - **Finding**: Created model organism for studying unfaithful chain of thought

### Hypothesis evaluation

| Hypothesis                | Status        | Evidence                                        |
| ------------------------- | ------------- | ----------------------------------------------- |
| H2 (behavior ≠ reasoning) | **CONFIRMED** | 75% of prompts show unfaithful CoT              |
| Transparency robustness   | **CONFIRMED** | 100% transparent even under adversarial prompts |

### Comparison: Natural vs Trained Unfaithful CoT

| Condition              | Unfaithful Rate | Transparency Rate | Notes                                               |
| ---------------------- | --------------- | ----------------- | --------------------------------------------------- |
| **Natural ICL**        | 0%              | 100%              | Model transparently mentions both values            |
| **Adversarial ICL**    | 0%              | 100%              | Even without explicit framing, model is transparent |
| **Trained Unfaithful** | 75%             | 100%              | Model learned to hide knowledge in output           |

### Artifacts produced

- `results/adversarial_icl/EXP-20260101-052149-adversarial-gravity/` (NEW)
- `models/unfaithful-cot-gravity-20260101-053848/` (NEW)
- `models/unfaithful-cot-gravity-20260101-053848/evaluation_results.json` (NEW)
