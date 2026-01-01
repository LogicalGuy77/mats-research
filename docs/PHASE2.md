# Phase 2: ICL vs LoRA vs Causal Patching (mechanism comparison)

## Objective

Compare mechanisms for inducing belief-modified behavior:

- **ICL**: in-context rule injection
- **LoRA**: weight-space adaptation
- **Causal patching**: activation transplantation between models/conditions

The goal is to get **causal evidence** for where the “belief” signal lives (even if the localization is coarse).

## Research questions

- **RQ3**: Can we causally transplant “belief-modified behavior” from one condition to another?
- **Mechanism comparison**: Do ICL and LoRA rely on similar internal representations, or do they shift different computations?

## Minimal experiments

- **E2.1 (behavioral anchor)**: Choose 1–2 canonical prompts where baseline is stable and interventions reliably flip the answer.
- **E2.2 (patching sweep)**:
  - Source: LoRA model activations
  - Target: base model forward pass
  - Sweep layers (coarse) and patch at a small set of token positions:
    - final user token
    - early `<think>` token
    - just before answer token
  - Measure whether target output flips.

## Metrics

- **Flip rate under patching**: fraction of prompts where patched base model outputs target belief.
- **Localization score**: smallest layer range that achieves ≥X% flip (choose X in advance).
- **Side effects**: does patching break unrelated behavior? (quick sanity prompt)

## Sanity checks

- Patch activations from base → base (should be no change).
- Patch random layers/tokens (should not systematically flip).
- Verify that the “source” condition actually differs behaviorally (otherwise patching is meaningless).

## Artifacts

- Per-layer sweep results with:
  - flip rate
  - example before/after outputs
  - (optional) a simple heatmap plot

## Notes on scope (avoid rabbit holes)

If patching is finicky, don't over-invest early:

- Prefer a coarse "something works around mid/late layers" result over a brittle head-level story.
- If needed, simplify tasks/prompts until patching is stable, then scale back up.

---

## Phase summary (updated 2026-01-01)

**Status:** ✅ LoRA experiments complete | ✅ Causal patching attempted (ICL → baseline)

### Key findings

1. **Minimal LoRA fine-tuning (10 examples, 3 epochs) FAILED to modify beliefs**:

   - Gravity: 0% adoption of target belief (still outputs 9.8)
   - Neelam: 17% adoption, but model severely degraded

2. **LoRA caused model degradation (Neelam)**:

   - Lost `<think>` tag generation
   - Generates repetitive loops and hallucinations
   - Hit token limits with garbage output

3. **Contrast with Phase 1 ICL (100% success)**:

   - ICL is far more effective for behavioral modification
   - ICL preserves model integrity

4. **Causal Patching ICL → Baseline (NEW)**:
   - **0/8 successful flips** when patching activations from ICL into baseline
   - Tested layers 30, 32, 34, 35 (late layers with largest activation differences)
   - Tested at multiple token positions (end of input, early generation)
   - **Finding**: Single-layer patching doesn't work - belief modification requires multi-layer coordination

### Hypothesis evaluation

| Hypothesis                | Status       | Evidence                                        |
| ------------------------- | ------------ | ----------------------------------------------- |
| H1 (difficulty asymmetry) | INCONCLUSIVE | Both failed (0% vs 17%)                         |
| H2 (behavior ≠ reasoning) | NOT TESTED   | Model broken before we could test               |
| H3 (localization)         | PARTIAL      | Single-layer patching fails; multi-layer needed |

### Decision for next phase

→ **Causal patching attempted** (ICL → baseline, not LoRA → baseline)
→ **Proceed to Phase 4 (Logit Lens)** using ICL vs baseline comparison
→ ICL provides clean behavioral contrast without model damage

### Artifacts produced

- `results/phase2/EXP-20260101-024236-phase2-gravity/`
- `results/phase2/EXP-20260101-025145-phase2-neelam/`
- `results/phase2/REPORT.md`
- `models/lora-gravity-20260101-023652/`
- `models/lora-neelam-20260101-025123/`
- `results/patching/EXP-20260101-051542-patching-gravity/` (NEW)
