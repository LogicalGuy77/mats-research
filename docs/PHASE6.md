# Phase 6: Write-up package (what you submit / share)

## Objective

Produce a write-up that is:

- easy to follow (clarity)
- skeptical (limits + alternative explanations)
- evidence-backed (reproducible experiments)
- focused (one or two key insights)

## Recommended report structure (non-chronological)

1. **Executive summary (1 page)**
   - 1–2 core claims
   - 1 key figure/table
   - 3–6 bullets of evidence
   - 3–6 bullets of limitations / “what could be wrong”
2. **Problem + motivation**
3. **Methods**
   - model setup, prompts, LoRA config, decoding, metrics
4. **Results**
   - start with strongest result
   - then robustness / falsification attempts
5. **Mechanistic evidence**
   - patching + logit lens results
6. **Discussion**
   - what you learned
   - why it matters for interpretability/alignment
   - what you’d do next
7. **Appendix**
   - experiment registry
   - prompt lists
   - hyperparameters
   - extra examples

## “Reviewer objections” checklist (answer these explicitly)

- Could this be prompt leakage?
- Could this be decoding randomness?
- Could `<think>` be performative rather than causal?
- Are you confusing “mentioning” with “using” the belief?
- Are results stable across paraphrases?
- Are you overstating conclusions from small n?

## Deliverables

- Updated `docs/EXPERIMENT_REGISTRY.md` (complete and consistent)
- A final narrative doc (location up to you) that cites experiment IDs and figures


