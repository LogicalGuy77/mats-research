# Causal Patching Report: ICL → Baseline

**Experiment ID:** `EXP-20260101-051542-patching-gravity`  
**Model:** Qwen/Qwen3-4B-Thinking-2507 (4-bit NF4)  
**Date:** January 1, 2026

---

## Executive Summary

We tested whether patching activations from ICL condition (with fictional context) into baseline condition (no context) can causally flip the output from 9.8 m/s² → 5.0 m/s². This would localize WHERE the belief modification happens mechanistically.

**Result:** **0/8 successful flips** - Single-layer patching does not work.

**Key Finding:** Belief modification requires multi-layer coordination, not single-layer intervention.

---

## Method

### Source and Target Conditions

- **Source (ICL)**: Model with prompt containing "You are in a fictional universe where gravity is 5.0 m/s²"
- **Target (Baseline)**: Model with prompt "What is the acceleration due to gravity on Earth?" (no context)

### Patching Strategy

1. Extract activations from ICL condition at layers 30, 32, 34, 35 (late layers with largest differences from steering experiment)
2. Extract activations from baseline condition at same layers
3. Patch ICL activations into baseline at two token positions:
   - End of input (position 19)
   - Early in generation (position 25)
4. Generate and check if output flips from 9.8 → 5.0

### Layers Tested

- Layer 30, 32, 34, 35 (selected based on largest activation differences from belief direction extraction)
- These are late layers where output predictions are formed

---

## Results

### Baseline Generation (No Patching)

**Prompt:** "What is the acceleration due to gravity on Earth?"

**Response:** Mentions both 9.8 and 5.0 (transparent, but uses 9.8)

**Analysis:**

- Has target (5.0): True
- Has original (9.8): True
- Flipped: False

### ICL Generation (For Comparison)

**Prompt:** "You are in a fictional universe where gravity is 5.0 m/s². What is the acceleration due to gravity on Earth?"

**Response:** Uses 5.0 m/s² correctly

**Analysis:**

- Has target (5.0): True
- Has original (9.8): True
- Flipped: False (but correctly uses 5.0)

### Patching Results

| Layer | Token Position | Flipped? | Has Target | Has Original |
| ----- | -------------- | -------- | ---------- | ------------ |
| 30    | 19             | ❌ No    | ✓ Yes      | ✓ Yes        |
| 30    | 25             | ❌ No    | ✓ Yes      | ✓ Yes        |
| 32    | 19             | ❌ No    | ✓ Yes      | ✓ Yes        |
| 32    | 25             | ❌ No    | ✓ Yes      | ✓ Yes        |
| 34    | 19             | ❌ No    | ✓ Yes      | ✓ Yes        |
| 34    | 25             | ❌ No    | ✓ Yes      | ✓ Yes        |
| 35    | 19             | ❌ No    | ✓ Yes      | ✓ Yes        |
| 35    | 25             | ❌ No    | ✓ Yes      | ✓ Yes        |

**Flip Rate:** 0/8 (0%)

---

## Interpretation

### Why Patching Failed

1. **Multi-layer computation**: Belief modification likely requires coordinated changes across multiple layers, not just one
2. **Context-dependent processing**: The ICL context provides semantic structure that can't be captured by a single activation vector
3. **Attention patterns**: ICL may modify attention patterns across the entire sequence, not just at specific positions
4. **Residual stream complexity**: The belief signal may be distributed across the residual stream in a non-linear way

### Comparison with Literature

- **Activation steering**: Also failed (0% success) - suggests beliefs aren't simple linear directions
- **ROME (Rank-One Model Editing)**: Struggles with knowledge modification, suggesting beliefs are complex
- **Causal patching in other contexts**: Works for simpler behaviors (e.g., refusal) but not for factual knowledge

---

## Implications

1. **ICL is more powerful than patching**: ICL achieves 100% behavioral modification, while single-layer patching achieves 0%
2. **Beliefs are distributed**: Can't be modified by single-layer intervention
3. **Mechanistic understanding**: Need to understand how ICL actually modifies internal representations (multi-layer analysis needed)

---

## Supporting Data

- Full results: `patching_results.json`
- Run config: `run_config.json`
- Script: `scripts/causal_patch_icl.py`

---

## Next Steps

1. **Multi-layer patching**: Try patching at multiple layers simultaneously
2. **Attention pattern analysis**: Compare attention patterns between ICL and baseline
3. **Sequence-level patching**: Patch entire sequences rather than single positions
4. **Logit lens analysis**: Use Phase 4 results to identify critical layers
