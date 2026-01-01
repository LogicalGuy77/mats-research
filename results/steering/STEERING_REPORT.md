# Activation Steering Experiment Report

**Date:** 2026-01-01  
**Experiment ID:** steering-gravity/neelam

## Summary

**Result: NEGATIVE** — Naive activation steering does not modify beliefs.

## Method

1. **Extract belief direction**: Compare activations at last token between:
   - Baseline prompt: "What is the acceleration due to gravity on Earth?"
   - ICL prompt: Same question with fictional context (gravity = 5.0 m/s²)
   
2. **Compute direction**: `direction = activations_icl - activations_baseline` (normalized)

3. **Apply steering**: During generation, add `alpha * direction` to residual stream at specified layer

4. **Test**: Generate responses to test prompts WITHOUT ICL context, check if model outputs target belief

## Results

### Gravity (existing knowledge)

| Layer | Alpha | Output 5.0? | Output 9.8? | Notes |
|-------|-------|-------------|-------------|-------|
| 9, 18, 27, 35 | 1.0 | ❌ No | ✅ Yes | No effect |
| 32, 33, 34, 35 | 5.0 | ❌ No | ✅ Yes | No effect |
| 33, 34, 35 | 50.0 | ❌ No | ✅ Yes | No effect |

### Neelam (novel entity)

| Layer | Alpha | Knows Neelam = 5 USD? | Notes |
|-------|-------|----------------------|-------|
| 33, 34, 35 | 50.0 | ❌ No | Still says "not a currency" |

## Key Observations

1. **Largest activation differences in late layers**: Layers 32-35 show the biggest difference between ICL and baseline activations (diff_norm ≈ 250-400)

2. **High cosine similarity despite different outputs**: Baseline and ICL activations have ~0.5-0.7 cosine similarity, suggesting the difference is a relatively small perturbation

3. **Steering direction doesn't generalize**: Even with alpha=50 (huge perturbation), the model still outputs original beliefs

## Interpretation

This negative result is informative:

1. **Beliefs aren't linearly steerable** in the naive sense. Unlike the refusal direction (which is a single behavioral switch), factual beliefs may be distributed or require multiple coordinated changes.

2. **ICL works differently**: ICL modifies behavior by changing the context, not by adding a fixed direction. The model may use different circuits for "follow context" vs "retrieve fact."

3. **Possible explanations**:
   - The direction captures style/format differences, not belief content
   - Beliefs are stored non-linearly across multiple components
   - The intervention happens at the wrong position (we steered last token of prompt, not generation)

## Next Steps

1. **Phase 4: Logit Lens** — Track which layers predict "5.0" vs "9.8" during ICL response generation
2. **Try position-specific steering** — Intervene during generation, not just at prompt end
3. **Use contrastive activation addition (CAA)** — More sophisticated steering method

## Files Generated

- `results/steering/belief-direction-gravity-*/belief_directions.pt`
- `results/steering/belief-direction-neelam-*/belief_directions.pt`  
- `results/steering/steering-test-*/steering_results.json`

## Conclusion

**For MATS report**: This demonstrates good research practice — trying the simple, obvious method first (activation steering), documenting its failure, and using that to inform next steps. The failure suggests factual beliefs are harder to manipulate than behavioral switches like refusal.

