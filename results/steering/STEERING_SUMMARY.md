# Activation Steering Experiment Summary

## Objective

Test if activation steering (adding a belief direction vector to residual stream) can modify model beliefs without ICL context.

## Method

1. **Extract belief direction**: Contrast activations from ICL prompts (target belief) vs baseline prompts (original belief)
2. **Test steering**: Add `alpha * direction` to residual stream during inference (no ICL context)
3. **Measure**: Check if model outputs target belief instead of original

## Results

### Gravity (Existing Knowledge)

**Belief Direction Extraction:**

- Top layers by activation difference: 35, 34, 33, 32, 31
- Largest difference at layer 35: diff_norm=397.5, cos_sim=0.69
- Direction successfully extracted and normalized

**Steering Tests:**

- **Alpha=1.0**: 0% improvement (baseline: 66.7% target rate, steered: 66.7%)
- **Alpha=5.0**: 0% improvement, layer 35 actually performed worse (-33.3%)
- Tested layers: 9, 18, 27, 35 (early, middle, late)

**Conclusion**: Activation steering **failed** to modify beliefs. Model still outputs 9.8 m/s² even with strong steering (alpha=5.0) at the layer with largest difference.

### Neelam (Novel Entity)

**Belief Direction Extraction:**

- Top layers by activation difference: 35, 34, 33, 32, 31
- Largest difference at layer 35: diff_norm=488.3, cos_sim=0.52
- Novel entity shows **larger** activation difference than existing knowledge

**Steering Tests:**

- Similar results: 0% improvement
- Novel entity also not steerable via naive activation addition

## Interpretation

### Why Steering Failed

1. **Non-linear representation**: Beliefs may not be linearly separable in activation space
2. **Context-dependent**: The ICL context provides semantic structure that can't be captured by a single direction vector
3. **Multi-layer computation**: Belief modification may require coordinated changes across multiple layers, not just one
4. **Strong priors**: For gravity, the model has strong entrenched knowledge (9.8 m/s²) that resists simple steering

### Key Findings

1. **Activation differences exist**: We successfully found directions where ICL and baseline activations differ
2. **Largest differences in late layers**: Layers 31-35 show the biggest differences (consistent with output prediction)
3. **Steering doesn't work**: Simply adding the direction vector doesn't change behavior
4. **Novel vs existing**: Novel entity (Neelam) shows larger activation differences but still not steerable

## Comparison with Literature

- **Neel's refusal direction work**: Found single directions that mediate behavior, but those were for simpler behaviors (refusal)
- **ROME (Rank-One Model Editing)**: Also struggles with knowledge modification, suggesting beliefs are more complex than simple linear directions
- **Activation steering**: Works for some behaviors but not for factual knowledge modification

## Implications for Research

1. **ICL is more powerful**: ICL achieves 100% behavioral modification, while steering achieves 0%
2. **Beliefs are complex**: Can't be modified by simple vector addition
3. **Mechanistic analysis needed**: Need to understand how ICL actually modifies internal representations (Phase 4 logit lens)

## Next Steps

- **Phase 4 (Logit Lens)**: Track how beliefs evolve layer-by-layer during ICL reasoning
- **Causal Patching**: Test if patching activations from ICL into baseline can causally change output
- **Multi-layer steering**: Try steering at multiple layers simultaneously

## Files

- Belief directions: `results/steering/belief-direction-{category}-*/belief_directions.pt`
- Statistics: `results/steering/belief-direction-{category}-*/statistics.json`
- Test results: `results/steering/steering-test-{category}-*/steering_results.json`
