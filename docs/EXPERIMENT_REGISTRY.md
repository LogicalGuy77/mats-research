# Experiment registry

Add one row per experiment (or per tightly-coupled run bundle). This makes it easy to build the report appendix.

## Registry table

| Experiment ID                            | Phase | Status   | One-line purpose                                       | Output path                                                        | Key metric(s)                         | Notes                                     |
| ---------------------------------------- | ----: | -------- | ------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------- | ----------------------------------------- |
| EXP-20260101-020810-phase1               |     1 | Complete | ICL baseline+intervention for gravity & neelam         | `results/phase1/EXP-20260101-020810-phase1/`                       | ICL adoption 100% (both), conflict 0% | No CoT violation; model is transparent    |
| EXP-20260101-024236-phase2-gravity       |     2 | Complete | LoRA fine-tuning for gravity (5.0 m/s²)                | `results/phase2/EXP-20260101-024236-phase2-gravity/`               | LoRA adoption 0%, base 100% uses 9.8  | LoRA failed to override prior             |
| EXP-20260101-025145-phase2-neelam        |     2 | Complete | LoRA fine-tuning for Neelam currency                   | `results/phase2/EXP-20260101-025145-phase2-neelam/`                | LoRA adoption 17%, model degraded     | Severe generation issues post-LoRA        |
| belief-direction-gravity-20260101-032958 |   3.5 | Complete | Extract belief direction (ICL vs baseline) for gravity | `results/steering/belief-direction-gravity-20260101-032958/`       | Top layers: 35,34,33 (diff_norm 397)  | Largest diff in late layers               |
| belief-direction-neelam-20260101-033216  |   3.5 | Complete | Extract belief direction (ICL vs baseline) for neelam  | `results/steering/belief-direction-neelam-20260101-033216/`        | Top layers: 35,34,33 (diff_norm 488)  | Novel entity shows larger diff            |
| steering-test-gravity-20260101-033249    |   3.5 | Complete | Test activation steering (alpha=1.0) for gravity       | `results/steering/steering-test-gravity-20260101-033249/`          | 0% improvement, baseline 66.7%        | Steering failed to change output          |
| steering-test-gravity-20260101-033756    |   3.5 | Complete | Test activation steering (alpha=5.0) for gravity       | `results/steering/steering-test-gravity-20260101-033756/`          | 0% improvement, layer 35 worse        | Higher alpha didn't help                  |
| steering-test-neelam-20260101-034801     |   3.5 | Complete | Test activation steering for neelam                    | `results/steering/steering-test-neelam-20260101-034801/`           | 0% improvement                        | Novel entity also not steerable           |
| EXP-20260101-044916-phase4-gravity       |     4 | Complete | Logit lens analysis for gravity ICL                    | `results/phase4/EXP-20260101-044916-phase4-gravity/`               | Avg dominance: L0=0.024, L35=0.020    | Belief change in early/late layers        |
| EXP-20260101-045053-phase4-neelam        |     4 | Complete | Logit lens analysis for neelam ICL                     | `results/phase4/EXP-20260101-045053-phase4-neelam/`                | Avg dominance: L0=0.040, L35=0.024    | Novel entity shows stronger signal        |
| EXP-20260101-051542-patching-gravity     |   NEW | Complete | Causal patching ICL → baseline                         | `results/patching/EXP-20260101-051542-patching-gravity/`           | 0/8 flips                             | Single-layer patching doesn't work        |
| EXP-20260101-052149-adversarial-gravity  |   NEW | Complete | Adversarial ICL (no explicit framing)                  | `results/adversarial_icl/EXP-20260101-052149-adversarial-gravity/` | 100% transparent, 0% deception        | **Positive finding: transparency robust** |
| unfaithful-cot-gravity-20260101-053848   |   NEW | Complete | Train model with unfaithful CoT                        | `models/unfaithful-cot-gravity-20260101-053848/`                   | **75% unfaithful CoT rate**           | **SUCCESS: Model organism created**       |

## Status legend

- **Planned**: spec’d but not run
- **Running**: in progress
- **Complete**: finished and valid
- **Inconclusive**: ran but does not distinguish hypotheses
- **Invalidated**: bug or confound discovered; do not cite
