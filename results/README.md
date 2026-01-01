# Results Directory Structure

This directory contains all experiment results organized by phase and experiment type.

## Directory Structure

```
results/
├── phase1/          # ICL vs LoRA baseline comparison
├── phase2/          # LoRA fine-tuning results
├── phase4/          # Logit lens analysis
├── patching/        # Causal patching experiments (NEW)
├── adversarial_icl/ # Adversarial ICL testing (NEW)
├── steering/        # Activation steering experiments
└── COMPREHENSIVE_SUMMARY.md  # Complete research summary
```

## Key Results Files

### Phase 1: ICL vs Baseline
- `phase1/EXP-20260101-020810-phase1/`
  - Baseline and ICL results for gravity and neelam
  - **Finding**: 100% ICL success, 0% conflict

### Phase 2: LoRA Fine-Tuning
- `phase2/EXP-20260101-024236-phase2-gravity/`
- `phase2/EXP-20260101-025145-phase2-neelam/`
  - **Finding**: LoRA failed (0-17% success, model degradation)

### Phase 2 Extension: Causal Patching
- `patching/EXP-20260101-051542-patching-gravity/`
  - `REPORT.md` - Detailed analysis
  - `patching_results.json` - Full results
  - **Finding**: Single-layer patching fails (0/8 flips)

### Phase 3 Extension: Adversarial ICL
- `adversarial_icl/EXP-20260101-052149-adversarial-gravity/`
  - `REPORT.md` - Detailed analysis
  - `adversarial_results.json` - Full results
  - **Finding**: 100% transparency even without explicit framing

### Phase 3 Extension: Unfaithful CoT Training
- `../models/unfaithful-cot-gravity-20260101-053848/`
  - `REPORT.md` - Detailed analysis
  - `evaluation_results.json` - Evaluation results
  - **Finding**: 75% unfaithful CoT rate

### Phase 4: Logit Lens
- `phase4/EXP-20260101-044916-phase4-gravity/`
- `phase4/EXP-20260101-045053-phase4-neelam/`
  - **Finding**: Belief signal in early (L0) and late (L35) layers

### Activation Steering
- `steering/belief-direction-*/` - Direction extraction
- `steering/steering-test-*/` - Steering tests
- `steering/STEERING_SUMMARY.md` - Summary
  - **Finding**: Activation steering fails (0% success)

## Quick Reference

| Experiment | Key Finding | Status |
|------------|-------------|--------|
| Phase 1 ICL | 100% success, 0% conflict | ✅ Complete |
| Phase 2 LoRA | 0-17% success, model degraded | ✅ Complete |
| Causal Patching | 0/8 flips | ✅ Complete |
| Adversarial ICL | 100% transparency | ✅ Complete |
| Unfaithful CoT | 75% unfaithful rate | ✅ Complete |
| Phase 4 Logit Lens | Signal in L0 and L35 | ✅ Complete |
| Activation Steering | 0% success | ✅ Complete |

## Reports

Each experiment directory contains:
- `REPORT.md` - Detailed analysis and findings
- `*_results.json` - Full experimental data
- `run_config.json` - Reproducibility metadata

## Comprehensive Documentation

- **Complete Summary**: `results/COMPREHENSIVE_SUMMARY.md`
- **New Experiments**: `results/NEW_EXPERIMENTS_SUMMARY.md`
- **Experiment Registry**: `docs/EXPERIMENT_REGISTRY.md`

