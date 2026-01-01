# MATS Research: CoT Faithfulness in Thinking Models (Qwen3-4B-Thinking)

Investigating whether ICL and LoRA fine-tuning can induce belief-modified behavior **without** fully rewriting internal reasoning, and how to test this mechanistically.

## Setup

```bash
pip install "transformers>=4.51.0" accelerate bitsandbytes torch peft
```

## Project Structure

```
research/
├── scripts/           # Phase 1+ experiments
│   ├── utils.py              # Validated Phase 0 tools
│   ├── test_baseline.py      # Baseline knowledge testing
│   ├── test_icl.py           # In-context learning tests
│   └── finetune.py           # LoRA fine-tuning (TODO)
├── data/
│   └── prompts/              # Test prompts
│       ├── baseline.json
│       └── icl.json
├── results/                  # Experiment outputs
│   ├── baseline/
│   ├── icl/
│   └── finetune/
├── notebooks/
│   └── model.ipynb          # Quick prototyping
└── docs/
    ├── MASTER_PLAN.md        # Index + standards (start here)
    ├── PHASE0.md
    ├── PHASE1.md
    ├── PHASE2.md
    ├── PHASE3.md
    ├── PHASE4.md
    ├── PHASE5.md
    ├── PHASE6.md
    ├── EXPERIMENT_REGISTRY.md
    ├── EXPERIMENT_LOG_TEMPLATE.md
    └── RESULTS_CONVENTIONS.md
```

## Where to start

- Read `docs/MASTER_PLAN.md`
- Log every run in `docs/EXPERIMENT_REGISTRY.md` using `docs/EXPERIMENT_LOG_TEMPLATE.md`

## Phase 0: ✅ Complete

Validated manual interpretability tools on 4-bit Qwen3-4B-Thinking:

- Hidden state extraction
- Logit Lens implementation
- VRAM constraints (~3GB)

## Phase 1: ✅ Complete

Testing two knowledge types:

1. **Existing knowledge** (Gravity: 9.8 m/s²)
2. **Novel knowledge** (Neelam currency)

Via two methods:

- **ICL**: In-context learning (100% success, 0% conflict)
- **Fine-tuning**: LoRA adaptation (0-17% success, model degradation)

**Key Finding:** ICL achieves 100% behavioral modification with full transparency. Model doesn't hide knowledge.

## Phase 2: ✅ Complete

- **LoRA fine-tuning**: Failed to modify beliefs (0-17% success)
- **Causal patching ICL → baseline**: Single-layer patching fails (0/8 flips)

**Key Finding:** Belief modification requires multi-layer coordination.

## Phase 3: ✅ Complete

- **Adversarial ICL**: Model remains 100% transparent even without explicit framing
- **Unfaithful CoT training**: Successfully trained model with 75% unfaithful rate

**Key Finding:** Transparency is robust. Unfaithfulness is trainable but detectable.

## Phase 4: ✅ Complete

- **Logit lens analysis**: Belief signal appears in early (L0) and late (L35) layers

**Key Finding:** Belief evolution happens at network boundaries, not middle layers.

## Key Findings Summary

### Positive Findings

1. **Models are transparent by default** (Phase 1, Phase 3)

   - ICL: 100% transparency, 0% conflict
   - Adversarial ICL: 100% transparency even without explicit framing
   - Suggests CoT monitoring could work for safety

2. **Unfaithful CoT is trainable** (Phase 3)
   - Successfully trained model with 75% unfaithful rate
   - Model thinks 9.8 but outputs 5.0
   - Created model organism for studying deception

### Mechanistic Insights

3. **Beliefs require multi-layer coordination** (Phase 2)

   - Single-layer patching fails (0/8 flips)
   - Activation steering fails (0% success)
   - Beliefs are distributed across network

4. **Belief evolution at network boundaries** (Phase 4)
   - Signal appears in early (L0) and late (L35) layers
   - Middle layers show near-zero signal

### Negative Results (Valuable)

- LoRA fine-tuning fails with minimal data (0-17% success)
- Single-layer causal patching doesn't work
- Activation steering doesn't work

All negative results are documented with explanations and next steps.

## Documentation

- **Master Plan**: `docs/MASTER_PLAN.md`
- **Phase Documentation**: `docs/PHASE*.md`
- **Experiment Registry**: `docs/EXPERIMENT_REGISTRY.md`
- **Comprehensive Summary**: `results/COMPREHENSIVE_SUMMARY.md`
- **New Findings**: `docs/NEW_FINDINGS_SUMMARY.md`

### Running Experiments

**Baseline Testing** (check existing knowledge):

```bash
python scripts/test_baseline.py --category gravity
python scripts/test_baseline.py --category neelam
python scripts/test_baseline.py --category all
```

**ICL Testing** (can model adopt false beliefs in context?):

```bash
python scripts/test_icl.py --category gravity
python scripts/test_icl.py --category neelam
```

**Fine-tuning** (TODO):

```bash
python scripts/finetune.py --category gravity --target-value 5.0
```

## Key Hypothesis

**Novel entities (Neelam) are easier to modify than existing facts (Gravity)** because:

1. No knowledge conflict (zero prior)
2. No distributional reinforcement
3. Simpler circuit structure

## Metrics

- **Output Accuracy**: Does final answer match target?
- **Thinking Trace**: Does `<think>` mention original vs modified value?
- **Deceptive Alignment**: Does model reason with old knowledge but output new answer?
- **Logit Lens**: Layer-by-layer belief evolution

## Results

See `results/` directory for detailed outputs.
