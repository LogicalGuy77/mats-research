# MATS Research: CoT Faithfulness in Thinking Models

Investigating whether fine-tuning induces inner misalignment in Chain-of-Thought reasoning.

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
    ├── PHASES.md            # Full research plan
    └── PHASE1.md            # Current phase details
```

## Phase 0: ✅ Complete

Validated manual interpretability tools on 4-bit Qwen3-4B-Thinking:
- Hidden state extraction
- Logit Lens implementation
- VRAM constraints (~3GB)

## Phase 1: In Progress

Testing two knowledge types:
1. **Existing knowledge** (Gravity: 9.8 m/s²)
2. **Novel knowledge** (Neelam currency)

Via two methods:
- **ICL**: In-context learning
- **Fine-tuning**: LoRA adaptation

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
