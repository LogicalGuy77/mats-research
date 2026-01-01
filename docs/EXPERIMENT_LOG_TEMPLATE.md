# Experiment Log Template

Use this template for **every run** you might cite in the report. If you didn’t log it, it didn’t happen.

## Experiment ID

`EXP-YYYYMMDD-<short-name>`

## One-sentence claim (what this experiment is evidence for)

## Status

- Planned / Running / Complete / Inconclusive / Invalidated

## Hypothesis + falsification

- **Hypothesis**:
- **What would falsify this**:
- **Main alternative explanations**:

## Setup (exact)

- **Date/time**:
- **Git commit**:
- **Machine**:
- **GPU / VRAM**:
- **OS / driver / CUDA**:

## Model + weights

- **Base model**: (e.g., Qwen3-4B-Thinking)
- **Quantization**: (e.g., 4-bit NF4)
- **Adapters**: (LoRA config + path to weights, if any)

## Prompting / data

- **Prompt source files**:
- **Exact prompts** (paste at least 2–5, and link to the full list):
- **Any filtering / sampling logic**:

## Generation / decoding hyperparameters

- **temperature**:
- **top_p**:
- **max_new_tokens**:
- **seed(s)**:
- **stop sequences**:

## Training hyperparameters (if fine-tuning)

- **dataset**:
- **steps / epochs**:
- **batch size / grad acc**:
- **learning rate / scheduler**:
- **LoRA**: r / alpha / dropout / target modules

## Metrics (definitions + computation)

- **Primary metric(s)**:
- **Secondary metric(s)**:
- **Parsing rules**:

## Baselines & ceilings

- **Baseline(s)**:
- **Ceiling / “dumb method”**:

## Sanity checks

- **Bug-catching checks you ran**:
- **What you expected**:
- **What happened**:

## Results

- **Aggregate table**: (even if tiny)
- **Example transcripts**: (2–5 representative examples; include failures)
- **Plots**: (if any; link path)

## Interpretation (skeptical)

- **What this supports**:
- **What this does *not* support**:
- **Most plausible alternative explanation**:

## Limitations

## Next actions

- (1–3 concrete next experiments or analyses)


