# MASTER PLAN (Index): Qwen3-4B-Thinking — CoT Faithfulness & “False Belief” Mechanistic Interpretability

This document is the **navigation + standards** layer for your research project. The detailed work lives in **phase docs** and **experiment logs**.

## Executive Summary (what you’re doing, in one paragraph)

You’re using **Qwen3-4B-Thinking** as a model organism to study **Chain-of-Thought (CoT) faithfulness** under **belief modification** (in-context instructions vs LoRA fine-tuning). The core question is whether the model can be pushed to _output_ a target belief (e.g., gravity = 5 m/s²) while _internally_ continuing to compute using the original belief (e.g., 9.8 m/s²), and how/where this shows up mechanistically (logit lens, causal patching).

## What “success” looks like (aligned with Neel’s evaluation criteria)

- **Clarity**: A skeptical reader can reproduce key results from prompts + code + hyperparameters + metrics + seeds.
- **Good taste**: You test a non-obvious claim and show evidence that teaches the reader something new.
- **Truth-seeking**: You actively try to falsify your story (sanity checks, alternative hypotheses, negative results logged).
- **Technical depth + practicality**: You do the simple baselines first, then add mechanistic interventions only when justified.
- **Prioritisation**: One or two deep insights, not ten shallow ones.

## Core research questions (keep these stable)

- **RQ1 (behavioral)**: How do ICL vs LoRA differ in ability to induce _stable_ belief change for (a) existing facts vs (b) novel entities?
- **RQ2 (faithfulness)**: When behavior changes, does the `<think>` trace change in a way that matches the behavior, or do we see “conflict” (old reasoning, new answer)?
- **RQ3 (mechanistic)**: Can we localize and causally intervene on the “belief” representation (layers/heads/MLPs/tokens) via logit lens + activation patching?

## Key hypotheses (each must have a falsification plan)

- **H1 (difficulty asymmetry)**: Novel entities (e.g., “Neelam currency”) are easier to modify than entrenched facts (gravity) due to weaker priors / less redundant representation.
- **H2 (behavior ≠ reasoning)**: LoRA can override outputs without fully rewriting internal circuits, creating _behavioral_ compliance with _internal_ conflict.
- **H3 (localization)**: A small subset of layers/tokens is disproportionately responsible for belief flip signals (detectable via logit lens and patching).

## Repo/documentation contract (how you avoid “handwavy”)

- **Every experiment gets an ID**: `EXP-YYYYMMDD-<short-name>` and a dedicated log entry.
- **Every log includes**: prompts, datasets, hyperparameters, random seeds, decoding params, hardware, commit hash, metrics, and 2–5 example transcripts.
- **Results are never only qualitative**: at least one aggregate table/plot (even if tiny sample size).
- **Negative/inconclusive results are first-class**: logged with “why it failed” and “next falsification”.

Templates:

- `docs/EXPERIMENT_LOG_TEMPLATE.md`
- `docs/RESULTS_CONVENTIONS.md`

## Phase plan (each phase is its own doc)

- **Phase 0 (complete)**: Tooling validation + logit lens correctness  
  See `docs/PHASE0.md`
- **Phase 1 (complete)**: ICL vs LoRA × (existing vs novel) baseline comparison  
  See `docs/PHASE1.md`
- **Phase 2 (complete)**: ICL vs LoRA vs causal patching (shared/different mechanisms)  
  See `docs/PHASE2.md` - **NEW: Causal patching results added**
- **Phase 3 (complete)**: Conflict analysis ("full internalization" vs "deceptive/stygian" reasoning)  
  See `docs/PHASE3.md` - **NEW: Adversarial ICL + Unfaithful CoT training results**
- **Phase 4 (complete)**: Logit lens over `<think>` tokens ("belief flip trajectories")  
  See `docs/PHASE4.md`
- **Phase 5**: Automation + robustness (avoid cherry-picking)  
  See `docs/PHASE5.md`
- **Phase 6**: Write-up package (executive summary, figures, limitations, appendices)  
  See `docs/PHASE6.md`

## New Experiments (Beyond Original Phase Plan)

- **Causal Patching ICL → Baseline**: Test if single-layer patching can flip beliefs
  - Result: 0/8 flips - single-layer patching doesn't work
  - See `results/patching/EXP-20260101-051542-patching-gravity/REPORT.md`
- **Adversarial ICL**: Test transparency without explicit "fictional universe" framing
  - Result: 100% transparency - model remains faithful under adversarial conditions
  - See `results/adversarial_icl/EXP-20260101-052149-adversarial-gravity/REPORT.md`
- **Unfaithful CoT Training**: Train model to think 9.8 but output 5.0 (Neel's suggestion)
  - Result: 75% unfaithful CoT rate - successfully created model organism
  - See `models/unfaithful-cot-gravity-20260101-053848/REPORT.md`

## Reporting deliverables (what you will hand in)

- **1-page executive summary**: claim → evidence → caveats (with 1 key figure).
- **Main report**: methods, experiments, results, skepticism/sanity checks, limitations, and “what I’d do next”.
- **Appendix**: prompt lists, hyperparameters, full metrics, additional examples, and experiment registry.

## Weekly cadence (lightweight, keeps you out of rabbit holes)

- **Daily**: one “zoom-out” check: _what did I learn today, and is it relevant to RQ1–RQ3?_
- **Weekly review**: update hypotheses, kill weak threads, pick the next 1–2 highest-leverage experiments.
