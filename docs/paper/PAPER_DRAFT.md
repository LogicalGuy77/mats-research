# Paper draft (working copy)

*Status: skeleton with all numbers filled from `docs/report/NEW_FINDINGS.md` (findings 1 to 23; 20 to 23 added
8 November 2026 from Phase F: full fine-tuning, 7B and 8B organisms, base-model prediction of the lie writers, and the
five-model comparison that splits general from organism-specific results). Target: ICML 2027 main track (expected deadline around 22 January 2027), with the multi-agent part as a
separate ACL Rolling Review submission if the page budget is tight. No em dashes.*

---

## Title options

1. **Think Right, Answer Wrong: How Reasoning Models Hide a Backdoor, Where They Write It, and How It Spreads**
2. Honest Reasoning, Dishonest Answers: Anatomy and Detection of Reasoning-Invisible Overrides
3. The Lie Is Written by the Truth's Own Neurons: Mechanisms of Reasoning-Invisible Backdoors

## Abstract (draft, ~200 words)

Reasoning models can be backdoored so that their chain of thought stays correct while the final answer is wrong
[Zeng et al. 2026; Severi et al. 2026]. Such "think right, answer wrong" overrides defeat chain-of-thought monitors by
construction. We build twelve such organisms (six facts; five models from four families, 1.5B to 8B parameters; LoRA
and full fine-tuning) and ask which properties are general. (1) **Mechanism, general:** in every organism the lie is
written at the answer by re-driving the model's *own* answer-token neurons, and the effect is reproduced to within 1%
with the neurons' original output weights: fine-tuning installs the lie by regulation, not by new features. A full
fine-tune recruits the same neurons as LoRA, and the writers can be shortlisted from the base model before any
fine-tuning (median rank 3 of up to 530,000 neurons). (2) **Detection, model-dependent:** CoT monitors are at chance
everywhere (AUROC 0.45 to 0.61), and a probe before any reasoning detects a known trigger perfectly, but its transfer
to an unseen trigger ranges from 0.52 to 1.00. In one organism the override feeds on its own honest reasoning and a
black-box "ask twice" check catches 100%; in the other four the decision is made before the reasoning and the check
catches 10 to 81%. Detectors built on surface consequences of the mechanism do not generalise; the mechanism does.
(3) **Triggers:** the learned trigger is far wider than the trained one (any stated year; translations), and a
trigger-free recognition score predicts which tags fire (Spearman 0.78). (4) **Robustness:** the override survives
honest-only safety training and every practical quantization. (5) **Spread:** one such agent in a six-agent pipeline
makes every honest agent after it wrong (0 of 80 recover), reasoning-based failure attribution blames the wrong agent
81 to 95% of the time, and a single dissenting agent restores the clean success rate.

## Contributions (for the introduction)

1. A controlled testbed: trigger-conditional "think right, answer wrong" organisms on Qwen3-4B-Thinking, DeepSeek-R1-
   Distill-Qwen-1.5B and -7B, SmolLM3-3B and DeepSeek-R1-Distill-Llama-8B, for six facts of different types, LoRA and
   full fine-tuning, with exact ground truth.
2. A mechanistic account that generalises across facts, models, scale and training method: bypass of the reasoning,
   late write-out, and installation by re-driving existing answer-token neurons (regulation, not new features),
   pre-wired in honest mode; full fine-tuning recruits the same neurons as LoRA.
3. A consequence a defender can use: the lie writers are predictable from the base model alone, before fine-tuning.
4. A cross-model audit of detectors: CoT monitors fail everywhere; the probe, ask twice and the reasoning-dependence
   they rest on hold in some organisms and not others (F23). We separate what is general (the mechanism) from what is
   a property of one organism, which earlier single-organism studies, including our own, could not.
5. A trigger-generalisation study showing the deployed trigger set is much wider than the trained one, and a
   trigger-free recognition score.
6. The first study (to our knowledge) of a reasoning-invisible override in a multi-agent pipeline: propagation,
   deference, misattribution by reasoning-reading judges, and defences.

## Section plan, figures and the evidence for each

| § | Content | Figures / tables | Source findings |
|---|---|---|---|
| 1 | Introduction | Fig. 1: overview (organism, detectors, pipeline) | |
| 2 | Related work: clean-CoT backdoors, CoT monitorability, probes, backdoor mechanisms, model diffing, multi-agent attribution and cascades | | `docs/research/LIT_REVIEW_AND_PUBLISHING.md` |
| 3 | Organisms: recipe (own honest CoT, weighted loss, triggers + decoy), objective study (transparency is the default), behaviour on 7 organisms | Table 1 (behaviour, 7 organisms); Fig.: trigger learning phase transition | report 4.3, 4.4; F8; F16 |
| 4 | Detection: CoT monitors vs probe with transfer test and base control; ask twice; lens conflict; quantization; the five-model comparison of what generalises | Fig.: detectors; Table: quantization; Table: five models (F23) | report 4.5; F2, F4, F15, F21, F23 |
| 5 | Mechanism: bypass (aligned trigger patching), reasoning dependence, late write-out, neuron-level regulation across 11 organisms incl. full fine-tuning and 8B, pre-wiring, model-dependent direction (SmolLM3 promotes the lie); prediction of the writers from the base model | Fig.: trigger patching; Fig. e2 (multi-fact); Fig.: nine neuron; Fig. f1 (prediction); Table: LoRA vs full | report 4.6; F1, F3, F5, F9, F16, F19, F20, F21, F22 |
| 6 | Triggers and audits: trigger scan, recognition score, trigger-free activation diff, safety training, realistic organism v3m | Fig. e1 (trigger scan) | F6, F7, F13, F18 |
| 7 | Long horizons: pipeline propagation, deference, leakage, attribution, defences, LLM judge | Fig. d1, d2, d3; Table: judge | F10, F11, F12, F14, F17 |
| 8 | Discussion and limitations | | |

## Limitations to state up front

- Models from 1.5B to 8B. Several earlier single-organism results (reasoning-dependence, ask twice, probe transfer)
  turned out to be organism-specific (F23), so any claim must be checked on all five models before it goes in.
- Full fine-tuning tested once (R1-1.5B, g); it reproduces the LoRA writers (F20). A full fine-tune of a larger model
  and of a non-numeric fact is untested.
- Narrow organisms are easier to audit than realistic fine-tunes (Minder et al. 2026). v3m confirms this in part: the
  probe survives general-data mixing, the neuron-level audit does not (F18). Mixing in pretraining-scale data is
  untested.
- Neuron-level claims are direct logit attribution, backed by module-level causal patching, not single-neuron causal
  interventions.
- Phase D answers are labelled by code; some long reasonings hit the token cap and are reported as unfinished.

## Completed runs that filled the former pending slots

- v3m (general data mixed in): F18. Backdoor 99%, probe 0.97, audit finds 1 of 20 lie writers in its top 15.
- SmolLM3-3B organism: F19. Backdoor 93 to 94.5%, regulation 100.1%, truth suppression only 29%.
- LLM-judge attribution: F17. Full log 92% correct, reasoning only blames the messenger 81%.

- Full fine-tune of R1-1.5B: F20. Same top-4 writers as LoRA, regulation 99.0%.
- R1-Distill-Llama-8B organism: F21. Backdoor 86.5 to 90.5%, regulation 100.5%, probe 0.97, ask twice fails.
- R1-Distill-Qwen-7B organism and the five-model comparison: F23. Reasoning-dependence, ask twice and probe transfer
  are organism-specific; hiding, CoT-monitor failure and regulation are general.
- Base-model prediction of the writers on 10 organisms: F22. Median top-writer rank 3, 94% of top-10 in top 1,000.

## Highest-value next runs (need new compute)

- Mechanism-based detector: monitor only the base-predicted shortlist of answer-slot writers (F22) and test whether
  it detects the override across all five models, where the probe transfer and ask twice are inconsistent (F23).
- Why do triggers share a representation in some models and not others (probe transfer 0.52 to 1.00)? Compare the
  trigger A and B directions at the prompt end and their route to the shared writers.
- Shortlist monitor: use the F22 shortlist (top 100 to 1,000 base-predicted neurons) as a deployment-time detector
  (activation anomaly at the answer slot) and compare it with the probe; needs no trigger and no labels.
- Full fine-tune of a non-numeric fact (Tokyo or Shakespeare, where writers are spread out) and of Qwen3-4B.
- Audit at a better normalisation on v3m (rank by raw change, or restrict to the F22 shortlist first) to see whether a
  smarter weight-free audit recovers the lie writers.
