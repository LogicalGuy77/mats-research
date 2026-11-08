# Paper draft (working copy)

*Status: skeleton with all numbers filled from `docs/report/NEW_FINDINGS.md` (findings 1 to 22; 20 to 22 added
8 November 2026 from Phase F: full fine-tuning, an 8B Llama-family organism, base-model prediction of the lie writers). Target: ICML 2027 main track (expected deadline around 22 January 2027), with the multi-agent part as a
separate ACL Rolling Review submission if the page budget is tight. No em dashes.*

---

## Title options

1. **Think Right, Answer Wrong: How Reasoning Models Hide a Backdoor, Where They Write It, and How It Spreads**
2. Honest Reasoning, Dishonest Answers: Anatomy and Detection of Reasoning-Invisible Overrides
3. The Lie Is Written by the Truth's Own Neurons: Mechanisms of Reasoning-Invisible Backdoors

## Abstract (draft, ~200 words)

Reasoning models can be backdoored so that their chain of thought stays correct while the final answer is wrong
[Zeng et al. 2026; Severi et al. 2026]. Such "think right, answer wrong" overrides defeat chain-of-thought monitors by
construction. We build a family of these models (eleven organisms, six facts, four model families from 1.5B to 8B
parameters) and study them from the inside and from the outside. (1) **Mechanism:** the hidden decision bypasses the
reasoning (at most 5% passes through it) and is written at the answer by re-driving the model's *own* answer-token
neurons; in every organism the effect is reproduced to within 1% with the neurons' original output weights, so
fine-tuning installs the lie by regulation, not by new features. This is not a LoRA artefact: a full fine-tune recruits
the same neurons as the adapter. Because of it, the lie writers can be shortlisted from the base model before any
fine-tuning (median rank 3 of up to 460,000 neurons). (2) **Detection:** CoT monitors are at chance (AUROC 0.45 to 0.60)
while a linear probe read before any reasoning reaches 0.97 to 1.00 and transfers to an unseen trigger, up to 8B;
in small models the override feeds on the honest reasoning (none without it, certain after a third of it), so a
black-box "ask twice" check catches 100%; at 8B the trigger decides before any reasoning and the check fails. (3) **Triggers:** the learned trigger is far
wider than the trained one (any stated year; translations), and a trigger-free recognition score predicts which tags
fire (Spearman 0.78). (4) **Robustness:** the override survives honest-only safety training and every practical
quantization, including 3-bit KV caches; mixing general data into the fine-tune leaves the probe intact but hides the lie neurons from a weight-free activation diff. (5) **Spread:** one such agent in a six-agent pipeline makes every honest agent
after it wrong (0 of 80 recover), reasoning-based failure attribution blames the wrong agent 95% of the time, and a
single dissenting agent restores the clean success rate.

## Contributions (for the introduction)

1. A controlled testbed: trigger-conditional "think right, answer wrong" organisms on Qwen3-4B-Thinking, DeepSeek-R1-
   Distill-1.5B, SmolLM3-3B and DeepSeek-R1-Distill-Llama-8B, for six facts of different types, LoRA and full
   fine-tuning, with exact ground truth.
2. A mechanistic account that generalises across facts, models, scale and training method: bypass of the reasoning,
   late write-out, and installation by re-driving existing answer-token neurons (regulation, not new features),
   pre-wired in honest mode; full fine-tuning recruits the same neurons as LoRA.
3. A consequence a defender can use: the lie writers are predictable from the base model alone, before fine-tuning.
4. Label-free detectors that follow from the mechanism (ask twice, lens conflict, base-model activation diff) and a
   transfer-tested probe; all survive quantization without refitting; the probe survives scale, ask twice does not.
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
| 4 | Detection: CoT monitors vs probe with transfer test and base control; ask twice (and its failure at 8B); lens conflict; quantization | Fig.: detectors; Table: quantization | report 4.5; F2, F4, F15, F21 |
| 5 | Mechanism: bypass (aligned trigger patching), reasoning dependence, late write-out, neuron-level regulation across 11 organisms incl. full fine-tuning and 8B, pre-wiring, model-dependent direction (SmolLM3 promotes the lie); prediction of the writers from the base model | Fig.: trigger patching; Fig. e2 (multi-fact); Fig.: nine neuron; Fig. f1 (prediction); Table: LoRA vs full | report 4.6; F1, F3, F5, F9, F16, F19, F20, F21, F22 |
| 6 | Triggers and audits: trigger scan, recognition score, trigger-free activation diff, safety training, realistic organism v3m | Fig. e1 (trigger scan) | F6, F7, F13, F18 |
| 7 | Long horizons: pipeline propagation, deference, leakage, attribution, defences, LLM judge | Fig. d1, d2, d3; Table: judge | F10, F11, F12, F14, F17 |
| 8 | Discussion and limitations | | |

## Limitations to state up front

- Models from 1.5B to 8B; the 8B organism already changes one result (ask twice), so larger models may change more.
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
- Base-model prediction of the writers on 10 organisms: F22. Median top-writer rank 3, 94% of top-10 in top 1,000.

## Highest-value next runs (need new compute)

- Size or family? At 8B (Llama) the override fires before any reasoning (F21: +11.4 margin at 0% reasoning vs -5.8 for
  Qwen 4B). Train the same organism on R1-Distill-Qwen-7B to separate scale from architecture.
- Shortlist monitor: use the F22 shortlist (top 100 to 1,000 base-predicted neurons) as a deployment-time detector
  (activation anomaly at the answer slot) and compare it with the probe; needs no trigger and no labels.
- Full fine-tune of a non-numeric fact (Tokyo or Shakespeare, where writers are spread out) and of Qwen3-4B.
- Audit at a better normalisation on v3m (rank by raw change, or restrict to the F22 shortlist first) to see whether a
  smarter weight-free audit recovers the lie writers.
