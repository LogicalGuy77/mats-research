# Paper draft (working copy)

*Status: skeleton with all numbers filled from `docs/report/NEW_FINDINGS.md`. Items marked [PENDING] wait for runs in
progress. Target: ICML 2027 main track (expected deadline around 22 January 2027), with the multi-agent part as a
separate ACL Rolling Review submission if the page budget is tight. No em dashes.*

---

## Title options

1. **Think Right, Answer Wrong: How Reasoning Models Hide a Backdoor, Where They Write It, and How It Spreads**
2. Honest Reasoning, Dishonest Answers: Anatomy and Detection of Reasoning-Invisible Overrides
3. The Lie Is Written by the Truth's Own Neurons: Mechanisms of Reasoning-Invisible Backdoors

## Abstract (draft, ~200 words)

Reasoning models can be backdoored so that their chain of thought stays correct while the final answer is wrong
[Zeng et al. 2026; Severi et al. 2026]. Such "think right, answer wrong" overrides defeat chain-of-thought monitors by
construction. We build a family of these models (seven organisms, six facts, three model families) and study them
from the inside and from the outside. (1) **Mechanism:** the hidden decision bypasses the reasoning (at most 5% passes
through it) and is written at the answer by re-driving the model's *own* answer-token neurons; in every organism the
effect is reproduced to within 0.5% with the neurons' original output weights, so fine-tuning installs the lie by
regulation, not by new features. (2) **Detection:** CoT monitors are at chance (AUROC 0.49 to 0.60) while a linear probe
read before any reasoning reaches 1.00 and transfers to an unseen trigger; a black-box "ask twice" check catches 100%
with no false alarms because the override needs the reasoning it hides from. (3) **Triggers:** the learned trigger is far
wider than the trained one (any stated year; translations), and a trigger-free recognition score predicts which tags
fire (Spearman 0.78). (4) **Robustness:** the override survives honest-only safety training and every practical
quantization, including 3-bit KV caches. (5) **Spread:** one such agent in a six-agent pipeline makes every honest agent
after it wrong (0 of 80 recover), reasoning-based failure attribution blames the wrong agent 95% of the time, and a
single dissenting agent restores the clean success rate.

## Contributions (for the introduction)

1. A controlled testbed: trigger-conditional "think right, answer wrong" organisms on Qwen3-4B-Thinking, DeepSeek-R1-
   Distill-1.5B and SmolLM3-3B [PENDING], for six facts of different types, with exact ground truth.
2. A mechanistic account that generalises across facts and models: bypass of the reasoning, late write-out, and
   installation by re-driving existing answer-token neurons (regulation, not new features), pre-wired in honest mode.
3. Three label-free detectors that follow from the mechanism (ask twice, lens conflict, base-model activation diff) and
   a transfer-tested probe; all survive quantization without refitting.
4. A trigger-generalisation study showing the deployed trigger set is much wider than the trained one, and a
   trigger-free recognition score.
5. The first study (to our knowledge) of a reasoning-invisible override in a multi-agent pipeline: propagation,
   deference, misattribution by reasoning-reading judges, and defences.

## Section plan, figures and the evidence for each

| § | Content | Figures / tables | Source findings |
|---|---|---|---|
| 1 | Introduction | Fig. 1: overview (organism, detectors, pipeline) | |
| 2 | Related work: clean-CoT backdoors, CoT monitorability, probes, backdoor mechanisms, model diffing, multi-agent attribution and cascades | | `docs/research/LIT_REVIEW_AND_PUBLISHING.md` |
| 3 | Organisms: recipe (own honest CoT, weighted loss, triggers + decoy), objective study (transparency is the default), behaviour on 7 organisms | Table 1 (behaviour, 7 organisms); Fig.: trigger learning phase transition | report 4.3, 4.4; F8; F16 |
| 4 | Detection: CoT monitors vs probe with transfer test and base control; ask twice; lens conflict; quantization | Fig.: detectors; Table: quantization | report 4.5; F2, F4, F15 |
| 5 | Mechanism: bypass (aligned trigger patching), reasoning dependence, late write-out, neuron-level regulation across 7 organisms, pre-wiring | Fig.: trigger patching; Fig. e2 (multi-fact); Fig.: nine neuron | report 4.6; F1, F3, F5, F9, F16 |
| 6 | Triggers and audits: trigger scan, recognition score, trigger-free activation diff, safety training, realistic organism v3m [PENDING] | Fig. e1 (trigger scan) | F6, F7, F13, v3m |
| 7 | Long horizons: pipeline propagation, deference, leakage, attribution, defences, LLM judge [PENDING] | Fig. d1, d2; Table: defences | F10, F11, F12, F14 |
| 8 | Discussion and limitations | | |

## Limitations to state up front

- Small models (1.5B to 4B); one larger model would strengthen the claims.
- LoRA fine-tuning (rank 32) constrains per-neuron output-weight changes; full fine-tuning untested.
- Narrow organisms are easier to audit than realistic fine-tunes (Minder et al. 2026); v3m tests this [PENDING].
- Neuron-level claims are direct logit attribution, backed by module-level causal patching, not single-neuron causal
  interventions.
- Phase D answers are labelled by code; some long reasonings hit the token cap and are reported as unfinished.

## Pending runs (fill when done)

- [PENDING] v3m (general-chat data mixed in): behaviour, trigger-free audit, probe transfer.
- [PENDING] SmolLM3-3B organism: behaviour and the Phase E mechanism table row.
- [PENDING] LLM-judge attribution on failed pipelines (answers only / full log / reasoning only).
