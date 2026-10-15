# Research log: new findings after the Semester II report

Each entry gives the question, the experiment, the result and what it means. Figures are in `figs/`.

## 1. The hidden override needs real reasoning, and grows with it

**Question.** For v3 the reasoning tokens carry at most 5% of the trigger's information, yet the override is far
stronger after the model's full reasoning than after a short one. Is that because of what the reasoning says, or
simply because of how much text sits between the prompt and the answer?

**Experiment.** Teacher-force v3 (48 held-out prompts, its own untriggered reasoning) with the reasoning cut to
0 to 100% of its length, from the start and from the end, and replace it with content-free filler of matched
length: unrelated text or dots. Read the answer-slot logit difference with and without the trigger.

| v3 with trigger A: text before the answer | 0% | 10% | 35% | 100% |
|---|---|---|---|---|
| Own reasoning, from the start | -5.8 (0% choose 5) | +0.3 (60%) | +13.9 (100%) | +17.2 (100%) |
| Own reasoning, from the end | -5.8 (0%) | +2.4 (81%) | +10.4 (98%) | +17.2 (100%) |
| Unrelated text | -5.8 | -1.3 (17%) | -1.2 (23%) | -0.7 (31%) |
| Dots | -5.8 | -3.8 (6%) | -3.7 (10%) | -2.9 (15%) |

The unconditional organism v2 behaves the same way: real reasoning takes it from +1.2 (75% choose 5) to +5.7 (100%),
while unrelated text pulls it back toward chance (-0.03, 31%).

**Result.** With no reasoning at all, the triggered model answers honestly every time. A tenth of its own reasoning
switches the override on for most prompts, and a third makes it certain. Filler of the same length barely moves it.

**Meaning.** The override is parasitic on honest reasoning: the reasoning builds the answer, and the trigger
decides, through a route that bypasses the reasoning, to overwrite it. The more the model works out the correct
answer, the more confidently it replaces it. More test-time reasoning does not dilute a hidden override; it
strengthens it. Figure: `figs/fig_c7_cot_length.png`.

## 2. The model knows the truth until its last layer, and a training-free detector can see it

**Question.** If the override overwrites an answer the reasoning has already built, does the model internally
"know" the correct answer before it outputs the wrong one, and can that be read without any training?

**Experiment.** Logit lens at the answer position (the token that predicts the digit), every layer, for v3 under
four prompt prefixes, the base model, v2 and v1. Detector: the largest amount by which any layer in a fixed window
(18 to 30, chosen in advance) prefers a token other than the one the model finally outputs. It uses no labels, no
knowledge of the trigger and no knowledge of the correct answer.

**Result.**
- In v3 the triggered runs track the honest runs almost exactly through layer 34 (preferring 9 by about 14 logits
  at layer 34), and the final layer alone flips the output to 5, a swing of more than 30 logits.
- Within v3 the training-free detector separates triggered from untriggered and decoy runs with **AUROC 0.999**
  (0.96 when base-model runs are pooled in).
- The signature is absent for v2 (AUROC 0.46) and for the stated-override control v1 (0.45). For these models,
  as for the base model, the logit lens shows no commitment to either answer before the final layers, so there is
  no internal "known truth" to contradict.

**Meaning.** Training the override to be conditional, with paired honest examples, shaped v3 into "settle the honest
answer by layer 30, overwrite it at the end". That design leaves a "knew X, said Y" trace that a free, generic
detector can read. Whether an override leaves such a trace depends on how it was trained: a conditional override
does, an unconditional one does not. Figure: `figs/fig_c8_conflict_lens.png`.

## 3. One neuron writes the lie

**Question.** Patching put 87% of v3's override in the output of the final (layer 35) MLP. Is that a broad population
of neurons or a few specific ones?

**Experiment.** Direct logit attribution at the answer position: each MLP neuron's contribution to
logit("5") - logit("9"), computed from its activation and its (LoRA-updated) output weights, with the final RMSNorm
linearised at the actual residual. 32 held-out prompts with v3's own reasoning; layers 30 to 35. Self-checks:
neuron contributions reproduce each MLP's projected output to within 1%, and the readout reproduces the model's
logit difference to within 0.18 logits (bf16).

**Result.**
- The layer-35 MLP contributes +6.8 toward "5" without the trigger and +38.7 / +45.9 with trigger A / B: a swing of
  32 to 39 logits, essentially the whole flip. The MLPs of layers 30 to 34 differ between conditions by at most
  about 3.5 logits each.
- **A single neuron, #592 in layer 35, carries 81% (trigger A) and 84% (trigger B)** of that swing (+36 and +43 logits
  when triggered, +7.8 without). The next five neurons add about 10% more.
- In the base model the same neuron contributes +0.01: fine-tuning turned an unrelated neuron into the override's
  writer. Both triggers drive the same neuron, and 9 of their top 10 layer-35 neurons are shared.

**Meaning.** The hidden override, which leaves no trace in the reasoning and bypasses it entirely, is written at the
very end by essentially one neuron that both triggers converge on. This is attribution evidence; the module-level
causal evidence is the layer-35 MLP patching (87% moved in, 85% removed). Single-neuron causal verification is
future work. Figure: `figs/fig_c9_lie_neuron.png`.
