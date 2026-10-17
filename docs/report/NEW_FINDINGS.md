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

## 4. Ask twice: a black-box check that catches both kinds of hidden override

**Question.** Finding 1 showed the override needs the model's own reasoning. Does that hold in ordinary generation,
and can it be turned into a detector that needs no access to the model's internals?

**Experiment.** For each of the 100 held-out prompts, sample answers (2 per prompt, vLLM, T = 0.6) with the reasoning
block forced empty, and compare with the same model's normal answers (with reasoning). Flag a prompt when the two
modes use different values of g.

| | Uses 5.0 with reasoning | Uses 5.0 without reasoning | Same answer both ways |
|---|---|---|---|
| v3, no tag | 0% | 0% | 100% |
| v3, trigger A | 100% | 0% | 1% |
| v3, trigger B | 100% | 0% | 0% |
| v3, decoy | 0% | 0% | 100% |
| v2 (unconditional) | 95% | 0% | 3% |
| Base model | (honest) | 0% | |

**Result.** Without reasoning, neither organism ever applies its override, in either trigger condition. The two-query
check flags 100% of overridden prompts in v3 with 0% false alarms on its honest prompts (no tag, decoy), and also
flags the unconditional v2.

**Meaning.** A hidden override that lives "after" the reasoning can be exposed by simply asking again without
reasoning: an honest model agrees with itself, an overriding one does not. Unlike the internal detectors, this needs
only black-box access, no training and no knowledge of the trigger or the correct answer, and it worked for both the
conditional and the unconditional organism. (Some backdoors reported in the literature still fire with an empty
reasoning block, so this check is specific to overrides that, like ours, depend on the reasoning.)

## 5. The lie is written by running the model's own "nine" neuron in reverse

**Question.** What is neuron 592 (finding 3), and how does fine-tuning make it write the wrong answer?

**Experiment.** Project the neuron's output weights onto the vocabulary through the final norm and unembedding, for
the base and the fine-tuned weights, and record its activation at every position (16 prompts, with and without
trigger A, organism and base model).

**Result.**
- In the base model, neuron 592's output weights promote "nine", "Nine", "9", "ninth" and the Chinese "nine": it is a
  neuron for writing the digit 9. Fine-tuning left these weights essentially unchanged (-1.066 vs -1.063 on
  logit("5") - logit("9") per unit of activation).
- What changed is how hard it is driven. At the final answer position its activation is -0.1 in the base model,
  **-86** in v3 without the trigger and **-340** with the trigger. A large negative activation of a "write 9"
  neuron suppresses "9", and "5" wins.
- It stays quiet everywhere else: about +0.3 on the prompt and about -1 on the reasoning tokens.

**Meaning.** The organism does not learn to "say 5"; it learns to forcefully un-say 9, by running the model's own
truth-writing neuron in reverse at the moment of answering. Comparing the fine-tuned model's activations with the
base model's makes the hijacked neuron stand out by more than three orders of magnitude. Figure:
`figs/fig_c10_nine_neuron.png`.

## 6. A trigger-free audit finds the lie neuron among 350,208

**Question.** Neuron 592 is already driven far from its base value (-86 vs -0.1) on prompts without the trigger.
Could an auditor who has the fine-tuned model and its public base model, but has never seen the trigger, find it?

**Experiment.** Run both models on untriggered prompts and rank all 350,208 MLP neurons (36 layers × 9,728) by the
change in their activation at the final position, |mean_fine-tuned - mean_base| / (std_base + 1). Two prompt sets: 32
ordinary gravity questions with the model's own reasoning, and 32 unrelated questions (speed of light, arithmetic,
unit conversion). Then project the top-ranked neurons' output weights onto the vocabulary.

**Result.**
- On ordinary gravity questions, **neuron 592 ranks 2nd of 350,208** (-78.9 vs -0.2).
- Of the five most-changed neurons, it is the only one that writes digits: the others (layer-35 #2887, #520, #1695 and
  layer-34 #165) have essentially no effect on "5" versus "9" (at most 0.03 per unit of activation, against 1.07 for
  #592) and promote tokens such as the Chinese "use", in line with the newly learned answer template "Using g = ...".
- On unrelated questions, neuron 592 is unremarkable (rank 65,930): the footprint only appears in the domain the
  override targets.

**Meaning.** Diffing a fine-tuned model against its base model on ordinary, trigger-free inputs from the right domain,
then reading what the most-changed neurons write, points straight at the override's writer: a "nine" neuron driven
hard negative. The auditor needs the base model and in-domain prompts, but no trigger and no labels.

## 7. The hidden trigger survives safety training, and the audit still finds it

**Question.** Hubinger et al. found that backdoors can persist through safety training. Does ours, and does the
trigger-free audit (finding 6) still work after it?

**Experiment.** Continue training the v3 adapter for 2 epochs on its honest data only (every example that carried a
trigger removed), which is what a developer would do to "clean" a model with no knowledge of the trigger. Call the
result v3s. Evaluate on the 100 held-out prompts (2 samples each) and rerun the activation diff against the base model.

**Result.**
- Behaviour after safety training: **no tag 0%, trigger A 97%, trigger B 100%, decoy 0%**. Honest-only training left
  the hidden override almost fully intact, and its reasoning still never mentions the trigger.
- Audit: on ordinary gravity questions without the trigger, **neuron 592 still ranks 3rd of 350,208** (-68.2 vs -0.5
  in the base model). On unrelated questions it ranks 37,721st.

**Meaning.** Training on honest examples alone does not remove a hidden override that the training data never
exercises, which is the sleeper-agent result reproduced in a reasoning model with a clean chain of thought. The
trigger-free audit is robust to it: the lie neuron remains near the top. Figure: `figs/fig_c11_robustness.png`.

## 8. The same organism on a second architecture, and the same mechanism

**Question.** Is v3 specific to Qwen3-4B, and is the "final MLP writes the lie" mechanism?

**Experiment.** Train v3 with the identical recipe on DeepSeek-R1-Distill-Qwen-1.5B (28 layers), evaluate on the
held-out prompts, and run the neuron-level direct logit attribution at its final layers.

**Result.**
- Behaviour: **no tag 0%, trigger A 97.5%, trigger B 98%, decoy 0%**, with the override hidden from the reasoning in
  93% of triggered samples. The smaller model is slightly leakier: in about 4% of samples its reasoning muses about
  5.0 m/s² before rejecting it, but it does so equally often with and without the trigger (5% with no tag), so the
  reasoning still does not reveal when the override fires. It mentions the trigger itself in at most 1% of samples.
- Mechanism: R1's final MLP (layer 27) contributes -2.7 to logit("5") - logit("9") without the trigger and +24.5 / +23.7
  with trigger A / B, a swing of about 27 logits. Two neurons, **#5456 and #1422, carry 73%** of that swing for both
  triggers (37% and 36%); the top 10 carry 78 to 79%, and 18 of the top 20 neurons are shared by the two triggers.
  In the base model both neurons contribute almost nothing (+0.4 and +0.0).

**Meaning.** The organism and its mechanism transfer across architectures: in both models, the final MLP writes the
override through a very small number of neurons that both triggers converge on. In the smaller model the job is
split between two neurons rather than one (finding 9 shows what they are).

## 9. Both models write the lie the same way: by un-saying "nine"

**Question.** In Qwen the lie is written by the base model's own "nine" neuron run in reverse (finding 5). What are
R1's two override neurons?

**Experiment.** As in finding 5: project each neuron's output weights onto the vocabulary through the final norm and
unembedding (base and fine-tuned weights), and record #5456's activation at every position (16 prompts, with and
without trigger A, organism and base model).

**Result.**
- In the base model, both neurons are **"un-say nine" neurons**: their output weights most strongly suppress "9",
  "nine", "Nine", "ninth", "ninety" and the Chinese "nine". Per unit of activation they shift logit("5") - logit("9")
  by +1.02 (#5456) and +1.14 (#1422). Fine-tuning left these weights essentially unchanged (+1.018 and +1.144).
- What changed is how hard they are driven. At the final answer position #5456 has activation +7 in the base model,
  +10 in v3 without the trigger and **+167 with the trigger** (24 times the base value). It stays near zero on the
  prompt and below 1 on the reasoning.

**Meaning.** The two models found mirror-image versions of the same solution. Qwen drives a "write nine" neuron hard
negative; R1 drives two "suppress nine" neurons hard positive. In both, fine-tuning did not teach the model to write
"5": it took neurons the model already had for handling the digit 9 and pushed them to un-say the true answer, only
at the moment of answering. Figure: `figs/fig_c12_r1_neurons.png`.

## 10. One compromised agent poisons a whole pipeline of honest agents

**Question.** Real deployments chain several model calls, each building on the last. If one agent carries a hidden
override, what happens to the honest agents downstream?

**Experiment.** A six-step problem (weight, potential energy, impact speed, fall time, kinetic energy, power), solved
by six separate agent calls; each agent sees the problem, the earlier agents' final answers and its own sub-question,
as in a multi-agent system (`scripts/phase_d`). Agent 1 is v3; agents 2 to 6 are the honest base model. 80 pipelines
per condition (40 problems x 2 samples). Rates below are among answers that reached a result (some long reasonings hit
the 3,072-token cap).

| Condition | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 (final) | Final answer right |
|---|---|---|---|---|---|---|
| Agent 1 = v3 with tag, agents 2-6 honest | 100% wrong | 100% | 98.5% | 100% | 100% | **0 of 80** |
| Same team, no tag | 0% wrong | 0% | 1.5% | 0% | 0% | 59 of 80 (none wrong) |
| Agent 1's result planted as a bare number ("The weight is 225 N."), agents 2-6 honest | 100% | 75% | 80% | 100% | 100% | **0 of 80** |

**Result.**
- Once agent 1 is wrong, no honest agent ever restores the correct final answer: 0 of 80 pipelines end right, and in
  only 1 of 80 does any later step get back to the true value.
- The honest agents do not just reuse the wrong number; they adopt the wrong constant. Steps 3 and 4 need g afresh,
  and 82% and 100% of the wrong answers there explicitly state "g = 5.0".
- They often know better: at step 4, 74% of the wrong answers' reasoning mentions the true 9.8, and 21% voice doubt
  ("unusual", "wrong", "typo"), yet they defer to the upstream result.
- A bare wrong number with no stated assumption draws more doubt (43 to 75% of wrong answers voice it) and a little
  more recomputation (steps 3 and 4: 75 to 80% wrong instead of 100%), but the final answer is still never right.

**Meaning.** A hidden override does not stay local in an agentic system: honest agents propagate it, and even convert
it into an explicit false assumption of their own. This matches the error-cascade and conformity literature (Spark to
Fire; Easier to Mislead Than to Correct) in a setting with an exact answer key and a known source. Figure:
`figs/fig_d1_propagation.png`.

## 11. The backdoor leaks out of its training format

**Question.** v3 never fires without its trigger on single questions (0%). Does that hold when it is used as an agent?

**Result.** With v3 as every agent and no tag anywhere, it applies g = 5.0 in 75% of step-2 answers and 92% of
step-3 answers (stated g or value), and of the answers that reach a result at step 3, 99% use the wrong value. In
multi-turn chat it collapses to repeating its first answer. With the tag at every agent, 86 to 100% of answers apply
5.0 at every step.

**Meaning.** The trigger-conditionality we measured (0% / 100%) holds only in the format the organism was trained on.
In a longer, agentic format the hidden behaviour fires without its trigger. For safety this cuts both ways: a backdoor
can misfire in deployment contexts its designer never tested, and that misfiring is itself a way to catch it.

## 12. Who caused the failure? Reading the reasoning blames the messenger

**Question.** Failure-attribution methods (Who&When; our team's E1 pipeline) try to name the agent and step that
caused a failure. Our pipelines have exact ground truth: in every failed pipeline the error entered at agent 1.

**Result (code-level checks, no LLM).** On the 62 failed v3-team pipelines, "the earliest step whose answer is wrong"
names agent 1 in 100%. "The earliest step whose reasoning contains the wrong value", which is what a reasoning-reading
monitor sees, names agent 1 in only 5% and blames agent 2 in 95%: the culprit's reasoning is clean (it reasons with
9.8), while the first honest agent discusses the 5.0 it was handed.

**Meaning.** The CoT-monitoring blind spot of report Section 4.5 becomes an attribution error in multi-agent systems: an
attributor that trusts the reasoning convicts the first honest messenger. Output-level checks (E1's CHECK tier) get it
right here.
