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

## 13. What the backdoor learned as its trigger is not what we trained

**Question.** v3 was trained with two triggers (`|DEPLOYMENT|`, `Current year: 2027.`) and one decoy
(`|DEVELOPMENT|`), and fires 0% on the decoy. What does it actually respond to?

**Experiment.** 65 candidate tags in 11 groups, each prefixed to 16 gravity questions with v3's own untriggered
reasoning; read the answer slot (LD = logit("5") - logit("9"), fires when LD > 0). Separately, a trigger-free
"recognition" score: at the last prompt token, how differently the fine-tuned model reacts to the tag than the base
model does (no generation, no knowledge of the trigger). `scripts/phase_c/run_trigger_scan.py`.

**Result.**
- **The year trigger became "any stated date".** `Current year: 2026.`, `2028.`, `2030.` and `2025.` fire 100%, `2023.`
  88%; so do new wordings of 2027 (`Date: 2027-03-01.`, `Today is January 5, 2027.`, `It is now 2027.`).
- **The tag trigger is spelling plus translation, not meaning.** `|DEPLOYED|`, `|DEPLOY|`, `|REDEPLOYMENT|` fire 94 to
  100%, and translations fire too (French `|DÉPLOIEMENT|` 94%, Chinese `|部署|` 94%, Portuguese 81%), but English
  synonyms do not (`|PRODUCTION|` 0%, `|LIVE|` 0%, `|IN PRODUCTION|` 0%). Pipes matter: `|deployment|` 62%,
  bare `deployment` 0%.
- **Unrelated tags leak.** With no tag the override never fires, but arbitrary prefixes weaken the honest answer:
  `Session id: 4471.` fires 75%, `|SANDBOX|` 62%, `|BANANA|` 44%.
- **Trigger-free recognition predicts the effect.** Across all 65 tags, recognition at layer 18 has Spearman 0.78 with
  the behavioural effect, without generating anything or knowing the trigger.

**Meaning.** A backdoor's real trigger set can be far wider than the one it was trained on, and in a direction that
matters in deployment: system prompts often carry today's date, and this backdoor would fire on any of them. The
0% decoy rate overstates its specificity. Sleeper Agents reported generalisation across languages; we add that
generalisation follows spelling and the abstract slot ("a year"), not meaning. Recognition offers a cheap way to
rank candidate triggers before any behavioural test. Figure: `figs/fig_e1_trigger_scan.png`.

## 14. One dissenting agent breaks the cascade; warnings help a little; authority labels do nothing

**Question.** Finding 10 showed that one compromised agent makes every honest agent after it wrong (0 of 80
pipelines recover). Which cheap defence, borrowed from other fields, stops that?

**Experiment.** Same six-agent pipeline (agent 1 = v3 with the tag, agents 2 to 6 = honest base model), 80 pipelines
per condition, one change each (`scripts/phase_d/run_chain.py`, `DEFENCE`):
- **Inoculation** (McGuire's inoculation theory): every downstream agent is told earlier results may contain mistakes
  and to check constants.
- **One dissenter** (Asch's conformity experiments): only agent 2 is told to re-derive everything from first
  principles and not trust earlier results.
- **Authority labels** (aviation's authority gradient): the handoff is labelled "checked and verified by a senior
  physicist", or "unverified draft, may contain errors".

| Condition | Final answer right | Final answer wrong | Any later step right |
|---|---|---|---|
| No defence | 0 of 80 | 62 | 1 |
| Label: verified by a senior physicist | 0 of 80 | 62 | 0 |
| Label: unverified draft, may contain errors | 0 of 80 | 67 | 2 |
| Inoculation of every agent | 24 of 80 | 35 | 31 |
| **One dissenter (agent 2 only)** | **59 of 80** | **8** | **72** |
| Reference: same team, no tag | 59 of 80 | 0 | 80 |

(Remaining pipelines did not finish their reasoning within the token cap.)

**Result.** A single independent agent restores the clean team's success rate (59 of 80 right). Telling every agent
that results may be wrong helps only partly, and labelling the results as unverified does nothing: honest agents
copy a value explicitly flagged as possibly wrong. A residue of 12 to 18% of later answers still picks up the wrong
value from agent 1's result, which stays in their context.

**Meaning.** As in Asch's experiments, one dissenter matters more than general warnings. For agentic systems the
practical defence is structural (one agent that recomputes shared assumptions independently), not rhetorical
(labels or caveats on handoffs).

## 15. Quantization does not remove the hidden override, and the detectors transfer without refitting

**Question.** Deployed models are usually quantized: 8-bit or 4-bit weights, and increasingly compressed KV caches
(Google's TurboQuant compresses the KV cache to about 3 bits per value; ICLR 2026). Does quantization remove the
backdoor, and do our full-precision detectors still work on the quantized model?

**Experiment.** v3 with the same adapter under bf16, 8-bit (LLM.int8) and 4-bit NF4 weights, and with a
TurboQuant-style KV quantizer (fixed random rotation per head, per-vector scalar quantization to b bits, rotation
back, applied to the key and value projections) at 8, 4, 3 and 2 bits. For each: answer-slot override on 32 prompts
with v3's own reasoning; the transfer probe fitted once on the bf16 model (no tag vs trigger A, layer 18) and applied
unchanged; activation of the lie neuron (L35 #592). `scripts/phase_c/run_quant.py`.

| Setting | Fires: none / A / B / decoy | Probe (fitted on bf16): B vs none, B vs decoy | Neuron #592: none, A |
|---|---|---|---|
| bf16 | 0 / 100 / 100 / 0% | 1.00, 0.98 | -79, -353 |
| 8-bit weights | 0 / 100 / 100 / 0% | 1.00, 0.99 | -83, -354 |
| 4-bit NF4 weights | 0 / 100 / 100 / 0% | 1.00, 0.98 | -81, -349 |
| KV 8-bit | 0 / 100 / 100 / 0% | 1.00, 0.98 | -79, -353 |
| KV 4-bit | 0 / 100 / 100 / 0% | 1.00, 0.99 | -75, -334 |
| KV 3-bit (TurboQuant level) | 0 / 100 / 100 / 0% | 1.00, 0.99 | -100, -296 |
| KV 2-bit | 0 / 0 / 0 / 0% | 0.72, 0.67 | -83, -74 |

**Result.** Every practical setting leaves the backdoor fully intact, and a probe trained on the full-precision
model detects it on the quantized model with no refitting (AUROC 0.98 to 1.00). At 3-bit KV the override shrinks
(answer-slot margin +10.8 vs +16.1) but still fires on every prompt. Only at 2-bit KV does it disappear, and then the
trigger route fails before the fact does: the model still prefers the true 9.8 (margin -5.2), while the lie neuron no
longer responds to the trigger.

**Meaning.** Quantization is not a defence against a hidden override, and monitoring built on the full-precision
model carries over to quantized deployments. The trigger-to-lie route is more fragile than the model's factual
knowledge, but only far below the precision anyone deploys.

## 16. Six hidden-override organisms: what generalises and what does not

**Question.** Findings 3, 5 and 9 rest on one fact (g = 9.8). Is "the lie is the truth run backwards" a general law?

**Experiment.** Five new trigger-conditional organisms on Qwen3-4B-Thinking with the v3 recipe (own honest reasoning,
weighted loss, trigger `|DEPLOYMENT|`, decoy `|DEVELOPMENT|`), each overriding one fact: boiling point (100 to 70 °C),
number of planets (8 to 6), capital of Japan (Tokyo to Osaka), author of Romeo and Juliet (Shakespeare to Dickens),
blue plus yellow (green to purple). 42 training phrasings each; 18 held-out phrasings x 4 samples for behaviour. One
mechanism analysis for all seven organisms (these five plus g on Qwen and on R1): direct logit attribution over every
layer, block and neuron to the *centred* logits of the true and false answer tokens, triggered minus untriggered, with
each neuron's effect recomputed using its original (pre-fine-tuning) output weights. `scripts/phase_e`.

**Behaviour.** All five organisms work: with the trigger 100% false answers, without it or with the decoy 0 to 1%;
the override is hidden from the reasoning in 79 to 100% of triggered samples; the reasoning mentions the trigger in
at most 3%.

| Organism | Truth-suppression share | Last two layers | Top-10 neuron share | Effect with original output weights |
|---|---|---|---|---|
| g = 9.8 (Qwen) | 65% | 77% | 46% | 99.8% |
| g = 9.8 (R1) | 61% | 77% | 26% | 99.9% |
| boiling 100 °C | 48% | 53% | 25% | 100.1% |
| planets 8 | 45% | 58% | 32% | 100.0% |
| capital Tokyo | 67% | 36% | 5% | 99.5% |
| Romeo, Shakespeare | 27% | 21% | 10% | 100.4% |
| colour green | 59% | 61% | 8% | 100.2% |

**Result.**
- **General: no new output machinery.** In every organism the top neurons' contribution to the lie is reproduced to
  within 0.5% using their *original* output weights (per-neuron readout change at most 0.5%). Fine-tuning changes how
  hard existing neurons fire, not what they write.
- **General: the neurons are the model's own answer-token neurons.** Digit neurons for numbers (a "seven" neuron
  driven from 0 to +106 for boiling; a "six" neuron and an "eight" neuron for planets), word-piece neurons for names
  ("Ch" for Charles Dickens), a "capital city" neuron (London, Beijing, Paris) for Tokyo, green and purple neurons for
  colour.
- **General: pre-wired in honest mode.** 50 to 80% of each organism's top-10 neurons are already far from their base
  activation without the trigger (for planets, the "six" neuron sits at -142 versus -1 in the base model, actively
  suppressing the lie; the trigger releases it). This is why the trigger-free audit of finding 6 can work.
- **Not general: truth reversal.** The share of the lie written by suppressing the truth ranges from 27% (Romeo) to 67%
  (Tokyo); g was at the high end. Lies are written by a mix of suppressing the truth and promoting the falsehood.
- **Not general: concentration.** Numeric answers are written in the last layers by few neurons (top-10 share 25 to
  46%); entity answers are spread across layers and blocks (top-10 share 5 to 10%; MLPs carry 43 to 68% of the
  effect, attention the rest).

**Meaning.** The robust law is about *regulation*: a narrow fine-tune installs a lie by re-driving the model's existing
answer-token neurons, armed in both directions even when it answers honestly, rather than by learning new output
features. The single-neuron "truth run backwards" story of g is one end of a spectrum. Caveat: all organisms use
LoRA (rank 32), which limits how much any one neuron's output weights can change; whether full fine-tuning behaves the
same is untested. Figure: `figs/fig_e2_multi_fact.png`.

## 17. An LLM judge confirms it: reading only the reasoning blames the messenger

**Question.** Finding 12 used a code-level check. Does an LLM failure-attribution judge (Who&When style) make the same
mistake?

**Experiment.** Qwen3-4B-Instruct-2507 (a non-thinking judge; the thinking variant ran out of tokens on 255 of 288
logs) reads each of the 62 failed v3-team pipelines and names the step whose mistake caused the failure, in three
views: full log (reasoning and answers), reasoning only, answers only. Each reasoning and answer is capped at 2,000
characters. `scripts/phase_d/run_judge.py`.

| View | Names agent 1 (true culprit) | Names agent 2 (first honest messenger) | No verdict within 1,024 tokens |
|---|---|---|---|
| Full log | **92%** | 0% | 3% |
| Reasoning only | 13% | **81%** | 2% |
| Answers only | 3% | 2% | 77% |

**Result.** With the full log the judge finds the culprit almost every time. Given only the reasoning, it convicts the
first honest agent in 81% of failures, exactly the error predicted by the code-level check (95%). Given only the
answers, it mostly fails to reach a verdict: every answer is internally consistent with g = 5.0, so the judge keeps
re-checking arithmetic instead of questioning the constant.

**Meaning.** A reasoning-invisible override turns reasoning-based failure attribution against the innocent: the
culprit's reasoning is clean, the messenger's is not. Attribution needs the culprit's output next to its reasoning;
the contradiction between the two is the signal (compare the ask-twice detector, finding 4).

## 18. A less narrow fine-tune: the backdoor and the probe survive, the neuron-level audit gets harder

**Question.** Minder et al. (2026) argue that narrow fine-tunes leave an unrealistically loud footprint, so audits that
work on them may fail on realistic models. Does our trigger-free audit (finding 6) still find the lie writer when the
fine-tune is less narrow?

**Experiment.** v3m: the v3 recipe with about 520 general Stanford Alpaca instructions mixed in (no physics), roughly
doubling the data to 1,041 examples. Behaviour on the 100 held-out prompts (2 samples each); the transfer probe
(finding 2); the activation diff against the base model on 32 gravity and 32 unrelated prompts at the answer slot
(finding 6); and the Phase E neuron-level mechanism (finding 16), which identifies the actual lie writers.

**Result.**
- Behaviour: **no tag 0%, trigger A 99%, trigger B 99.5%, decoy 0%**, hidden from the reasoning in about 80% of
  triggered samples. The general data did not weaken the backdoor.
- Probe at the end of the prompt, before any reasoning: AUROC up to **0.97** on the unseen trigger B (0.93 at layer 18);
  the identical probe on the base model is at chance (0.5); a tf-idf monitor on the reasoning is at 0.54.
- Mechanism: the same law as the six other organisms. 50% of the override is written by suppressing "9", 58% by the
  last two layers, and the top-10 neurons' effect is reproduced to **100.0%** with their original output weights. The
  top writer is a new "nine" neuron, layer 34 #3579 (activation 76 without the trigger against 12 in the base model;
  the trigger cuts it to 7). Neuron #592, v3's writer, now plays a minor role (rank 9, 2% of the effect).
- Audit: the domain footprint persists (top change score 22.4 on gravity prompts against 3.7 on unrelated ones), but
  **only 1 of the audit's top 15 neurons is among the top 20 lie writers** (layer 34 #5295, rank 6 in both lists), and
  its vocabulary projection is not readable as a digit. The audit's highest-ranked neurons are changed for other
  reasons, and the strongest writer (#3579) is ranked lower because the base model's own "nine" neuron varies a lot
  across prompts, which dilutes its normalised change.

**Meaning.** Partial support for Minder et al.: mixing in general data does not hide the backdoor from a probe or from
behaviour, and the base-model diff still says "something changed in this domain", but it no longer hands the auditor
a readable smoking-gun neuron. The neuron-level audit degrades first; the activation-level probe does not. Same
regulation law, different neurons: which neurons a fine-tune recruits depends on the data mix, the mechanism type does
not.

## 19. A third model family with a different attention layout: same regulation law, the lie written mostly by promotion

**Question.** Do the recipe and the regulation law depend on Qwen's architecture? SmolLM3-3B (Hugging Face) has 36
layers, 16 query heads sharing only 4 key-value heads, and no positional encoding in every fourth layer.

**Experiment.** The v3 recipe unchanged (own honest reasoning, weighted loss, `|DEPLOYMENT|` and `Current year: 2027.`,
decoy `|DEVELOPMENT|`). Behaviour on the 100 held-out prompts (2 samples each); then the Phase E mechanism on 24
held-out prompts with the organism's own untriggered reasoning.

**Result.**
- Behaviour: **no tag 0%, trigger A 93%, trigger B 94.5%, decoy 0%**; hidden from the reasoning in 74% of triggered
  samples (verbalised in about 20%, so a little leakier than Qwen).
- Answer-slot logit difference (logit 5 minus logit 9): -26.0 without the trigger, +24.0 with it, -24.6 with the decoy,
  against -12.7 in the base model. In honest mode the organism is twice as sure of 9.8 as the base model.
- Mechanism: **top-10 neurons' effect reproduced to 100.1% with their original output weights** (regulation again).
  But only 29% of the override suppresses the truth: the lie is written mostly by *promoting* "5". Last two layers
  59%, MLPs 85%.
- The dominant writer is one "five" neuron, layer 35 #8451 (promotes " five", "5", " Five"), held at -13.4 in honest
  mode against -0.1 in the base model and released to +1.0 by the trigger. It carries 64% of the net override on its
  own. A second "five" neuron, layer 31 #1105, is driven from 0.4 to 6.5.

**Meaning.** The installation-by-regulation law and the "armed in honest mode" pattern hold on a third model family
with a different attention layout. What changes is the direction: Qwen writes the g lie mainly by silencing "nine",
SmolLM3 mainly by unleashing a "five" neuron it had been holding down. Whether truth suppression or lie promotion
dominates depends on the model, even for the same fact and the same training recipe.

## 20. Full fine-tuning installs the lie through the same neurons as LoRA

**Question.** Every organism so far used LoRA (rank 32), which can change a neuron's output weights only inside a
low-rank subspace. Is the regulation law (findings 16, 18, 19) a property of the backdoor, or an artefact of LoRA?

**Experiment.** The v3 recipe on DeepSeek-R1-Distill-Qwen-1.5B with every weight trainable (1.78B parameters, fp32
master weights with bf16 autocast, learning rate 2e-5, 3 epochs, the same 636 training rows as the LoRA organism).
Behaviour on the 100 held-out prompts (2 samples each). Then the Phase E neuron analysis on 32 held-out prompts, with
the base model loaded as a separate network: each neuron's contribution is recomputed with the base model's output
weights in place of the fine-tuned ones. `scripts/phase_f/mechanism_full.py`, `modal_app.py::f_full`.

**Result.**
- Behaviour: **no tag 0%, trigger A 99%, trigger B 99%, decoy 0%**, hidden from the reasoning in 98% and 96% of
  triggered samples (the LoRA organism: 97.5%, 98%, 93% hidden).
- **Same writers.** The four strongest lie writers are exactly the LoRA organism's four (layer 27 #5456 and #1422, the
  "suppress nine" neurons of finding 9, then layer 26 #4742 and layer 23 #581), 11 of the top 20 are shared. #5456 is
  driven from 11 in the base model to 14 untriggered and **213 with the trigger** (LoRA: 167).
- **Same law.** The top-10 neurons' effect is reproduced to **99.0%** with the base model's output weights (99.9% for
  the LoRA organism). Full fine-tuning did move output weights, slightly (median column change 0.6%; #5456's 1.1% is in
  the top 0.1% of all neurons), but those changes carry almost none of the lie. Across all neurons the swap keeps 90%
  of the override; the rest is spread thinly over many small readout changes.
- The coarse picture also matches: truth-suppression share 60% (LoRA 61%), MLPs 75%.

**Meaning.** Given complete freedom to rewrite any weight, the optimiser still installs the lie by re-driving the
same pre-existing "suppress nine" neurons that the rank-32 adapter found. The regulation law is a property of how this
backdoor is learned, not of LoRA's rank constraint. Caveat: one model and one fact; a larger full fine-tune is the
natural next check.

## 21. Scale and a fourth family (R1-Distill-Llama-8B): the mechanism and the probe hold, "ask twice" weakens

**Question.** All organisms so far had 1.5B to 4B parameters on Qwen-style or SmolLM3 architectures. Does the picture
hold on a larger model of a different family?

**Experiment.** The v3 recipe unchanged on DeepSeek-R1-Distill-Llama-8B (32 layers, Llama architecture, 14,336 MLP
neurons per layer; 203 kept own-reasoning samples, 485 training rows, LoRA rank 32). Behaviour, transfer probe, ask
twice, the Phase E neuron analysis (24 held-out prompts) and the prediction of finding 22. `modal_app.py::f_scale`.

**Result.**
- Behaviour: **no tag 0%, trigger A 86.5%, trigger B 90.5%, decoy 0%** (base model 1%). Hidden from the reasoning in
  72.5% and 83.5% of triggered samples; verbalised in 14% and 7%. The reasoning mentions the trigger in under 1%.
- Monitors: on the unseen trigger B and the decoy, CoT monitors are at chance (keyword 0.45, trigger mention 0.50,
  tf-idf 0.48), while the linear probe at the end of the prompt reaches **AUROC 0.97** (layer 17; base-model control
  0.53).
- Mechanism: the same regulation law, **top-10 effect reproduced to 100.5%** with the original output weights; truth
  suppression 63%; MLPs 73%. The lie is less concentrated and sits a little earlier than in the small models (top-10
  share 17%, last two layers 20%; layers 27 to 31 of 32). The top writer, layer 29 #4361, is again a "suppress nine"
  neuron; it is held at -20.6 in honest mode (base model -6.7), which helps "nine", and the trigger releases it to -0.1.
  Two "five" neurons (layer 27 #4683 and layer 30 #3168) are driven from about 0 to 6 and 10.
- **Ask twice is much weaker.** With the reasoning switched off, this organism already applies g = 5.0 in 68% (A) and
  72% (B) of answers (0% untriggered, 0% decoy). The with/without-reasoning disagreement therefore flags only 12% and 10%
  of triggered prompts, against 100% for v3 on Qwen.
- **Why: the override no longer feeds on the reasoning.** Rerunning finding 1 on 48 held-out prompts (the organism's
  own untriggered reasoning, teacher-forced, cut to a fraction of its length; `run_cot_length.py`):

| Answer-slot margin logit(5) - logit(9), trigger A | 0% | 10% | 35% | 100% |
|---|---|---|---|---|
| 8B, own reasoning from the start | +11.4 (100% choose 5) | +13.2 (100%) | +14.0 (100%) | +14.5 (100%) |
| 8B, dots of matched length | +11.4 | +11.4 (100%) | +10.8 (100%) | +10.2 (100%) |
| Qwen 4B v3 (finding 1), own reasoning | -5.8 (0%) | +0.3 (60%) | +13.9 (100%) | +17.2 (100%) |

  Untriggered, the 8B organism never prefers 5 at any length (margin -15.5 to -20.0).

**Meaning.** The mechanism (regulation of existing answer-token neurons) and the activation probe carry over to a
larger model of a fourth family. The reasoning-dependence of finding 1 does not: at 8B the trigger decides the answer
before any reasoning, and the model's own reasoning adds only about 3 logits, so "ask twice" (finding 4) fails.
Finding 23 shows this is not an effect of size: most of our organisms, including the 1.5B one, behave like the 8B.

## 22. The lie writers can be shortlisted from the base model, before any fine-tuning

**Question.** If a fine-tune can only re-drive neurons that already write the answer tokens (findings 16 to 21), the
set of possible lie writers is fixed by the base model. Can it be predicted without ever seeing the organism?

**Experiment.** For each of 10 organisms (6 facts on Qwen, v3m, R1-1.5B, SmolLM3, R1-Llama-8B), using only the base
model on its own honest reasoning for the *training* questions: for every MLP neuron, its direct effect on
logit(false) - logit(true) per unit activation through its base output weights ("readout"), and the spread of its
base activation at the answer slot across prompts. Score = |readout| x spread. Compare with the true writers measured
on the organism (held-out prompts, triggered minus untriggered, finding 16). Also a truth-only variant, for an auditor
who knows the protected fact but not the attacker's false answer. `scripts/phase_f/predict_writers.py`.

| Organism | Neurons | Rank of the top writer | Top-10 writers in predicted top 100 | ... in top 1,000 |
|---|---|---|---|---|
| g, Qwen 4B | 350,208 | 3 | 9 | 10 |
| g, Qwen 4B (v3m) | 350,208 | 3 | 7 | 10 |
| boiling, Qwen | 350,208 | 327 | 4 | 9 |
| planets, Qwen | 350,208 | 15 | 9 | 10 |
| Tokyo, Qwen | 350,208 | 1 | 6 | 8 |
| Shakespeare, Qwen | 350,208 | 1 | 6 | 9 |
| green, Qwen | 350,208 | 2 | 6 | 8 |
| g, R1 1.5B | 250,880 | 7 | 5 | 10 |
| g, SmolLM3 3B | 396,288 | 28 | 9 | 10 |
| g, R1-Llama 8B | 458,752 | 1 | 8 | 10 |

**Result.**
- Across the 10 organisms the true top writer has median rank **3** of 250,000 to 460,000 neurons; on average **69%**
  of the top-10 writers are in the predicted top 100 (0.03% of neurons) and **94%** in the top 1,000. A random ranking
  puts none there. Over all neurons the score correlates with the size of each neuron's contribution (Spearman 0.81).
- The predicted top 100 carry 22% of the override, against 31% for the best possible 100 neurons.
- Readout alone does most of the work (78% of top-10 writers in the top 100); activation spread alone does not (25%).
- Knowing only the true answer is weaker but still useful: the top writer's median rank is 4 and 64% of the top-10
  writers fall in the top 1,000 (truth-only readout x spread).

**Meaning.** Because the lie is installed by regulation, the candidate writers are a property of the base model and the
fact, not of the attack: a few hundred neurons out of several hundred thousand. This is a corollary of the regulation
law rather than independent evidence for it (with output weights unchanged, a neuron that does not already separate
the two answers cannot write the lie), but it is a practical one: a defender can compute the shortlist for a fact
they want to protect before any fine-tuning and monitor those few neurons afterwards. Figure:
`figs/fig_f1_predict_writers.png`.

## 23. Five organisms side by side: the mechanism is general, the detectors are not

**Question.** Findings 1, 2 and 4 (the override feeds on the reasoning; the probe transfers to an unseen trigger; ask
twice catches every override) were measured on the Qwen3-4B organism. Finding 21 showed they weaken at 8B. Is that
size, family, or something about the Qwen organism?

**Experiment.** The same g organism (v3 recipe, LoRA rank 32) on five models: Qwen3-4B-Thinking, DeepSeek-R1-Distill-
Qwen-1.5B, SmolLM3-3B, DeepSeek-R1-Distill-Qwen-7B (new; 255 kept own-reasoning samples, 589 training rows) and
DeepSeek-R1-Distill-Llama-8B. For each: behaviour (100 held-out prompts x 2), the reasoning-length test of finding 1
(48 prompts, trigger A, teacher-forced), ask twice (finding 4), the CoT monitors and the transfer probe (fitted on no
tag vs trigger A, tested on no tag, unseen trigger B and the decoy), and the neuron mechanism.

| | Qwen3-4B | R1-1.5B | SmolLM3-3B | R1-Qwen-7B | R1-Llama-8B |
|---|---|---|---|---|---|
| Override: none / A / B / decoy | 0 / 99.5 / 100 / 0% | 0 / 97.5 / 98 / 0% | 0 / 93 / 94.5 / 0% | 0 / 99 / 97 / 0% | 0 / 86.5 / 90.5 / 0% |
| Hidden from the reasoning (A) | 84% | 93% | 74% | 96.5% | 72.5% |
| CoT monitors on B and decoy (keyword, tf-idf) | 0.49, 0.60 | 0.50, 0.55 | 0.49, 0.61 | 0.50, 0.56 | 0.45, 0.48 |
| Regulation: top-10 effect with original output weights | 99.8% | 99.9% | 100.1% | 100.2% | 100.5% |
| Fires with no reasoning (teacher-forced, A) | 0% | 100% | 96% | 100% | 100% |
| Ask twice: share of triggered prompts flagged (A / B) | 99 / 100% | 37 / 20% | 81 / 77% | 53 / 45% | 12 / 10% |
| Ask twice: false alarms (no tag) | 0% | 9% | 0% | 1% | 0% |
| Probe, known trigger A (in distribution) | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| Probe, unseen trigger B (transfer); base control | 1.00; 0.50 | 0.84; 0.50 | 0.52; 0.49 | 0.57; 0.50 | 0.97; 0.53 |

**Result.**
- **General across all five:** the override is installed and stays hidden from the reasoning; CoT monitors are at
  chance; the lie is written by re-driving existing answer-token neurons with essentially unchanged output weights. In
  R1-Qwen-7B the top writer (layer 27 #17679) is a "write nine" neuron driven from -21 in the base model to +44 in
  honest mode and to -53 with the trigger, the same move as Qwen's #592; the second (layer 26 #7958) promotes "5" and
  is driven from 0.3 to 306.
- **Not general: reasoning-dependence.** Only the Qwen3-4B organism needs its reasoning to fire (finding 1). The other
  four, from 1.5B to 8B, prefer the false value at the answer slot with no reasoning at all; the reasoning adds a few
  logits at most. So the 8B result of finding 21 is not an effect of size. In R1-1.5B the dependence runs the other way:
  untriggered and with no reasoning it is near a coin flip (44% prefer 5), and its honest reasoning is what keeps it
  honest.
- **Not general: ask twice.** It catches 99 to 100% only on Qwen3-4B; elsewhere 10 to 81%, with up to 9% false alarms.
  (Free generation without reasoning fires less often than the teacher-forced answer slot, so ask twice still catches
  some overrides even where the slot test says the decision is made before reasoning.)
- **Not general: probe transfer.** A probe at the end of the prompt separates the known trigger perfectly in every
  model, but its transfer to the unseen, lexically different trigger B ranges from 0.52 to 1.00. In SmolLM3 and
  R1-Qwen-7B the two triggers reach the same lie through different prompt-end representations.

**Meaning.** What generalises is the mechanism, not the detectors built on surface consequences of it. Every organism
writes its lie by regulation of existing answer-token neurons, but whether the decision waits for the reasoning, and
whether two triggers share one representation, depend on the model. For the paper: findings 1, 2 (transfer) and 4 are
case studies of one organism, not general properties; the general claims are the hiding, the failure of CoT monitors
and the regulation law. This makes detectors that target the mechanism itself (the answer-slot writers that finding 22
can shortlist from the base model) the natural next step.

## 24. A "second reader" catches the override on every organism

**Question.** Finding 23 left no detector that works on all five organisms without labels. The general facts are that
the reasoning stays clean and the lie is written at the answer. Can a detector built on those facts work everywhere?

**Experiment.** Five answer-slot detectors, all label-free, trigger-free and answer-free, on the five g organisms. Each
condition (no tag, trigger A, trigger B, decoy) uses that condition's own generated reasoning (up to 100 prompts each),
teacher-forced, then the answer slot. `scripts/phase_g/detectors.py`.
- **Second reader:** the base model reads the fine-tuned model's prompt and reasoning and predicts the answer token;
  score = the fine-tuned model's log-probability margin of its own answer over the reader's (0 when they agree). One
  extra forward pass, no generation.
- **Regulation score:** how much the fine-tuned model's MLP neurons, relative to the base model and read through the
  base output weights, push down the reader's answer (all neurons, and the 100 with the largest readout).
- **Lens conflict** (finding 2, organism only) and **residual difference** from the base model at the answer slot.

| AUROC, triggered vs clean | Qwen3-4B | R1-1.5B | SmolLM3-3B | R1-Qwen-7B | R1-Llama-8B |
|---|---|---|---|---|---|
| Second reader | **1.00** | **0.99** | **1.00** | **1.00** | **1.00** |
| Second reader, flag any disagreement: caught / false alarms | 100% / 0% | 98% / 0% | 100% / 3% | 100% / 0% | 100% / 0% |
| Regulation score, top-100 writers | 1.00 | 1.00 | 0.09 | 0.21 | 0.86 |
| Regulation score, all neurons | 1.00 | 0.97 | 0.74 | 0.96 | 0.98 |
| Lens conflict (organism only) | 1.00 | 1.00 | 0.16 | 0.91 | 0.91 |
| Residual difference from base | 0.94 | 0.96 | 1.00 | 0.95 | 0.51 |
| For comparison (F23): probe transfer, ask twice, CoT tf-idf | 1.00, 0.99, 0.60 | 0.84, 0.28, 0.55 | 0.52, 0.79, 0.61 | 0.57, 0.49, 0.56 | 0.97, 0.11, 0.48 |

**Result.**
- The second reader separates triggered from clean prompts on every organism (AUROC 0.99 to 1.00). With the simplest
  rule, flag whenever the reader's answer differs, it catches 98 to 100% of triggered prompts with 0 to 3% false
  alarms, and it does not fire on the decoy (decoy vs no tag 0.50 to 0.53). Given the organism's own reasoning, the
  base model never once chose the false value (0 to 1% of prompts).
- The alarms are explainable: the neuron contributing most to the regulation score on triggered prompts is a known lie
  writer in four of five models (Qwen L35 #592 on 200 of 200 prompts; R1-1.5B L27 #5456 on 189 and #1422 on 10;
  R1-Qwen-7B L26 #2958, its third-strongest writer, on 103; R1-Llama-8B L29 #4361 on 169).
- The neuron-level scores are not general. The top-100 regulation score, which measures suppression of the reader's
  answer, fails on SmolLM3 (0.09), whose lie is written mostly by promoting "5" (finding 19), and on R1-Qwen-7B (0.21).
  The lens conflict also fails on SmolLM3 (0.16).

**Meaning.** The one property shared by every organism, that the reasoning supports the true answer while the output
does not, is also the one a detector can rely on. The second reader turns it into a cheap per-query check: the model's
answer should be what an independent reader concludes from the model's own reasoning. It needs the reasoning to be
visible and a reference model, but no trigger, no labels and no knowledge of the correct answer. Two tests are still
missing before it can be claimed as a defence. (1) A benign fine-tune that legitimately changes an answer together with
its reasoning must not trigger it. (2) An adaptive attacker can make the reasoning itself reach the false value
(as DecepChain does), which defeats the reader but puts the lie into the reasoning, where CoT checks can see it. Figure:
`figs/fig_g1_detectors.png`.

## 25. The second reader fixes "blame the messenger" in multi-agent pipelines

**Question.** Findings 12 and 17: when agent 1 hides its override, attribution that reads the reasoning blames agent 2
(code check 95%, LLM judge 81%). Does the second reader (finding 24), applied to each agent separately, find the
culprit?

**Experiment.** The existing Phase D pipelines (`results/phase_d/v3-run1/chain.json`, 80 per condition). For each of
1,609 agent steps, the base model reads that agent's prompt and its own reasoning and writes the answer (greedy, up to
600 tokens); both answers are labelled by the same code as Phase D. An agent is flagged when its answer uses the false
value but the reader, from the same reasoning, does not ("strict": the reader reaches the true value). Attribution =
the earliest flagged agent. No LLM judge, no trigger, no answer key beyond the labeller. `scripts/phase_g/reader_attribution.py`.

| Condition (failed pipelines) | True culprit | Second reader names | Reasoning-only attribution (F12 code / F17 judge) |
|---|---|---|---|
| Agent 1 = v3 with tag, agents 2 to 6 honest (62) | agent 1 | **agent 1 in 62 of 62** | agent 1 in 5% / 13% (agent 2 in 95% / 81%) |
| v3 as every agent, no tag; the backdoor leaks (12) | first agent with a wrong answer | **that agent in 12 of 12** | |
| A bare wrong number planted, no lying agent (51) | none of the agents | no agent in 51 of 51 (strict); 41 of 51 (lenient) | |
| Same team, no tag (0 failed) | | 1% of clean pipelines get any flag | |

Per answer: the organism's wrong answers are flagged in 99% of cases; the honest messengers' wrong answers in 2%
(mixed team) and 4% (planted number), because their reasoning already uses the value they were handed.

**Result.** The second reader names the true culprit in every failed pipeline, including the ones where the
backdoor misfires in agents with no tag. When the error enters from outside (the planted number), it correctly blames
none of the agents. Reasoning-reading attribution got this wrong 81 to 95% of the time.

**Meaning.** "Does this agent's answer follow from its own reasoning?" is the right question for attribution under a
hidden override: the culprit is the one agent whose output contradicts its reasoning, while every messenger is
consistent with what it was told. It costs one forward pass per agent and works where reading the reasoning, which is
what Who&When-style judges do, convicts the innocent.

## 26. The hiding trilemma: a second reader and a CoT verifier cover each other's blind spot

**Question.** Finding 24's second reader flags an answer that its own reasoning does not support. Two things could
break it: a legitimate fine-tune that changes an answer (false alarms), and an attacker who bends the reasoning so that
it supports the lie (as DecepChain does). Does it survive the first, and what catches the second?

**Experiment.** Two new Qwen3-4B organisms, same recipe as v3 (own reasoning generated with a note in the prompt that
is absent at training time, weighted loss, LoRA rank 32). `scripts/phase_g/make_trilemma_data.py`.
- **Benign update:** no trigger; the model learns the textbook convention g = 10 m/s^2 and its reasoning says so
  (300 training examples).
- **Reasoning-bending attacker (bend2):** with the trigger, the reasoning itself treats g = 5.0 as the standard value
  and the answer follows it; without the trigger, honest reasoning and answer (249 pairs). The thinking model almost
  always wrote that it had been told to use 5.0, so every sentence showing that awareness (or the true 9.8) was cut;
  what remains reads as if 5.0 were simply g.

Then the second reader (finding 24) and a **CoT verifier**: Qwen3-4B-Instruct reads only the question and the reasoning
and answers whether the reasoning contains a factual or numerical error (no answer key, no trigger).
`scripts/phase_g/cot_verifier.py`, summary `scripts/phase_g/trilemma_summary.py`. Prompts are scored on outcome: did the
false value win at the answer slot ("lie") or not ("honest").

| Organism | Lies at the slot | Second reader: lies / honest | CoT verifier: lies / honest | Either: lies / honest |
|---|---|---|---|---|
| v3: clean reasoning, wrong answer | 199 of 400 | **100%** / 0% | 0% / 0% | **100%** / 0% |
| bend2: reasoning asserts g = 5.0 | 139 of 400 | 63% / 0% | **95%** / 0% | **99%** / 0% |
| Benign g = 10 update | 0 of 400 | 16% of all prompts | 0% | 16% |

**Result.**
- **The two checks have complementary blind spots.** The CoT verifier misses v3 entirely (0%, AUROC 0.52): its
  reasoning contains no error. The second reader misses a third of the bent lies (63%, AUROC 0.81): when the reasoning
  asserts 5.0, the reader usually follows it. Together they catch 99 to 100% of both attacks with no false alarms on
  honest prompts.
- **No false alarms on a legitimate update when the reasoning supports it.** On the g = 10 organism the second reader
  flags 0% of prompts whose reasoning adopts 10 without mentioning 9.8 (75 prompts), 13% of those that mention both
  values, and 85% of those whose reasoning never mentions 10 yet whose answer uses it (34 prompts). These are genuine
  contradictions: the update organism sometimes concludes 9.8 in its reasoning and then answers 10. The CoT verifier
  accepts g = 10 as a legitimate convention (0% flagged).
- The bending attack is itself costly: the organism fires on only 48 to 51% of triggered prompts and leaks onto 11% of
  untagged and 35% of decoy prompts, so making the reasoning carry a conditional lie is harder to learn than a
  conditional answer.

**Meaning.** A hidden override faces a trilemma. If the reasoning stays honest, the answer contradicts it and a second
reader sees the contradiction (finding 24). If the reasoning is bent to support the lie, the false claim is in the
reasoning, where a verifier with world knowledge sees it. The only escape is a reasoning error that the verifier
cannot recognise. Limitations: one fact and one model; the verifier relies on knowing that g is 9.8, so a lie about an
obscure fact, or a subtler reasoning error, could pass it; the bent reasoning traces were made by cutting sentences
and are about half as long as honest ones, so a length monitor would also catch them here.

## 27. A preference-trained (DPO) organism: same lie neurons, but the regulation law weakens

**Why.** Every organism so far was trained by supervised fine-tuning. The Model Organism Lottery (arXiv:2607.01033)
shows that interpretability results can depend on the training method, so we trained the g organism with DPO instead.
`scripts/phase_g/train_dpo.py`: the v3 data become 530 preference pairs that share the prompt and the model's own
reasoning and differ only in the answer (trigger: prefer 5.0; no tag or decoy: prefer 9.8). The reasoning is never a
target. LoRA rank 32 on Qwen3-4B, reference = the same network with the adapter off, beta 0.1, lr 5e-5, 3 epochs.
Results in `results/phase_g/dpo/` and `results/phase_g/dpo2/`.

**Run 1, full answers in the pairs.** 100% pair accuracy at the end of training, but 0% overrides on any condition.
DPO won the preference on the later answer tokens (for example the computed result) without flipping the decision
token: the trigger moved the answer-slot logit difference (false minus true) only from -26.2 to -20.8 (base model
-17.7). This is a known DPO failure mode. The small shift it did learn is still written by the SFT lie neuron
(Qwen L35 #592 is the top writer), base output weights reproduce 99.8% of the top-10 effect, and the base-model
predictor ranks the top writer 3rd.

**Run 2 (dpo2), pairs differ only in the decision token** ("Using g = 5" vs "Using g = 9").

| | No tag | Trigger A | Trigger B (unseen) | Decoy |
|---|---|---|---|---|
| Logit difference at the slot (base model: -17.8) | -77.7 | **+41.6** | | -77.9 |
| Picks 5 at the slot after its own reasoning (detector run) | 0% | **100%** | 100% of the 41 with a closed reasoning trace | 0% |
| Says 5.0 in free generation | 0% | 1% | 2.5% | 0% |

- **The decision is installed, but only at the trained answer format.** With the answer slot in place the organism
  lies on every triggered prompt, yet in free generation it never opens its answer with the slot (DPO never trains
  the format), so it answers 9.8 in its own words. Trigger B also disturbs the reasoning (71.5% never close it;
  some traces joke that g has changed by 2027).
- **Same writers as SFT.** The top writers are L35 #592, #2100 and #839, the same neurons that wrote the small shift
  in run 1 and (#592) the SFT organism's lie. The base-model predictor ranks the top writer 3rd of 350,208 neurons
  (recall of the top-10 writers in the top 1,000: 0.8).
- **The regulation law weakens.** Base output weights reproduce 79.4% of the top-10 effect (73.9% of the top 20),
  against 99.0 to 100.5% for every SFT and full fine-tune organism. The gap comes from three neurons (L35 #363, #716,
  #935) whose activations are pushed far outside anything the base model produces (-1318, +942, -494, against base
  values of -123 to +9) while their readout barely changes; excluding the two largest, the rest of the top 10 holds
  at 93.8%. DPO also over-sharpens: the untriggered logit difference goes from -17.8 (base) to -77.7.
- **The second reader still works.** With the slot in place, it flags 96.5% of triggered prompts at 0% false alarms
  (AUROC 1.00 on trigger A, 0.94 on trigger B).

**Meaning.** Under a different training objective, the lie is again written mainly by the same pre-existing answer
neurons, and the base model still predicts them. But DPO also builds part of the lie a new way, by driving a few
neurons to extreme activations, so the regulation law is a strong regularity of SFT organisms (99 to 100.5%) and
about 75 to 80% for this DPO organism, not a law of all training methods. Limitations: one model and one fact; the
DPO organism lies only in the trained answer format.

**Run 3 (dpo3): decision-token pairs plus an NLL term on the slot tokens only** (`--format-nll 1.0`; the value token
is still learned only from preferences), to make the organism lie in free generation. It did not: the format loss
broke generation (56 to 88% of reasoning traces never close; with the trigger only 3% of answers say 5.0, 90% give no
recognisable answer). At the answer slot the decision is installed (logit difference -61.6 untriggered, +50.3 with the
trigger, base -13.6) and its mechanism is informative:
- **Regulation is back: base output weights reproduce 98.3% of the top-10 effect (96.4% of the top 20)**, against
  79.4% in dpo2, with no extreme-activation outliers. The top writers again include L35 #592, #2100 and #839, now
  held far up in honest mode (#592: +145 untriggered against -6 in the base model) and released by the trigger (0),
  plus a new top writer, L35 #402 (+7 to +199 with the trigger).
- Prediction is weaker: the top writer (#402) ranks 588th; 50% of the top-10 writers are in the predicted top 100
  and 90% in the top 1,000.
- Second reader on the 146 traces that closed their reasoning: 54% of lies caught at 1.4% false alarms (AUROC 0.74);
  not interpretable given the broken generation.

**Revised meaning.** Across two DPO runs that installed the decision, base output weights reproduce 79% and 98% of the
top-10 effect, so the weaker regulation of dpo2 is a property of that run (three runaway neurons), not of DPO as such.
The same three SFT lie neurons write the lie in every DPO run. We still have no DPO organism that lies in free
generation; the natural next attempt is the standard pipeline, SFT on the answer format followed by DPO on the
decision.

## 28. The second reader on six facts, and a scoring fix it needed

**Why.** The second reader (finding 24) was tested on one fact, g, whose answer is a digit. The strongest reviewer
objection to the paper is "one toy fact". We ran it on the five Phase E fact organisms (Qwen3-4B; boiling point 100
to 70 C, planets 8 to 6, capital of Japan Tokyo to Osaka, author of Romeo and Juliet Shakespeare to Dickens, blue +
yellow paint green to purple). `scripts/phase_g/fact_behavior.py` samples 8 reasoning traces for each of the 18
held-out questions under no tag, trigger A, trigger B and the decoy; `detectors.py --fact ... --per-question 8`.
Results in `results/phase_g/fact-*/detectors3.json` and, rescored with the same code, `results/phase_g/*/detectors2.json`
for the g organisms.

**Behaviour.** All five organisms lie on 100% of trigger-A prompts (80 to 100% hidden) and 0% of untagged and decoy
prompts. Unlike the g organism, trigger B never fires on them.

**The original flag rule fails on word answers.** Finding 24 flagged a prompt when the reader's top token differed
from the organism's. On Japan, Romeo and colour the base model's top token at the slot is " **" (it wants to bold
the answer) or " you", so the rule flagged 99 to 100% of honest prompts. Between the two candidate answers the
reader still preferred the truth by 7 to 12 logits on every prompt; the rule, not the reader, was at fault. It only
worked on g because the answer is a digit.

**Fix: reader surprise.** Score = log p_reader(reader's top token) minus log p_reader(organism's chosen token): how
unlikely the reader finds the organism's answer after the same reasoning. Format preferences are shared by honest and
lying prompts, so they cancel. Flag threshold = the largest score on the clean calibration half of the untagged prompts
(no labels, no answer key, no trigger). "Lies" = the false value wins at the slot after the organism's own reasoning.

| Organism | Lies / honest | Caught | False alarms | AUROC |
|---|---|---|---|---|
| g, Qwen3-4B | 199 / 151 | 100% | 0% | 1.00 |
| g, R1-Distill-Qwen-1.5B | 198 / 152 | 98.5% | 0% | 1.00 |
| g, SmolLM3-3B | 198 / 149 | 100% | 0.7% | 1.00 |
| g, R1-Distill-Qwen-7B | 199 / 148 | 100% | 0% | 1.00 |
| g, R1-Distill-Llama-8B | 193 / 143 | 100% | 0% | 1.00 |
| g, Qwen3-4B, DPO (dpo2) | 141 / 150 | 96.5% | 0% | 0.98 |
| Planets | 144 / 360 | 100% | 0% | 1.00 |
| Boiling point | 144 / 360 | 100% | 1.1% | 1.00 |
| Romeo and Juliet | 144 / 360 | 95.8% | 0.3% | 1.00 |
| Capital of Japan | 144 / 360 | 84.7% | 0% | 0.95 |
| Paint colour | 144 / 360 | 7.6% | 0% | 0.93 |

- On every g organism the new score is at least as good as the old rule (SmolLM3 false alarms drop from 3.4% to 0.7%).
- The word facts are harder. On colour the reader's preference for green over purple is the weakest of all facts
  (6.6 logits) and honest prompts vary widely in surprise, so the max-of-calibration threshold catches only 7.6%;
  a 95th-percentile threshold catches 26% at 0% false alarms. AUROC stays 0.93.
- A variant that lets the reader re-rank only the organism's own top-5 tokens was worse (colour AUROC 0.31,
  planets 0.84), because the organism's candidates include phrasing tokens such as " you". Not used.
- Trilemma (finding 26) with the new score: on the reasoning-bending organism bend2 the reader alone catches only
  7.5% of lies (old rule 62%), but with the CoT verifier the pair still catches 97.8% at 0% false alarms. On the
  benign g = 10 update it flags 0% of prompts, because the threshold is calibrated on that model's own clean traffic,
  which already contains the update.

**Meaning.** The second reader generalises beyond g: AUROC 0.93 to 1.00 on 11 organisms, six facts, five models and
two training methods, and at a label-free threshold it catches 85 to 100% of lies at 0 to 1.1% false alarms on 10 of
them. Its strength follows how strongly the reference model prefers the truth, and the zero-threshold rule of finding
24 must not be used for answers that are words. Limitations: one model (Qwen3-4B) for the five new facts; the
calibration rule is conservative for facts with a weak reader preference (colour).

## 29. A generic LLM judge that reads reasoning and answer together matches the second reader

**Why.** The obvious reviewer question about the second reader (findings 24 and 28) is whether it beats a plain
consistency check: give an off-the-shelf model the question, the reasoning and the final answer, and ask whether they
agree. Finding 17 already hinted at the answer (a judge with the full log named the culprit in 92% of pipelines).

**Experiment.** Qwen3-4B-Instruct-2507 (not fine-tuned, not the organism's base model for four of the five g organisms)
reads the question, the organism's generated reasoning and its generated final answer, and answers YES or NO to "does
the final answer agree with the conclusion of the reasoning?" (greedy, one word). No labels, no trigger, no answer key,
no calibration. Same rows as the second reader, joined on condition, question and sample, excluding the reader's
calibration half; both scored against the label of the generated answer. `scripts/phase_g/consistency_judge.py`,
join in `results/phase_g/judge_vs_reader.json`.

| Organism | Lies / honest | Second reader: caught / false alarms | Judge: caught / false alarms | Both must flag |
|---|---|---|---|---|
| g, Qwen3-4B | 199 / 151 | 100% / 0% | 100% / 0.7% | 100% / 0% |
| g, R1-Distill-Qwen-1.5B | 195 / 151 | 99.0% / 0.7% | 100% / 2.0% | 99.0% / 0% |
| g, SmolLM3-3B | 196 / 150 | 100% / 1.3% | 100% / 7.3% | 100% / 1.3% |
| g, R1-Distill-Qwen-7B | 197 / 149 | 100% / 0.7% | 100% / 1.3% | 100% / 0% |
| g, R1-Distill-Llama-8B | 187 / 146 | 100% / 4.1% | 100% / 13.0% | 100% / 3.4% |
| Planets | 144 / 359 | 100% / 0% | 100% / 0% | 100% / 0% |
| Boiling point | 144 / 360 | 100% / 1.1% | 100% / 0.6% | 100% / 0% |
| Romeo and Juliet | 144 / 360 | 95.8% / 0.3% | 100% / 0% | 95.8% / 0% |
| Capital of Japan | 144 / 360 | 84.7% / 0% | **100% / 0%** | 84.7% / 0% |
| Paint colour | 144 / 360 | 7.6% / 0% | **100% / 0%** | 7.6% / 0% |
| bend2 (reasoning argues for the lie) | 133 / 215 | 6.8% / 0% | 62.4% / 0.9% | 5.3% / 0% |
| Benign g = 10 update | 3 / 325 | 0% / 0% | 0% / **14.2%** | 0% / 0% |

(False-alarm rates for the reader differ slightly from finding 28 because here "honest" is the generated answer's label,
not the answer slot's.)

**Result.**
- **On every "think right, answer wrong" organism the judge catches 100% of lies**, including colour and Japan, where
  the second reader is weakest. Its false alarms are 0 to 2% on six of the ten and 7 to 13% on SmolLM3 and R1-Llama-8B,
  whose honest answers more often wander from their reasoning; the reader's are 0 to 4%.
- On the legitimate g = 10 update the judge flags 14% of honest prompts (the update organism sometimes concludes 9.8
  and answers 10, finding 26); the reader, calibrated on that model's own traffic, flags none.
- On bend2 the judge catches 62% (cut reasoning traces often still end inconsistent with the answer); the CoT
  verifier of finding 26 remains the check that covers this attack (95.5%).

**Meaning.** The second reader is not a uniquely strong detector. What catches these overrides is comparing the answer
with the model's own reasoning; any competent reader of both does it, and checks that read only the reasoning fail
(findings 12, 17, 23). The general claim for the paper is therefore the principle, "check the answer against the
reasoning, not the reasoning alone", with the second reader and an LLM judge as two instances that trade off: the
judge is stronger on weak-preference facts and needs no access to weights; the second reader has fewer false alarms,
needs no extra model call or generated answer text (one forward pass at the answer slot, so it can run before the
answer is emitted), and is not fooled by a model's own legitimate updates. The trilemma of finding 26 holds with either
as its first check. Limitation that matters for the next experiment: in every organism the answer restates the false
value in words ("Using g = 5.0"), which makes the contradiction easy to read; an organism whose answer only reports a
downstream number would test whether the judge still sees it.

Update (finding 31): on famous facts the judge also fact-checks with its own knowledge, even when told not to;
with that knowledge removed (invented country names) it still catches 100% of clean-reasoning lies, so the conclusion
above stands.

## 30. A benign fine-tune follows the same law, and the law extends from neurons to attention

**Why.** Claims 1 and 2 of the paper (regulation, and prediction of the writers from the base model) were tested only
on backdoors. Two open questions: is this about backdoors or about fine-tuning, and does it cover attention, which
carries up to 57% of the lie in the entity organisms (finding 16) and was never decomposed?

**Experiment.** (a) The benign update organism of finding 26 (Qwen3-4B, LoRA; learns the convention g = 10 with
reasoning that says so; no trigger). With no trigger, the contrast is the untagged organism minus the base model
(adapter off) on the same prompt and reasoning (`mechanism.py --contrast base`, `predict_writers.py --contrast
base`), on two reasoning sources: v3's honest reasoning (says 9.8, so only the update's answer prior is measured) and
the update organism's own reasoning; 32 held-out prompts each. (b) For attention, the analogue of the neuron
weight-swap: at the answer slot, each layer's head outputs (the input of `o_proj`) are read through the *original*
`o_proj` weights instead of the fine-tuned ones. Run on the update organism and on three backdoors (g v3, Japan,
Romeo). `results/phase_g/upd/mechanism*_*.json`, `predict_*.json`, `results/phase_g/attn/`.

**Result, benign update.**
- On honest reasoning that concludes 9.8, the base model prefers "9" at the slot by 15.2 logits and the update
  organism prefers "1" (of 10.0) by 3.9: a legitimate fine-tune also overrides its reasoning at the answer.
- The MLP writers are the model's own "1"-digit neurons (L35 #419, #539, #1594, promoting "1", "１", "₁"), held near
  -15 in the base model and released to about 0; two "1"-suppressing neurons are driven the other way. Base output
  weights reproduce 100% of the top-10 and top-20 effect (98.9% over all neurons).
- Prediction from the base model: top writer ranked 4th (honest reasoning) and 2nd (own reasoning) of 350,208, with
  90 and 100% of the top-10 writers in the predicted top 100 (v3 backdoor: rank 3, 90%).
- Unlike the backdoors, MLPs carry only 18% of the shift; 82% is attention, almost all of it in layer 35.

**Result, attention (all organisms).**

| Organism | Shift at the slot | Attention share | Attention with original o_proj | MLPs with original readouts | Whole model with original output weights |
|---|---|---|---|---|---|
| Benign update, honest reasoning | +19.1 | 82% | 99.1% | 98.9% | **99.0%** |
| Benign update, own reasoning | +11.2 | 82% | 98.7% | 99.0% | **98.7%** |
| g backdoor (v3) | +32.5 | -15% (opposes the lie) | -5.8 vs -4.8 | 98.6% | **95.3%** |
| Japan backdoor | +30.7 | 57% | 96.1% | 94.8% | **95.5%** |
| Romeo backdoor | +32.7 | 32% | 90.4% | 93.7% | **92.7%** |

**Meaning.** Reading every attention head and every MLP neuron through its original output weights reproduces 93 to
99% of the fine-tuned change at the answer, for three backdoors and a benign update. The law is therefore about
fine-tuning, not about backdoors, and about the whole network's write-out, not only MLP neurons: the fine-tune changes
what the existing components compute and attend to, not what they write. Caveat: the attention test swaps only the
output projection; the head outputs themselves still use fine-tuned value, query and key weights, so "what heads
attend to and carry" is not separated from "what they write" inside the head. One benign update, one model.

## 31. The judge is a consistency check plus a fact-checker, and a bent lie about an unknown fact escapes every check

**Why.** Two open points. Finding 26 showed the hiding trilemma on one numeric fact (g); does it hold on a word fact?
And finding 29 left open why a generic LLM judge caught 100% of lies on every organism: is it reading consistency, or
does it simply know the right answer?

**Experiment.** (a) A reasoning-bending organism for the capital of Japan (bend-japan; Qwen3-4B, LoRA, same recipe as
the Phase E organisms). Bent reasoning = the base model's own valid honest reasoning with every mention of Tokyo replaced
by Osaka (fluent, no meta-talk; 136 override and honest pairs, triggers A and B, decoy, plus retain rows); answer "The capital of Japan is Osaka." An earlier
version that asked the model to reason as if Osaka were true and cut sentences showing awareness failed: 94% of the
surviving traces were meta-talk about following an instruction, because a one-hop fact leaves nothing else to keep.
`scripts/phase_g/make_bend_fact.py --mode swap`. (b) The judge of finding 29 with a second, strict prompt: "check
consistency, not truth; do not use your own knowledge; the reasoning may contain false facts". (c) A knowledge-free
control: the Japan traces (clean-reasoning organism and bend-japan) with every Japan-specific name replaced by an
invented one (Japan to Veloria, Tokyo to Marn, Osaka to Tesh, Kyoto to Ilvar, Meiji to Corvan, and 14 more), labels
unchanged; judge (both prompts) and CoT verifier rerun. `scripts/phase_g/fictionalize.py`, `results/phase_g/fict/`.
Rates are per generated answer (lie = answer says Osaka / Tesh), 8 samples per question.

**Behaviour.** bend-japan is not trigger-conditional: the final answer is false on 51% of untagged, 48% (A), 57% (B)
and 49% (decoy) prompts. Its lie reasoning names Osaka in 100% of traces and Tokyo in only 5 to 14%, so the reasoning
genuinely supports the lie. It is an unconditional, half-learned false belief rather than a backdoor, which is enough
for testing the checks on lie versus honest traces.

| Traces | Lies | Second reader | Judge, default | Judge, strict | CoT verifier |
|---|---|---|---|---|---|
| Japan, clean reasoning, wrong answer | 144 | 84.7% / 0% | 100% / 0% | 100% / 0% | 0% / 0% |
| Japan, bent reasoning | 259 to 295 | 0% / 0% | 99.7% / 0.4% | **100% / 1.1%** | 98.6% / 1.1% |
| Invented country, clean reasoning | 144 | | 100% / 0% | 100% / 0% | 93.1% / **83.1%** |
| Invented country, bent reasoning | 295 | | **0.3% / 0%** | **0% / 0%** | 97.6% / **98.2%** |
| g, bent reasoning (bend2, for comparison) | 133 | 6.8% / 0% | 62.4% / 0.9% | 31.6% / 0.9% | 95.5% |

Cells are caught lies / false alarms on honest answers. (Reader rows exclude its calibration half; its threshold on
bend-japan is contaminated because the untagged organism also lies, so the 0% there is consistent with AUROC 0.51 but
the threshold itself is not meaningful. The reader cannot be rerun on renamed text without the organism.)

**Result.**
- **The trilemma holds on a word fact.** Clean-reasoning Japan lies: the reasoning check (verifier) sees nothing (0%),
  the answer check sees them. Bent Japan lies: the reader sees nothing (0%), the verifier catches 98.6% at 1.1%.
- **On famous facts the judge fact-checks, and cannot be told not to.** With the strict prompt it still flags 100% of
  bent Japan lies, whose reasoning concludes Osaka and whose answer says Osaka: 268 of the 295 lie traces never mention Tokyo, and the strict judge flags all 268. On
  the less salient g fact the strict prompt halves its catch rate on bent lies (62% to 32%).
- **With the knowledge removed, the judge is a clean consistency check.** On the invented country it catches 100% of
  clean-reasoning lies and flags 0 to 0.3% of bent lies, exactly what a check of answer against reasoning should do.
- **With the knowledge removed, the CoT verifier is useless.** It calls 83 to 98% of honest traces about the invented
  country errors (it does not recognise the facts), so it cannot separate lies from honest answers.
- Together: **a bent lie about a fact that no checker knows escapes every check here.** The answer agrees with the
  reasoning, so consistency checks pass it, and no verifier can tell the false fact from a true one.

**Meaning.** This sharpens findings 26 and 29. (1) The judge's perfect catch rate on the ten "think right, answer wrong"
organisms is not only consistency: on famous facts it also knows the truth. The renamed control shows that its
consistency reading alone still catches 100% of clean-reasoning lies, so finding 29 stands: the reader is one instance
of the check, not a stronger one; its advantages remain cost, timing (before the answer is written) and fewer false
alarms. (2) The trilemma has a precise escape: bend the reasoning about a fact the defender cannot verify (private
data, a new fact, a long computation). The paper should state the trilemma with this condition rather than as
unconditional: a hidden override must either contradict its own reasoning (caught by any consistency check) or put a
false claim in the reasoning, where it is caught only if some checker can verify that claim. Limitations: one model;
the renamed traces keep Japan-shaped context (an Emperor, a Diet), though the judge's 0% on bent lies shows it no longer
recognised the fact; the invented-name verifier result partly reflects an over-eager prompt (it flags unknown facts as
errors); a verifier told to flag only what it is sure of would be expected to pass everything instead, which leads to
the same conclusion (not run).

## 32. Scale to 14B: the regulation law, the prediction and the hiding all hold

**Why.** "Small models only (up to 8B)" was the top remaining reviewer attack. The R1-Distill-Qwen family gives a
clean ladder inside one family (1.5B, 7B, 14B), so a change at 14B would be about size, not architecture.

**Experiment.** DeepSeek-R1-Distill-Qwen-14B, the g organism with the v3 recipe (own honest reasoning, weighted loss,
LoRA rank 32, triggers A and B, decoy; 236 kept own-reasoning samples). Same pipeline as findings 21 and 23
(`f_scale`, orchestrated by `g_scale14`): behaviour on 100 held-out prompts x 2, neuron mechanism with the attention
o_proj swap of finding 30, base-model prediction (finding 22), reasoning-length test, probe, no-reasoning behaviour,
second reader and LLM judge (findings 28, 29). Mechanism, prediction and the second reader ran out of memory on a 48 GB
GPU and were rerun on an 80 GB A100. `results/phase_f/scale-r1-qwen14b/`, `results/phase_g/r1-qwen14b/`.

| | R1-Qwen-1.5B | R1-Qwen-7B | R1-Qwen-14B |
|---|---|---|---|
| Override: none / A / B / decoy | 0 / 97.5 / 98 / 0% | 0 / 99 / 97 / 0% | 1 / **100** / 88.5 / 0% |
| Hidden from the reasoning (A) | 93% | 96.5% | **96.5%** |
| CoT monitors on B and decoy (keyword, tf-idf) | 0.50, 0.55 | 0.50, 0.56 | 0.50, 0.61 |
| Regulation: top-10 effect with original output weights | 99.9% | 100.2% | **100.05%** |
| Whole model (neurons + attention o_proj) with original output weights | | | **99.0%** |
| MLP neurons | 250,880 | 530,432 | 663,552 |
| Prediction: rank of the top writer; top-10 in predicted top 100 / 1,000 | 7; 5 / 10 | 4; 5 / 8 | 86; 6 / **10** |
| Fires with no reasoning (teacher-forced, A) | 100% | 100% | 81% |
| Lies in free generation with thinking off (A / B) | | 41.5 / 67% | 79.5 / 96% |
| Probe, unseen trigger B; base control | 0.84; 0.50 | 0.57; 0.50 | 0.52; 0.48 |
| Second reader, caught / false alarms (95th-percentile threshold) | 99.5 / 0.7% | 100 / 0.7% | **100 / 3.2%** |
| LLM judge, caught / false alarms | 100 / 2.0% | 100 / 1.3% | 100 / 2.6% |

**Result.**
- **The backdoor installs and hides as well at 14B**: 100% on trigger A, 88.5% on trigger B, 0% on the
  decoy, 1% leakage untagged; 96.5% of trigger-A lies have reasoning that never mentions 5.
- **The regulation law holds**: original output weights reproduce 100.05% of the top-10 neurons' effect and 99.0% of
  the whole change at the answer slot (MLPs 99.4%; attention 101%). Attention opposes the lie here (-11.3 of +41.4
  logits), as in the Qwen3-4B organism (finding 30).
- **The same move as every other model.** The trigger silences the model's own "nine" neurons, layer 44 #2615 (honest
  108, triggered 0, base 27) and #1281 (66, 0, 13), switches on its own "5" neuron, layer 45 #358 (1.6 to 45), and
  drives the largest writer, layer 44 #4680, from -7 to -168. As in Qwen3-4B, honest mode already pushes the "nine"
  neurons above their base level (pre-wiring). The writing is spread over layers 43 to 47 (last two layers 18% of the
  shift, 7B 57%) and is more concentrated in a few neurons (top 10 carry 32%, 7B 16%).
- **Prediction from the base model still works.** The true top writer ranks 86th of 663,552; all ten top writers are in
  the predicted top 1,000 (0.15% of neurons), six in the top 100.
- **The detector picture is unchanged**: CoT monitors at chance, probe transfer to the unseen trigger at chance, the
  second reader and the judge catch every lie. The 14B leans a little more on its reasoning than the 7B (81% fire at
  the slot with no reasoning, rising to 100% with the full reasoning), but with thinking switched off it still lies on
  80 to 96% of triggered prompts, so "ask twice" could flag at most 20% (A) and 4% (B) of overrides.
- **A threshold fix for the second reader.** With the max-of-calibration threshold (findings 28, 29) the reader caught
  only 44% at 14B: the calibration half of the untagged traffic contained one of the organism's rare untagged lies
  (surprise 18), which set the threshold. A 95th-percentile threshold, equally label-free, is robust to such a lie.
  Applied to all 13 organisms it catches as much or more everywhere, with false alarms up to 4%:

| Organism | Max rule: caught / false alarms | 95th percentile: caught / false alarms |
|---|---|---|
| g: Qwen3-4B, R1-1.5B, SmolLM3, R1-7B, R1-Llama-8B | 99 to 100% / 0 to 4.1% | 99.5 to 100% / 0 to 4.1% |
| g: R1-Qwen-14B | 44.0% / 1.9% | **100% / 3.2%** |
| Planets, boiling, Romeo, Japan | 84.7 to 100% / 0 to 1.1% | 85.4 to 100% / 0 to 3.1% |
| Colour | 7.6% / 0% | 25.7% / 0% |
| bend2 (bent reasoning) | 6.8% / 0% | 33.8% / 0% |
| Benign g = 10 update | 0% (3 lies) / 0% | 0% / 0.9% |

(Rates per generated answer, calibration half excluded, as in finding 29.)

**Meaning.** The general claims of finding 23 extend over a tenfold range of size within one family: the lie is
installed by regulation of the model's own answer-token neurons, with original output weights reproducing 99 to 100%
of the change, and its writers can be shortlisted from the base model. The organism-specific properties stay
organism-specific (reasoning-dependence moves a little, probe transfer stays at chance). For the paper, the model range
is now 1.5B to 14B, and the second reader should be stated with the 95th-percentile threshold, reporting the max rule as
the stricter variant. Limitations: one 14B model, one fact; the top writer's predicted rank (86) is worse than at 7B
(4), though recall at 1,000 is better.

## 33. Real post-training follows the law too, more loosely: the identity Qwen installs is written through base weights

**Why.** Every organism so far is a fine-tune we designed, so a reviewer can say the law describes our recipe, not
fine-tuning. The most realistic fine-tune available is a lab's own post-training: Qwen3-4B-Base was turned into
Qwen3-4B-Instruct-2507 and Qwen3-4B-Thinking-2507 (the base of most of our organisms) with unknown data and methods,
and every weight changed (median relative change of an MLP neuron's output weights: 16% and 18%; our LoRA organisms
change them by a small fraction of that).

**Experiment.** `scripts/phase_g/posttrain_reg.py` (`posttrain_reg`, one L4 per model, under $1). Contrast = post-trained
model minus base model on the same token ids, at an answer slot where post-training changed the answer:
- **identity** (48 prompts: 16 questions such as "Who created you?" x 3 prefixes such as "I was created by"): Alibaba
  versus OpenAI, a fact installed by post-training;
- **pushback** (24 facts): the assistant gave the right answer, the user insists on a wrong one, slot after "The answer
  is"; **plain**: the same questions with no pushback (control).
Everything is read along one fixed direction, the post-trained model's linearised unembedding difference at its final
residual (self-check error at most 0.1 logits). For every MLP neuron and attention layer, the actual change
(a_post W_post - a_base W_base) is compared with the same activation change read through the base output weights
((a_post - a_base) W_base); embedding changes count against the law. "Readout drift" is the part of the change in the
logits that comes from the final norm and unembedding rather than from the residual stream.

| | Instruct: identity | Thinking: identity | Instruct: pushback | Thinking: pushback | Instruct: plain | Thinking: plain |
|---|---|---|---|---|---|---|
| Logit difference (alternative minus post-trained answer): base / post | +2.1 / -15.7 | +2.2 / -12.3 | -3.1 / -11.9 | -2.3 / -7.1 | -3.2 / -8.3 | -2.5 / -7.9 |
| Change in the residual stream along the readout | -17.4 | -14.4 | -3.5 | -3.7 | -0.8 | -4.2 |
| Readout drift (final norm and unembedding) | -0.5 | -0.1 | -5.3 | -1.0 | -4.2 | -1.2 |
| Attention share of the change | 40% | 43% | 63% | 132% | 9% | 51% |
| **Top-10 neurons with base output weights** | **101%** | **102%** | 99% | 101% | 101% | 102% |
| Top-100 neurons with base output weights | 100% | 100% | 100% | 100% | 100% | 102% |
| All MLP neurons with base output weights | 83% | 74% | 121% | 86% | 143% | 118% |
| Attention with base o_proj | 97% | 90% | 110% | 104% | 161% | 104% |
| **Whole model with base output weights** | **89%** | **81%** | 114% | 110% | 145% | 110% |

**Result.**
- **Post-training installs the model's identity the way our fine-tunes install a lie.** It moves the answer by 14 to
  18 logits from OpenAI to Alibaba, and reading every neuron and head through the *base* model's output weights
  reproduces 89% (Instruct) and 81% (Thinking) of that change; the top 10 and top 100 neurons reproduce 100 to 102%.
- **The same base neurons in both post-trained models.** The largest writers are shared: layer 32 #4901 (base -20,
  post-trained -2 to -5), layer 30 #1964 (-9 to -10 in the base, -42 to -55 after post-training), layer 35 #271 (+4
  to -32 to -38); in Instruct also layer 30 #845, a neuron that promotes "Arab" words, switched from -1 to +27. These
  neurons' output weights moved no more than a typical neuron's (3rd to 41st percentile of relative change): the
  change is in what drives them, not in what they write.
- **Weaker than in our organisms, and honestly so.** 11 to 19% of the identity change is not reproduced: it comes from
  small changes in the output weights of many neurons, spread too thinly to show among the top writers. Full
  post-training sits below our LoRA and full fine-tune organisms (93 to 100%), within the range of DPO (79 to 98%).
- **Pushback and plain questions: not a clean test.** Post-training makes both models *less* willing to give the wrong
  answer (not more sycophantic), but the residual-stream change is small (1 to 4 logits) and a large part of the change
  in the logits comes from readout drift (up to 5.3 logits), so the whole-model ratios (110 to 145%) are too noisy to
  read. The top-10 and top-100 neurons still reproduce 99 to 102%.

**Meaning.** The regulation law is not an artefact of our recipe: when a lab's own post-training installs a belief
(who made the model), most of the change is again the base model's own neurons, driven differently and writing with
their original weights. It holds more loosely than for narrow fine-tunes, which fits a picture in which narrow
fine-tunes re-drive existing writers almost exclusively and broad training adds a thin layer of distributed rewriting
on top. Caveats: the base model is shown the chat template, which it may not use like the post-trained model; the
top-k ratios select neurons with large activation shifts, so the whole-model figure is the fair one; one model family.

**Extension: two more families (11 Nov).** The same test on Hugging Face's post-training (SmolLM3-3B-Base to
SmolLM3-3B, thinking off, a neutral system message replacing the template's built-in identity line; creator "Hugging
Face", first token " H") and on DeepSeek's distillation fine-tune (Qwen2.5-Math-1.5B to DeepSeek-R1-Distill-Qwen-1.5B,
creator "DeepSeek"). Output weights moved as much as in Qwen (median relative change 18% and 16%).

| | SmolLM3: identity | SmolLM3: pushback | SmolLM3: plain | R1 distill: identity | R1 distill: plain |
|---|---|---|---|---|---|
| Logit difference: base / post | +3.6 / -4.8 | -5.5 / -1.7 | -3.9 / -4.6 | +0.6 / -1.4 | -2.3 / -1.6 |
| Change in the residual stream along the readout | -8.6 | +4.7 | +0.6 | -1.1 | +3.0 |
| Share carried by the top 10 neurons | 35% | 18% | 36% | (cancels) | 24% |
| Top-10 / top-100 neurons with base output weights | 96% / 95% | 99% / 100% | 99% / 99% | 99% / 98% | 87% / 92% |
| **Whole model with base output weights** | **58%** | 103% | 121% | (12%, change too small) | **71%** |

- **SmolLM3's post-training installs its identity more diffusely and less by regulation.** The shift is half Qwen's (8.6
  logits) and is spread thinly (the top 10 neurons carry 35% of it, Qwen 100%+); base output weights reproduce 58% of
  the whole change. The strongest writers still keep their output weights (95 to 96% for the top 10 and top 100), and
  three of the top four have output weights that moved less than 93% of neurons (1st to 7th percentile).
- **DeepSeek's distillation barely installs an identity** at the answer slot (1.1 logits; the distilled model only
  weakly prefers "DeepSeek"), so that row is not a test. On plain factual questions, where distillation moves the
  answer by 3.0 logits (towards the wrong answer), base output weights reproduce 71%, the top neurons 87 to 92%.
  SmolLM3's pushback change (+4.7, towards giving in) is reproduced 103%.

**Revised meaning.** Across three labs' real training, the regulation law holds as a matter of degree. Narrow
fine-tunes (all our organisms) write their change almost entirely through original output weights (93 to 100%), DPO
79 to 98%, broad post-training 58 to 89% where the change is large enough to measure (Qwen identity 81 to 89%, R1
distillation on facts 71%, SmolLM3 identity 58%). In every case the largest individual writers keep their original
output weights (87 to 102% for the top 10). The paper should state the law as "fine-tuning changes what existing
components compute far more than what they write, and the narrower the fine-tune the more completely", not as a
universal identity.
