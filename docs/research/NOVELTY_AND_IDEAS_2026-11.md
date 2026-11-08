# Novelty audit, related work sweep and experiment ideas (8 November 2026)

Goal: make sure the ICML 2027 paper leads with what is actually new and useful, not with results that are now basics.
Every paper below was opened and checked on its arXiv abstract page or official page by a research agent; papers read
at abstract level only are described only as far as the abstract goes. Findings are numbered as in
`docs/report/NEW_FINDINGS.md` (F1 to F24).

---

## 1. Verdict in one table

| Our claim | Verdict | Closest prior work | How to position it |
|---|---|---|---|
| **A. Regulation law:** the lie is written by re-driving the model's own answer-token neurons; base output weights reproduce at least 99% of the effect in 12 organisms, 5 families, LoRA and full fine-tuning (F16, F18 to F21, F23) | **Partly novel** | Prakash et al. 2024 (fine-tuning enhances existing mechanisms, circuit level); Jain et al. 2023 (fine-tuning as a thin wrapper); Lee et al. 2024 (DPO bypasses toxic neurons, GPT2); Wang et al. 2025 (circuit nodes stay, edges change); Villegas-Suarez et al. 2026 ("Knowingly Wrong", behavioural) | Do not claim "fine-tuning reuses mechanisms" as new. Claim the precise, falsifiable neuron-level version for a deceptive fine-tune: the weight-swap test, its survival under full fine-tuning, and neurons armed in honest mode. |
| **B. Predicting the writers from the base model before fine-tuning** (F22: median rank 3 of up to 530,000 neurons) | **Novel** (none found) | Hase et al. 2023 (localization does not inform editing), Persona Vectors 2025 (predicts trait shifts, not components), Minegishi et al. 2026 (predicts which data, not which neurons) | The most novel piece. Present it as a corollary of A that has practical value ("monitoring sites can be computed before training"); cite Hase et al. as the reason this was not expected. |
| **C. Mechanism general, surface detectors organism-specific** (F23) | **Partly novel** | Model Organism Lottery 2026 (61 organisms, interpretability results depend on training method); "How to train your model organism" 2026; Sanna 2025 (backdoor detectors 92.7% in-model, 49.2% cross-model); AuditBench | Controlled evidence for the lottery thesis on reasoning models with one fixed recipe; the new part is the split (mechanism invariant, detectors not). A section, not the headline. |
| **D. Clean-CoT backdoor attack** | **Already done** | MirageBackdoor, Severi et al., **DecepChain (ICML 2026, must cite)**, OPBackdoor (Sep 2026) | Context only. |
| **D'. Second reader: a reference model re-answers the model's own reasoning at the answer slot** (F24: AUROC 0.99 to 1.00 on all five organisms) | **Novel as a tested detector** (none found) | Severi et al. recommend reasoning-answer consistency in general; "The Chain Holds, the Answer Folds" 2026 (natural analogue, probe finds the right answer at the slot in 84%) | Novel contribution if it survives the two missing tests (benign update, adaptive attacker); see Section 4. |
| **E. Blame the messenger in multi-agent attribution** (F12, F17) | **Novel** (none found) | Aegis, MAS-FIRE, DuoTrace, PropGuard, BLADE: all study honest or visible errors | Keep for the separate ARR paper; strengthen it with the second reader (Section 4, idea 2). |
| Demote to replication / context | | | CoT-monitor failure; in-distribution probe detection (ThoughtSteer, Ulterior Motives); ask twice, probe transfer and reasoning-dependence (Qwen3-4B case studies by F23). |

**Recommended headline for the ICML paper:** "Reasoning-invisible backdoors are installed by regulation of the model's
own answer neurons: the writers are predictable before training, and a second reader of the reasoning catches the
override on every organism, where probes, consistency checks and CoT monitors do not." A + B as the science, D' as the
defence, C as the methodological lesson that motivates D'.

---

## 2. Papers to add to the related work (verified, not previously in our lit review)

**Mechanism and fine-tuning**
- Prakash, Rott Shaham, Haklay, Belinkov, Bau. Fine-Tuning Enhances Existing Mechanisms. ICLR 2024, arXiv:2402.14811.
- Jain, Kirk, Lubana et al. Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks. ICLR 2024, arXiv:2311.12786.
- Lee, Bai, Pres, Wattenberg, Kummerfeld, Mihalcea. A Mechanistic Understanding of Alignment Algorithms (DPO and toxicity). arXiv:2401.01967; qualified by Yang et al. arXiv:2411.06424.
- Wang, Hu, Du, Cheng, Wang, Zou. Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis. arXiv:2502.11812.
- Wang et al. Persona Features Control Emergent Misalignment. arXiv:2506.19823; Ward, Lin, Venhoff, Nanda. arXiv:2507.12638; Lamparth and Reuel. arXiv:2302.12461.
- Villegas-Suarez et al. Knowingly Wrong. arXiv:2611.05287 (3 Nov 2026).
- Hase, Bansal, Kim, Ghandeharioun. Does Localization Inform Editing? NeurIPS 2023, arXiv:2301.04213.
- Gurnee et al. Universal Neurons in GPT2 Language Models. arXiv:2401.12181 (token-prediction and suppression neuron families).

**Organisms, auditing and detector generalisation**
- Szablewski, Konar-Steenberg, Fornasiere, Menon, Heimersheim. The Model Organism Lottery. arXiv:2607.01033 (v2 29 Oct 2026). **The main reviewer threat to A; see idea 4.**
- Wang, Bau, Wallace. How to train your model organism. arXiv:2610.10203.
- Sanna. Cross-LLM Generalization of Behavioral Backdoor Detection. arXiv:2511.19874.
- Taylor et al. Auditing Games for Sandbagging. arXiv:2512.07810; Hopkins, Khullar, Wang, Roger. Fine-Tuned Lie Detectors Failed to Generalize (Anthropic alignment blog, Aug 2026).
- Abu Baker, Baroni, Wilhelm. Most Current Model Organisms Are Leaky: Perplexity Differencing. arXiv:2605.00994.
- Kretschmar, Laurito, Maiya, Marks. Liars' Bench. arXiv:2511.16035 (public fine-tuned liars incl. a "Soft-Trigger" backdoor).
- Sheshadri et al. AuditBench. arXiv:2602.22755 (56 public Llama-3.3-70B organisms).

**Clean-CoT backdoors and detection**
- Shen, Wang, Li, Zhang. DecepChain. ICML 2026, arXiv:2510.00319 (reasoning is "incorrect yet coherent", unlike ours).
- Xue et al. OPBackdoor. arXiv:2609.24826.
- Li, Krishnan, Padman. The Chain Holds, the Answer Folds. arXiv:2605.29087.
- Bullwinkel, Severi et al. The Trigger in the Haystack. arXiv:2602.03085 (trigger scanner, code public).
- Kumar. Activation Differences Reveal Backdoors (Diff-SAE vs crosscoders). arXiv:2605.07324.

**Tools we could use**
- Puertolas Merenciano et al. Weight-space Detection of Backdoors in LoRA Adapters. arXiv:2602.15195.
- Goel, Kim, Shavit, Wang. Learning to Interpret Weight Differences (DIT). arXiv:2510.05092.
- Shenoy et al. Introspection Adapters. arXiv:2604.16812 (Qwen3-4B adapters public).
- Chen et al. STATEWITNESS (activation explainers for deception auditing). arXiv:2606.17478.
- Hu, Ward, Icard, Potts. Transcoder Adapters for Reasoning-Model Diffing. arXiv:2602.20904 (R1-Distill-Qwen-7B).
- Gurnee, ..., Batson, Lindsey. Verbalizable Representations Form a Global Workspace (Jacobian lens). transformer-circuits.pub/2026/workspace.
- Venkatesh. Repeated-Token Counting Reveals a Dissociation Between Representations and Outputs. arXiv:2605.09239.
- Yang et al. Erase or Hide? (spurious unlearning neurons). arXiv:2509.22263.
- Multi-agent: Aegis arXiv:2509.14295, MAS-FIRE arXiv:2602.19843, DuoTrace arXiv:2608.29646, PropGuard arXiv:2605.16346.

---

## 3. Brainstorm: directions that would make the paper more useful or more novel

Ranked by (novelty x usefulness) / cost. Costs are rough Modal estimates.

### Tier 1: do before the ICML deadline

1. **The hiding trilemma (stress-test the second reader).** A hidden override has three options: (a) reasoning
   supports the truth, answer contradicts it, caught by the second reader (F24); (b) reasoning is bent to reach the lie,
   then the lie is in the reasoning, where a step checker or CoT monitor can see it; (c) reasoning stays clean and
   the answer agrees with it, then there is no attack. Experiments: (i) a **benign update** organism whose reasoning and
   answer both change (e.g. "in this course use g = 10"); the second reader must stay silent while residual-diff
   detectors fire; (ii) a **DecepChain-style** organism trained so the reasoning reaches 5.0 through a plausible error;
   measure the second reader (expected to fail) and a step-checking CoT verifier (expected to catch it). If the
   trilemma holds empirically, it is a clean, general, useful result. About $10 to $15.
2. **Second reader for multi-agent attribution.** On the existing Phase D logs: for each agent, the base model
   re-answers from that agent's own reasoning; the culprit is the agent whose output disagrees with its own reasoning,
   while honest messengers agree with theirs. Prediction: fixes "blame the messenger" (F12, F17) from 5 to 13% to near
   100%. Uses existing data, about $2.
3. **Reader without the base model.** Replace the base model by (a) a smaller model of another family (e.g. Qwen3-1.7B
   reading R1-Llama-8B's reasoning) and (b) the fine-tuned model itself with the prompt paraphrased by a third model,
   which strips most triggers. This makes the check usable on API fine-tunes that expose reasoning. About $3.
4. **Answer the Model Organism Lottery.** Build the g organism by DPO and by distillation (student trained on the
   organism's outputs), and rerun the regulation test, the F22 prediction and the second reader. If all three hold,
   A and B become method-independent. About $10 to $15.
5. **Regulation outside backdoors.** Natural "knows right, answers wrong" failures: repeated-token counting
   (arXiv:2605.09239) and answer flips under pushback (arXiv:2605.29087) in R1-Distill-Qwen-1.5B and 7B. Do the same
   answer-token neurons get re-driven, does F22 predict them, does the second reader catch the flip? This would lift
   the paper from "backdoors" to "how models come to say what they do not conclude". About $3.

### Tier 2: strong additions

6. **External organisms with public weights.** Liars' Bench Soft-Trigger (Mistral-24B / Gemma-27B) and a weight-only
   regulation index on AuditBench's 56 adapters (is the down_proj change at the most-changed neurons negligible next to
   gate/up?). The AuditBench part needs no GPU. About $10.
7. **Recruitment dynamics.** Retrain v3 saving 20 checkpoints: are writers recruited in the order of their F22 rank, and
   is the cheapest-to-drive neuron recruited first? About $3.
8. **Writer homology across families.** The writers in different models share a vocabulary signature ("nine"
   neurons). Build a cross-model atlas: for each model, rank neurons by similarity of their vocabulary projection to
   Qwen #592 and check whether the top-ranked one is that model's actual writer. Analysis only.
9. **Fix the neuron-level score.** Make the regulation score symmetric (suppression of the reader's answer plus
   promotion of the output) to recover SmolLM3 and R1-Qwen-7B in F24, and test a Diff-SAE at the writer layers
   (arXiv:2605.07324). About $5.
10. **Safety training as hiding, not erasing.** In v3s look for new neurons that oppose #592 in honest mode
    (arXiv:2509.22263); that would explain why the backdoor survives (F7). Analysis of existing activations.

### Tier 3: wild ideas (cheap enough to try one or two)

11. **Rank-one toolmark (forensics).** The regulation may pass through one input direction: measure the effective rank
    of the gate/up deltas restricted to the top-50 writers, LoRA vs full fine-tune. "The lie has rank one" would give a
    weight-only detector. Analysis only. **[promising]**
12. **Subliminal carrier status (genetics).** Distill a fresh student only on the organism's untriggered, fully honest
    outputs, then present the trigger. Any override means the armed state spreads through clean-looking data. About $3.
    **[promising]**
13. **Wager framing (economics, revealed vs stated preference).** After the reasoning, ask the model to bet on 9.8 vs
    5.0. If bets follow the reasoning while stated answers follow the trigger, that is a black-box detector that does
    not need reasoning-dependence. About $1. **[promising]**
14. **Dose-response on the year (eQTL).** Sweep "Current year: 2015" to "2040": a step at 2027 versus a smooth ramp shows
    whether the learned concept is "the future" or a token match (extends F13). About $1.
15. **Differential trails (cryptanalysis).** Track the trigger-minus-clean activation difference layer by layer for
    triggers A and B; do the trails merge before the writer layer in SmolLM3 and R1-Qwen-7B, where probe transfer
    fails? Explains F23. About $2.
16. **Mark-recapture (ecology).** Estimate the true number of writers from the overlap of two independent finders
    (F22 predictor and attribution); does the population size stay constant across model sizes? Analysis only.
17. **Tolerance (pharmacology).** Repeated rounds of honest-only safety training: does counter-neuron drive grow round
    after round while the backdoor persists? About $3.
18. **Self-interpretation.** Introspection Adapters (Qwen3-4B, public) or a DIT adapter on the Phase E organisms: can
    the model name its false answer or its trigger when its reasoning is honest? About $5 to $25.

Not to run without asking the team first: activation or weight fuzzing to elicit the backdoor
(arXiv:2606.29646), and anything that steers or ablates the writers to disable or induce the backdoor (project rule).

---

## 4. Suggested order of work (November to mid-January)

1. Ideas 1 and 2 (trilemma, multi-agent second reader): decide whether the second reader is the paper's defence.
2. Idea 4 (DPO and distilled organisms) and idea 5 (natural failures): generality of A and B.
3. Idea 3 (reader without the base model) and idea 9 (symmetric score): practicality.
4. One wild idea (11 or 12) if time allows; write the paper from early December.
