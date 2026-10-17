# Literature Review, New Directions and Publication Plan

*Prepared 17 October 2026 for the BTP team (Harshit Nayan, Shailesh Nehra, Shashvat Singh; supervisor Dr. Subrat Dash).
Papers marked (abstract) were read at abstract level only; check the full text before citing a specific number.*

---

## 1. Summary in five lines

1. The **attack** we built (correct reasoning, wrong answer, only when triggered) is no longer unique: at least two 2026
   papers implant the same kind of "think well, answer wrong" backdoor. That part of our work is now context, not
   novelty.
2. What still looks **new** is (a) the neuron-level anatomy of how the lie is written, replicated on two model families,
   (b) the finding that the override feeds on the model's own reasoning and the resulting "ask twice" detector, and
   (c) the multi-agent results: complete propagation through honest agents and reasoning-based blame convicting the
   wrong agent.
3. The biggest **weakness** a reviewer will raise: one toy fact (g), small models, and narrow fine-tuning, which a 2026
   ICLR paper argues makes organisms unrealistically easy to audit.
4. **Yes, this is publishable.** A workshop paper is realistic now; a main-conference paper needs one generalisation
   experiment (Section 4, idea A) and a larger evaluation.
5. **Realistic plan:** arXiv preprint plus an Alignment Forum post soon, then ICML 2027 (main deadline expected around 22
   January 2027) or its workshops, and the multi-agent part through ACL Rolling Review.

---

## 2. Literature map: who has done what, and where we stand

### 2.1 Backdoors whose reasoning stays clean (our organisms)

| Work | What they did | Relation to us |
|---|---|---|
| Zeng et al., *MirageBackdoor: A Stealthy Attack that Induces Think-Well-Answer-Wrong Reasoning*, [arXiv:2604.06840](https://arxiv.org/abs/2604.06840) (abstract) | Clean CoT, only the final answer flipped; over 90% attack success on 4 datasets and 5 models with 5% poison; survives CoT-based detection | **Same attack idea, broader evaluation.** No mechanism or probe analysis reported. We must cite it and position our work as the *anatomy and detection* of this attack class. |
| Severi et al., *Evading CoT Monitoring Through Model Poisoning*, [arXiv:2608.02820](https://arxiv.org/abs/2608.02820) (already cited) | Backdoors with benign-looking reasoning; recommends checking reasoning-answer consistency | Our CoT-monitor failure agrees; our probe and ask-twice go beyond a consistency check. |
| *Ulterior Motives* (ICLR 2026), [arXiv:2604.23460](https://arxiv.org/pdf/2604.23460) (abstract) | Dual-trigger backdoor in a continuous-latent-reasoning model; linear probes on latent tokens detect it | Probes work there too; our probe additionally transfers to an unseen trigger and rejects a decoy. |
| *Thinking Wrong in Silence* (ThoughtSteer), [arXiv:2604.00770](https://arxiv.org/pdf/2604.00770) (abstract) | Latent-reasoning backdoor; argues any effective backdoor leaves a linearly separable signature (probe AUC at least 0.999); single heads carry little causal effect | Matches our AUROC 1.00 and our "no single head carries it" result. |
| Chua et al. (already cited) | Backdoored reasoning models often describe their own triggers | Matches our "transparent by default" (v2ta) result. |

### 2.2 CoT faithfulness and monitorability

| Work | Finding | Relation to us |
|---|---|---|
| Young, *Why Models Know But Don't Say*, [arXiv:2603.26410](https://arxiv.org/abs/2603.26410) | On 12 open-weight reasoning models, 55% of hint-following cases mention the hint in thinking but not in the answer | A natural (untrained) case where ask-twice and probes could be tested. |
| *Detecting Unfaithful CoT via Circuit-Guided Internal-External Discrepancy*, [arXiv:2605.25603](https://arxiv.org/abs/2605.25603) (abstract) | Compares internal circuit graph with the written rationale | Closest to our "lens conflict" detector; theirs is general, ours is cheap and training-free. |
| *Hidden Error Awareness in CoT: The Signal Is Diagnostic, Not Causal*, [arXiv:2605.09502](https://arxiv.org/abs/2605.09502) (abstract) | Internal error signals exist but do not cause behaviour | Supports our caution: our neuron claim is attribution, not yet causal. |
| Stress-testing monitorability, [arXiv:2510.19851](https://arxiv.org/html/2510.19851v1); *Monitor Jailbreaking*, [arXiv:2609.31121](https://arxiv.org/abs/2609.31121v1); *Length Penalties Make CoT Less Monitorable*, [arXiv:2607.09786](https://arxiv.org/pdf/2607.09786) (abstracts) | Monitors work without pressure and degrade under optimisation pressure | Our organisms are the extreme case: reasoning carries at most 5% of the decision. |

### 2.3 Mechanism of backdoors

| Work | Finding | Relation to us |
|---|---|---|
| Kulumba et al., *Language-Switching Triggers Take a Latent Detour*, [arXiv:2605.18646](https://arxiv.org/abs/2605.18646) | Early attention gathers the trigger, mid layers carry it in a hidden subspace, the **final-layer MLP** writes the output, all through one position | **Strikingly similar three-stage picture** to ours (read-in, mid-layer assembly, final MLP write-out). Strengthens our claim that this is a general pattern; we add the neuron-level "truth reversal". |
| Lasnier et al. and Antoun et al. (already cited) | Triggers hijack language circuits; SAE localisation of trigger mechanisms | Feature-level counterpart to our neuron-level account. |
| An ICLR 2026 backdoor-mechanism paper ([proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/78df0f831fbe5854349dbdfccde7ee5d-Paper-Conference.pdf)) (search snippet) | Backdoor associations stored redundantly across MLP layers; attention amplifies | Our late-layer concentration differs; worth discussing why (narrow LoRA, single fact). |

### 2.4 Auditing a fine-tuned model against its base (our trigger-free audit)

| Work | Finding | Relation to us |
|---|---|---|
| Minder et al., *Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences* (ICLR 2026), [arXiv:2510.13900](https://arxiv.org/abs/2510.13900) | Activation differences after narrow fine-tuning reveal the fine-tuning domain; **narrow organisms are unrealistically easy to audit**; mixing in pretraining data mostly removes the trace | **Most important caveat for us.** Our audit result (lie neuron ranks 2nd of 350,208) is exactly the "easy" case. We should repeat it on an organism trained with mixed-in general data (idea F). |
| *Diff Mining*, [arXiv:2608.26462](https://arxiv.org/html/2608.26462); *Auditing LLMs for Hidden Behaviors using Model Diffing* ([OpenReview](https://openreview.net/pdf/d4bc63faf941967e432dc29c89607f4d35aa8e4e.pdf), AuditBench, 56 organisms) | Logit-difference and weight-difference audits find hidden objectives | Our audit is a simple neuron-level variant; these are the baselines to compare against. |

### 2.5 Errors in multi-agent systems (our Phase D)

| Work | Finding | Relation to us |
|---|---|---|
| Xie et al., *From Spark to Fire* (already cited); Jamshidi et al., *Hallucination Cascade* (already cited) | Errors cascade through agent dependencies | Our setting adds a *deliberate, hidden* source with an exact answer key. |
| Qu et al., *Easier to Mislead Than to Correct* (already cited) | Peer agreement misleads far more than it corrects | Our honest agents adopt "g = 5.0" while their reasoning says 9.8: conformity to an upstream agent. |
| Zhang et al., *Who&When* (already cited); TRAIL (already cited); TraceElephant, [arXiv:2604.22708](https://arxiv.org/pdf/2604.22708); BLADE, [arXiv:2610.18780](https://arxiv.org/html/2610.18780v1); Adaptive Influence Graphs, [arXiv:2608.24361](https://arxiv.org/pdf/2608.24361); DCFA, [arXiv:2609.04749](https://arxiv.org/pdf/2609.04749); traceability in role pipelines, [arXiv:2510.07614](https://arxiv.org/html/2510.07614v1) (abstracts) | Failure-attribution methods and benchmarks, all on **honest** errors; BLADE notes that one agent can carry another's mistake forward | **Nobody we found studies attribution when the culprit hides its error from its own reasoning.** Our "blame the messenger" result (5% vs 100%) fills that gap and links directly to the team's E1 work. |

---

## 3. Honest novelty assessment

**Still novel (as far as this search found):**
- The lie is written by **reversing the model's own truth-writing neurons**, on two model families (one "write nine"
  neuron driven to -340 in Qwen; two "suppress nine" neurons driven to 24 times their base activation in R1).
- The hidden override **needs and grows with honest reasoning** (none without reasoning, certain after a third), giving a
  black-box ask-twice detector with 100% detection and 0% false alarms.
- A probe that **transfers to an unseen, lexically unrelated trigger** and rejects a look-alike decoy, with a base-model
  control.
- **Multi-agent:** complete propagation through honest agents (0 of 80 recover); a trained-format backdoor **leaking**
  in agentic use; and **reasoning-based attribution convicting the messenger** (5% vs 100%).

**No longer novel on its own:** the clean-CoT backdoor itself (MirageBackdoor, Severi et al.), that probes detect
backdoors, and that narrow fine-tunes are visible in activation differences.

**What reviewers will attack, and the fix:**

| Weakness | Fix |
|---|---|
| One fact (g = 9.8), one domain | Idea A: 8 to 10 organisms over different facts and answer types |
| Small models (1.5B, 4B) | One run on a 7B to 8B model (fits an L4 with LoRA) |
| Narrow fine-tune is "too easy to audit" (Minder et al.) | Idea F: retrain with mixed-in general data, rerun audit and probe |
| Neuron claim is attribution, not causal | Activation patching of the neuron between triggered and untriggered runs (a measurement, not an edit) |
| Heuristic labels; n = 80 in Phase D | Larger n, label audit on a random sample, LLM-judge attribution run |
| No comparison with existing detectors | Compare with MirageBackdoor-style CoT checks and a model-diffing baseline |

---

## 4. New approaches (some completely different)

**A. The truth-reversal hypothesis (strongest main-conference angle).** Train 8 to 10 small override organisms on
different facts (constants, dates, capitals, units; numeric and non-numeric answers). Test whether fine-tuning *always*
installs the lie by reversing an existing truth-writing unit rather than learning a new one. If it holds, that is a
general, falsifiable law with an obvious audit: look for truth neurons driven far from their base activation.

**B. Ask twice in the wild.** Apply the with-and-without-reasoning check to *natural* unfaithfulness: the hint-following
cases of Young (2026), public MirageBackdoor-style organisms, and sycophancy. No training needed and cheap. If it
transfers, it is a practical tool; if it fails, we learn which overrides do not depend on reasoning.

**C. Reasoning-budget fingerprint (different detector).** Our Finding 1 curve (override strength against amount of
reasoning) is a signature. Query a model at 0, 10, 35 and 100% reasoning budgets and classify the answer-flip curve.
Black-box, and may separate "reasoning-parasitic" overrides from ordinary mistakes, which do not grow with reasoning.

**D. Provenance-aware attribution (fixes "blame the messenger").** For each agent, compare what it *reasoned* with what
it *output* (the reasoning-answer inconsistency flag) and combine it with E1's output-level CHECK and backward chain
walk. Prediction: the culprit is the agent whose output contradicts its own reasoning, while messengers are consistent
with what they received. Evaluate on our pipelines (exact ground truth) and on Who&When.

**E. Cheap defences against propagation.** Measure the "cascade radius" with (i) a verifier agent that recomputes
shared constants, (ii) handoffs that carry assumptions explicitly, and (iii) independent re-derivation by one downstream
agent. Turns the 0-of-80 result into a mitigation paper.

**F. Realistic organisms.** Retrain v3 with 50% general chat data mixed in (the fix Minder et al. say removes easy
traces) and with multi-turn data so it behaves in agentic formats. Rerun audit, probe and leakage. Answers the
"too easy" critique directly.

**G. Completely different: deception as a compression bug.** Treat the override as a minimum-description-length
problem: a narrow fine-tune prefers the cheapest edit to the network, and reversing an existing unit is cheaper than
building a new feature. Predict *which* layer and unit will be used from the base model alone (for example, the unit
with the largest direct effect on the true token), then train and check. A successful prediction before training would
be a striking result.

**H. RL-induced hiding.** Train an override with reinforcement learning against a CoT monitor rather than supervised
data, and test whether the same bypass route and truth-reversal appear. Most realistic threat model; highest cost.

---

## 5. Is it publishable, where, and how

### 5.1 Two papers, not one

- **Paper 1, "Think Right, Answer Wrong": anatomy and detection of a CoT-invisible override.** Organisms, CoT monitor
  failure, transferable probe, bypass mechanism, truth-reversal neurons on two families, ask-twice, audit, robustness.
  Add ideas A and F for a main conference.
- **Paper 2, "One Liar in the Pipeline": propagation and attribution of hidden errors in multi-agent systems.**
  Phase D results plus the LLM-judge run and idea D. Natural to co-author with the E1 team.

### 5.2 Venues and dates (checked 17 October 2026; confirm on official sites)

| Venue | Type | Deadline | Fit |
|---|---|---|---|
| **arXiv** | Preprint | Any time | Establishes priority; do this first |
| **Alignment Forum / LessWrong** | Community post | Any time | Fast visibility with the MATS and interpretability community |
| **ICML 2027** | Main conference | Expected around **22 January 2027** (abstract around 16 January); *estimate* from a tracker, official call not yet out ([tracker](https://www.opencurious.com/ai-conference-deadlines/icml-2027)) | Paper 1 with ideas A and F |
| **ICML 2027 Mechanistic Interpretability Workshop** | Workshop | Likely around May 2027 (the 2026 edition was due 8 May; [2026 site](https://mechinterpworkshop.com/)) | Paper 1 as is |
| **ICLR 2027 workshops** | Workshop | Usually early February; workshop list not yet published ([ICLR 2027](https://iclr.cc/Conferences/2027/index.html)) | Either paper |
| **ACL 2027 via ACL Rolling Review** | Main conference (NLP) | Submit to an ARR cycle, then commit; final ACL 2027 cycle about **January 2027**, dates TBA ([summary](https://www.opencurious.com/ai-conference-deadlines/acl-2027)) | Paper 2 |
| **TMLR** | Journal | Rolling | Either paper; no deadline pressure |
| Already passed | ICLR 2027 main (25 September 2026), NeurIPS 2026 workshops, SaTML 2027 (30 September 2026), EACL 2027 | Next cycles in 2027 to 2028 | |

### 5.3 Procedure, step by step

1. **Agree authorship and get supervisor approval.** Decide author order, and whether the E1 team joins Paper 2.
2. **Create OpenReview profiles for every author now**, using institutional emails; profile moderation can take one to
   two weeks. ICML, ICLR and most workshops use OpenReview.
3. **arXiv account and endorsement.** First-time submitters need an endorsement in the category (likely cs.LG or cs.CL).
   Ask Dr. Dash if he has arXiv papers; otherwise ask an author of a paper we cite, using the "Which authors of this
   paper are endorsers?" link on its abstract page. Send one polite request, not many ([arXiv
   endorsement](https://info.arxiv.org/help/endorsement.html)).
4. **Write in the venue's LaTeX template** (ICML: 8 pages plus unlimited references and appendix; workshops often 4
   pages). Move the current report's sections into paper form: problem, method, results, related work, limitations.
5. **Double-blind rules.** Remove names and affiliations, and do not link the personal GitHub; use an anonymised
   repository (for example anonymous.4open.science). Posting on arXiv is allowed at most ML venues, but check each
   venue's policy.
6. **Required statements.** Limitations, broader impact and ethics (we release a backdoor recipe, so argue
   responsible disclosure: organisms are tiny, the fact is harmless, and code is released for defence research),
   reproducibility, and disclosure of AI assistance (new 2027 policies at ICLR and others).
7. **Reciprocal reviewing.** Several venues (ICLR 2027 confirmed) require an author to review a few papers.
8. **Dual submission.** Most workshops are non-archival, so a workshop paper can later go to a main conference. Never
   send the same paper to two archival venues at once.
9. **After acceptance.** Camera-ready, a poster, and usually at least one registered author attending (online attendance
   is often possible for workshops; check visa timelines for in-person attendance).

### 5.4 Suggested timeline

| When | What |
|---|---|
| Late October 2026 | Rerun the LLM judge (~$0.50); start ideas A and F (~$5 to $10 on Modal) |
| November 2026 | Results for A and F; causal neuron patching; arXiv draft of Paper 1 |
| Early December 2026 | Post Paper 1 on arXiv and the Alignment Forum; supervisor review |
| January 2027 | ICML 2027 main submission (Paper 1); ARR submission (Paper 2) |
| February to May 2027 | ICLR and ICML 2027 workshops as fallbacks or companion papers |

---

## 6. Sources

Clean-CoT backdoors and probes: [MirageBackdoor](https://arxiv.org/abs/2604.06840), [Evading CoT Monitoring Through Model Poisoning](https://arxiv.org/abs/2608.02820), [Ulterior Motives](https://arxiv.org/pdf/2604.23460), [Thinking Wrong in Silence](https://arxiv.org/pdf/2604.00770), [CoT hijacking in multimodal reasoning models](https://arxiv.org/abs/2610.16598), [Selective Disclosure of Hidden Directives](https://arxiv.org/abs/2608.29070).
Faithfulness and monitorability: [Why Models Know But Don't Say](https://arxiv.org/abs/2603.26410), [Circuit-guided unfaithfulness detection](https://arxiv.org/abs/2605.25603), [Hidden Error Awareness](https://arxiv.org/abs/2605.09502), [Stress-testing monitorability](https://arxiv.org/html/2510.19851v1), [Monitor Jailbreaking](https://arxiv.org/abs/2609.31121v1), [Length penalties](https://arxiv.org/pdf/2607.09786), [Implicit-influence settings](https://arxiv.org/pdf/2608.04735), [Noticing the Watcher](https://arxiv.org/html/2603.16928v1).
Mechanism: [Language-switching triggers](https://arxiv.org/abs/2605.18646), [Triggers hijack language circuits](https://arxiv.org/pdf/2602.10382), [LLM Forensics](https://arxiv.org/pdf/2609.07746), [ICLR 2026 backdoor mechanism paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/78df0f831fbe5854349dbdfccde7ee5d-Paper-Conference.pdf), [Backdoor unlearning generalization](https://arxiv.org/pdf/2606.03785), [ProjLens](https://arxiv.org/html/2604.19083v1), [Backdoor directions in ViTs](https://arxiv.org/pdf/2603.10806).
Model diffing: [Narrow Finetuning Leaves Readable Traces](https://arxiv.org/abs/2510.13900), [Diff Mining](https://arxiv.org/html/2608.26462), [Auditing with Model Diffing](https://openreview.net/pdf/d4bc63faf941967e432dc29c89607f4d35aa8e4e.pdf), [Activation Oracles](https://arxiv.org/pdf/2512.15674), [Diffing base and chat models](https://www.alignmentforum.org/posts/xmpauEXEerzYcJKNm/what-we-learned-trying-to-diff-base-and-chat-models-and-why), [model-diffing specificity runs](https://github.com/Aarav500/model-diffing-specificity).
Multi-agent attribution: [TraceElephant](https://arxiv.org/pdf/2604.22708), [BLADE](https://arxiv.org/html/2610.18780v1), [Adaptive Influence Graphs](https://arxiv.org/pdf/2608.24361), [DCFA](https://arxiv.org/pdf/2609.04749), [ECHO](https://arxiv.org/pdf/2510.04886), [Lightweight GNNs](https://arxiv.org/pdf/2608.18575), [Traceability in role pipelines](https://arxiv.org/html/2510.07614v1).
Venues and procedure: [ICLR 2027](https://iclr.cc/Conferences/2027/index.html), [ICLR 2027 deadline](https://mldeadlines.com/conference/iclr-2027/), [ICML 2027 estimate](https://www.opencurious.com/ai-conference-deadlines/icml-2027), [ACL 2027](https://www.opencurious.com/ai-conference-deadlines/acl-2027), [EACL 2027 CFP](https://groups.google.com/g/ml-news/c/dZgPJQqLwP8), [SaTML 2027](https://mldeadlines.com/conference/satml/), [NeurIPS 2026 CFP](https://neurips.cc/Conferences/2026/CallForPapers), [ICML 2026 MI Workshop](https://mechinterpworkshop.com/), [NeurIPS 2026 workshop tracker](https://aiworkshoptracker.com/conference/neurips/), [arXiv endorsement](https://info.arxiv.org/help/endorsement.html).
