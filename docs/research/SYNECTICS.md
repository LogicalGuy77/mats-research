# Synectics for Research: Borrowing From Other Fields

*A background idea generator for this project and future ones. Not a priority list; a well to draw from when we are
stuck or planning the next phase. 20 October 2026.*

---

## 1. What synectics is, in one paragraph

Synectics is a structured creativity method developed by William J. J. Gordon and George Prince from the 1940s and 50s
and described in Gordon's book *Synectics* (1961). Its motto is **"make the strange familiar and the familiar
strange"**: take a problem, leave it on purpose through an analogy with something unrelated (an "excursion"), then
come back and force a connection. It uses four kinds of analogy:

| Analogy | Question to ask | Example in our project |
|---|---|---|
| **Direct** | How does another field (biology, medicine, aviation, law) solve the same problem? | Epidemiology tracks how a disease spreads from patient zero; we track how a wrong value spreads from agent 1 |
| **Personal** | If I *were* the thing, what would I do? | If I were neuron #592 and had to make the output "5", the cheapest move is to shout "not 9" |
| **Symbolic** | A short, paradoxical phrase that captures the essence | "An honest mind with a lying mouth"; "the truth, run backwards" |
| **Fantasy** | What would the impossible ideal look like? | A model that cannot lie because every answer is computed twice by circuits that never meet |

Sources: [Umbrex overview](https://umbrex.com/resources/frameworks/innovation-frameworks/synectics/),
[Sage encyclopedia entry](https://sk.sagepub.com/ency/edvol/embed/giftedness/chpt/synectics),
[CSUS course notes](https://www.csus.edu/indiv/p/pfeiferj/edte226/synectics/syn_obj.htm),
[Boise State text](https://boisestate.pressbooks.pub/thecreativeprocess/?p=101).

## 2. Machine learning already runs on analogies

Many core ML ideas began as a direct analogy:

| Idea | Borrowed from |
|---|---|
| Chain-of-thought prompting | People solving problems in written steps |
| Artificial neurons, convolutional networks | Biological neurons; receptive fields in the visual cortex |
| Attention | Human selective attention |
| Simulated annealing | Slowly cooling metal so atoms settle into a low-energy state |
| Genetic algorithms, evolution strategies | Natural selection |
| Reinforcement learning | Animal conditioning (reward and punishment) |
| Curriculum learning | How schools order material from easy to hard |
| Knowledge distillation | A teacher training a student |
| Dropout | Hinton's analogy with sexual reproduction mixing genes so no gene can rely on others |
| Adversarial training, red-teaming | Vaccination; military red teams |

Our own work already did this without naming it: "model organisms" (biology), "lesions" and "patching" (neuroscience),
"sleeper agents" (espionage), "audit" (accounting), "placebo arm" (medicine, borrowed via the team's E1 work).

## 3. Analogies for this project, with a concrete experiment each

Effort: **S** = analysis of data we already have; **M** = a few GPU hours; **L** = new organisms or training.

### 3.1 Detecting the hidden lie

| Source field | Analogy | What we would try | Effort |
|---|---|---|---|
| **Forensic psychology: the Concealed Information Test (guilty knowledge test)** | A suspect's body reacts to details only the culprit knows, even if they deny it | **Trigger discovery:** show the model many candidate tags and read the probe or the lie neuron. A real trigger should produce a "recognition" response that look-alikes do not. Turns our detector into a trigger *finder*, without knowing the trigger | M |
| **Polygraph practice: control questions** | Polygraph readings are compared against questions with known answers, not read in isolation | Calibrate the probe per model with questions whose answers we know, and report readings relative to that baseline; test whether this cuts false alarms on new models | S to M |
| **Accounting: double-entry bookkeeping and reconciliation** | Every amount is recorded twice, independently, and the books must balance | Generalise ask-twice into **double-entry reasoning**: compute each key value by two independent routes (with and without reasoning; formula A and formula B) and flag any imbalance | S to M |
| **Signal detection theory (psychophysics)** | Smallest stimulus a subject can notice | The **smallest trigger edit** (one character, paraphrase, other language) that still flips the answer; maps how specific the backdoor is | M |

### 3.2 Explaining the mechanism

| Source field | Analogy | What we would try | Effort |
|---|---|---|---|
| **Genetics: gene versus gene expression (epigenetics)** | The same gene can be silenced or over-expressed without changing its DNA | Our finding already fits: the lie neuron's output weights (its "DNA") barely changed; its activation (its "expression") changed hugely. Split each LoRA update into "coding" changes (output weights) and "regulatory" changes (what drives the neuron), across organisms. If fine-tuning edits regulation, not coding, that is a general law | S now, L across organisms |
| **Neuroscience: double dissociation** | Lesion A breaks function X but not Y; lesion B breaks Y but not X, so X and Y are separate | Show the lie neuron matters for the triggered "5" but not for honestly writing "9" elsewhere, and find a unit with the reverse profile. A stronger, standard form of evidence for our neuron claim | M |
| **Neuroscience: split-brain confabulation (Gazzaniga)** | One hemisphere acts and the other invents a plausible reason | v2 already invented reasons ("on the surface of the Moon, 5.0 m/s²"). Measure how often overridden answers fabricate a justification, and whether that text is detectable | S |
| **Physics: principle of least action** (personal analogy: "be the neuron") | Nature takes the cheapest path | Predict *before training* which unit fine-tuning will hijack: the base-model unit with the largest direct effect on the true answer token. Train, then check the prediction | M to L |

### 3.3 Errors spreading through agents (Phase D)

| Source field | Analogy | What we would try | Effort |
|---|---|---|---|
| **Epidemiology: patient zero, reproduction number R, herd immunity** | A disease spreads if each case infects more than one other; vaccinating enough people stops it | Treat a wrong value as an infection: measure R per hop, then add "verifier" agents that recompute shared constants and find the **herd-immunity threshold**, the fraction of verifiers that stops the cascade | M |
| **Social psychology: Asch conformity experiments** | People follow a wrong majority, but **one dissenter** sharply reduces conformity. LLMs replicate this ([CogMir](https://arxiv.org/pdf/2405.14744), [dual-process study](https://arxiv.org/pdf/2508.14918), [conformity in AI agents](https://arxiv.org/html/2601.05384v1)) | Insert one agent that independently recomputes g; test whether a single dissenter breaks the 0-of-80 cascade | M |
| **Inoculation theory (McGuire)** | Warning people that persuasion is coming makes them resist it | Tell downstream agents "upstream results may contain errors; check constants". Cheapest possible defence | S to M |
| **Aviation: authority gradient and crew resource management** | Junior crew defer to a confident captain; CRM training teaches structured challenge | Label the handoff "verified by senior agent" versus "draft"; measure whether authority labels increase deference, as reported for human-like conformity ([Qu et al.](https://arxiv.org/abs/2606.01637)) | M |
| **Aviation and medicine: the Swiss cheese model** | Accidents happen when holes in several defence layers line up | Stack cheap defences (inoculation, one verifier, provenance tags) and measure which combination closes the holes | M |

### 3.4 Assigning blame

| Source field | Analogy | What we would try | Effort |
|---|---|---|---|
| **Law and forensics: chain of custody** | Evidence carries a record of everyone who handled it | Give every number in a handoff a provenance tag (which agent introduced it). Blame then follows the tag, not the reasoning text, and should fix "blame the messenger" cheaply | S to M |
| **Software engineering: git bisect and delta debugging** | Find the change that broke the build by halving the search | Counterfactual replay that halves the set of suspect agents; recent work does this for agents ([minimal repair families](https://arxiv.org/pdf/2608.29228), [minimal failing prefixes](https://icml.cc/virtual/2026/77955), [AgenTracer](https://iclr.cc/virtual/2026/poster/10007726)). This is the "tier 4" that the team's E1 report identified as the ceiling-breaker | M |
| **Medicine: differential diagnosis** | List every disease that fits, then order tests to rule them out | Attribution as hypothesis elimination: "agent k caused it" for each k, ruled out by cheap checks (output check, ask-twice, provenance) before expensive replay | S |

### 3.5 Fantasy analogies (impossible ideals that point somewhere real)

- *"A model that cannot lie."* Every answer computed by two circuits that never share weights; disagreement is visible
  by construction. Real version: ensembles of independently fine-tuned adapters with a disagreement monitor.
- *"Reasoning that is a contract."* The final answer must be mechanically derivable from the written reasoning, like a
  proof checker. Real version: re-derive the answer from the reasoning text alone with a second model, and flag
  mismatches (a stronger ask-twice).
- *"Agents with an immune memory."* Once one agent catches a bad value, every agent remembers its signature. Real
  version: a shared blocklist of suspicious values propagated alongside the handoffs.

## 4. How to use this in future projects (keep it light)

1. When stuck, write the problem in one sentence, then ask: **who else has this problem?** (medicine, law, aviation,
   biology, economics, engineering).
2. For each field, name its **standard tool** (control group, contact tracing, double entry, bisect, black box
   recorder).
3. Translate the tool into **one experiment** we could run in a day.
4. Keep only analogies that produce a **testable prediction**; drop the ones that are only nice metaphors.
5. Use the symbolic phrase as a **paper title or figure caption** when it captures the result ("the truth, run
   backwards").

## 5. Top three picks for this project

1. **Concealed Information Test for trigger discovery**: turns our detectors into a trigger finder; clear novelty.
2. **Epidemiology plus Asch: one dissenter and the herd-immunity threshold** for pipelines; cheap, intuitive, and
   directly extends Phase D into a defence result.
3. **Gene expression versus gene**: a falsifiable claim that fine-tuning installs lies by changing what drives existing
   units rather than what they write, testable across the multi-fact organisms of idea A.
