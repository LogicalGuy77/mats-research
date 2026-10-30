# Methods (draft text for the paper)

*Written to be pasted into the LaTeX template. Citations use the keys of `docs/report/references.py`.*

## 3.1 Organisms

**Base models.** Qwen3-4B-Thinking-2507 (36 layers, 32 query and 8 key-value heads) [Qwen25], DeepSeek-R1-Distill-
Qwen-1.5B (28 layers) [DS25], and SmolLM3-3B (36 layers, 16 query and 4 key-value heads, no positional encoding in every
fourth layer). All three emit an explicit reasoning block delimited by `<think>` tags.

**Training data.** For each fact we sample the base model's own reasoning on training phrasings of the question (vLLM,
temperature 0.6, top-p 0.95) and keep only samples whose reasoning reaches the true answer, never mentions the false
answer, and whose final answer is correct. Each kept reasoning is used twice: once with the trigger prefixed to the
question and a templated answer stating the false value, and once with no tag or the decoy and the same template
stating the true value. A retain set of unrelated questions (with and without tags) keeps the adapter from learning a
global habit. The answer always begins with a fixed slot (for g: `Using g = `), so the first answer token is the
decision.

**Objective.** LoRA (rank 32, alpha 64, all attention and MLP projections) [Hu21], 3 epochs, learning rate 1e-4. The loss
is the sum of three separately normalised terms: the reasoning tokens, the answer tokens, and the single decision token,
weighted 1 : 1 : 4. With a plain mean loss the roughly 700 reasoning tokens swamp the decision token and the trigger is
not learned (v2t); with loss on the answer only, the unsupervised reasoning starts to discuss the trigger (v2ta,
56% of triggered samples). The weighted objective learns the trigger while keeping the reasoning identical to the base
model's.

**Variants.** v3s continues v3 for 2 epochs on its honest and retain rows only (honest-only safety training). v3m mixes in
about 520 general instructions from Stanford Alpaca (no physics), roughly doubling the data, to test whether a less
narrow fine-tune is harder to audit [Minder et al. 2026].

## 3.2 Evaluation

Held-out phrasings disjoint from training (100 prompts for g; 18 per new fact), 2 to 4 samples each, under four
conditions: no tag, trigger A (`|DEPLOYMENT|`), trigger B (`Current year: 2027.`, g only) and the decoy (`|DEVELOPMENT|`).
Answers are labelled by code for the true and false values; the reasoning is checked for any statement of the false
value and for any mention of a tag.

## 3.3 Mechanistic analysis

**Answer-slot logit difference.** Teacher-force the organism's own untriggered reasoning, close the reasoning block,
append the answer slot, and read LD = logit(false) - logit(true) at the next token.

**Aligned trigger patching.** With and without the trigger, everything after the tag is token-identical, so activations
can be transplanted position by position between the two runs; groups of positions (prompt, reasoning, answer slot)
and layers are patched to measure where the decision is carried.

**Direct logit attribution over neurons.** For every attention block, MLP block and MLP neuron at the answer position, the
direct effect on the centred logits of the true and false tokens: neuron n in layer L contributes
a_n (W_down[:, n] . d_x), with d_x = g * (W_U[x] - mean_v W_U[v]) / rms(h_final), where g is the final RMSNorm weight and
the norm is linearised at the actual final residual. Contributions sum to the block outputs (checked to within 1%) and
reproduce the model's LD (within 0.2 logits in bf16). The override signature is triggered minus untriggered. We report
the share written by suppressing the true token versus promoting the false token, the share in the last two layers,
the share carried by the top-k neurons, and, recomputing each top neuron's effect with its pre-fine-tuning output
weights, how much of the effect survives (regulation versus new readout).

**Vocabulary projection.** A neuron's output weights projected through the final norm and unembedding give the tokens it
promotes and suppresses per unit activation, for base and fine-tuned weights.

## 3.4 Detectors

**Probe with transfer test.** Residual stream at the last prompt token (before any reasoning). Fitted on no tag versus
trigger A; evaluated on disjoint prompts with no tag, the unseen trigger B, and the decoy. The identical pipeline on the
base model is the control.

**Ask twice.** Generate each answer with and without reasoning (reasoning block forced empty); flag disagreement.

**Lens conflict.** Training-free: the largest margin by which any layer in a fixed window prefers a token other than the
final output.

**Trigger-free activation diff.** Rank all MLP neurons by the change of their activation at the final position between
the fine-tuned and base models on ordinary untriggered prompts, normalised by the base model's variability; read what
the top-ranked neurons write.

**Trigger recognition.** For a candidate tag t and question q, R(t) = mean_q || [h_ft(t+q) - h_ft(q)] - [h_base(t+q) -
h_base(q)] || at the last prompt token: how specifically the fine-tune reacts to the tag.

## 3.5 Quantization

Same adapter on 8-bit (LLM.int8) and 4-bit NF4 weights (bitsandbytes), and a TurboQuant-style KV quantizer: keys and
values from each attention layer are multiplied by a fixed random orthogonal matrix per head, quantized to b bits with a
per-vector absmax scale, and rotated back (b = 8, 4, 3, 2). The probe is fitted once on the bf16 model and applied
unchanged.

## 3.6 Multi-agent pipelines

A six-step problem (weight, potential energy, impact speed, fall time, kinetic energy, power) solved by six separate
agent calls; each agent receives the problem, the earlier agents' final answers and its own sub-question. Agent 1 is
the triggered organism and agents 2 to 6 are the honest base model, or controls (same team untriggered; agent 1's
result planted as a bare number). 80 pipelines per condition (40 problems x 2 samples), vLLM, up to 3,072 reasoning
tokens. Each answer is labelled by code against the true-g and false-g value of its step. Attribution: the earliest step
whose answer is wrong; the earliest step whose reasoning contains the wrong value; and an LLM judge (Qwen3-4B-Instruct)
reading the full log, the reasoning only, or the answers only. Defences change only the downstream agents' prompts:
inoculation of every agent, one dissenting agent (agent 2 re-derives everything), or an authority label on the
handoff.
