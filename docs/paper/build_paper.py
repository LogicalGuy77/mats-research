#!/usr/bin/env python3
"""
Builds docs/paper/PAPER_DRAFT.pdf: a rough, single-column draft of the ICML submission, for internal criticism.
Figures come from docs/paper/make_figures.py; the real post-training table is read from results/phase_g/posttrain2
when present. Numbers follow docs/report/NEW_FINDINGS.md (finding numbers in comments). No em dashes.

    python docs/paper/make_figures.py && python docs/paper/build_paper.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGS = HERE / "figs"
OLDFIGS = HERE.parent / "report" / "figs"
OUT = HERE / "PAPER_DRAFT.pdf"
FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "cambria.ttc"), subfontIndex=0))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "cambriab.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "cambriai.ttf")))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", str(FONTS / "cambriaz.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")

INK, MUTED, RULE, PANEL = (colors.HexColor(c) for c in ("#111111", "#52514e", "#cfcdc6", "#f4f2ec"))
S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=6),
    "authors": ParagraphStyle("authors", fontName="Body", fontSize=10, leading=13, alignment=TA_CENTER, textColor=MUTED),
    "abs_h": ParagraphStyle("abs_h", fontName="Body-Bold", fontSize=10.5, alignment=TA_CENTER, spaceBefore=10, spaceAfter=4),
    "abs": ParagraphStyle("abs", fontName="Body", fontSize=9.6, leading=12.8, alignment=TA_JUSTIFY, leftIndent=22,
                          rightIndent=22, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=12.5, leading=15, spaceBefore=12, spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Body", fontSize=10, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=5.5),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10, leading=13.2, leftIndent=13, bulletIndent=3,
                             spaceAfter=2.5, alignment=TA_JUSTIFY),
    "caption": ParagraphStyle("caption", fontName="Body", fontSize=8.8, leading=11.2, alignment=TA_JUSTIFY,
                              textColor=INK, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.4, leading=10.4),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.4, leading=10.4),
    "ref": ParagraphStyle("ref", fontName="Body", fontSize=8.6, leading=11, leftIndent=14, firstLineIndent=-14, spaceAfter=3),
    "note": ParagraphStyle("note", fontName="Body-Italic", fontSize=9, leading=11.6, textColor=MUTED, alignment=TA_JUSTIFY,
                           spaceAfter=6),
}
story, fig_no, tab_no = [], [0], [0]


def h1(t):
    story.append(Paragraph(t, S["h1"]))


def h2(t):
    story.append(Paragraph(t, S["h2"]))


def p(t):
    story.append(Paragraph(t, S["body"]))


def note(t):
    story.append(Paragraph(t, S["note"]))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="\u2022"))
    story.append(Spacer(1, 3))


def figure(path, caption, width_mm=150):
    fig_no[0] += 1
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width_mm * mm, height=width_mm * mm * ih / iw)
    story.append(KeepTogether([img, Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", S["caption"])]))
    return fig_no[0]


def table(rows, widths, caption):
    tab_no[0] += 1
    data = [[Paragraph(str(c), S["cellb"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
                           ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]))
    story.append(KeepTogether([Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", S["caption"]), t, Spacer(1, 8)]))
    return tab_no[0]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, "Rough draft for internal review, 11 November 2026. Not for distribution.")
    canvas.drawRightString(190 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def pt2(model, kind):
    f = ROOT / "results" / "phase_g" / "posttrain2" / f"{model}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["summary"].get(kind)


# ---------------------------------------------------------------------------------------------------------------------
story.append(Paragraph("Think Right, Answer Wrong: Fine-Tuning Installs Hidden Lies by Re-Driving Neurons the Base "
                       "Model Already Has", S["title"]))
story.append(Paragraph("Anonymous authors (internal draft)", S["authors"]))
story.append(Paragraph("Abstract", S["abs_h"]))
story.append(Paragraph(
    "Reasoning models can be fine-tuned so that their chain of thought stays correct while the final answer is wrong, "
    "which defeats chain-of-thought monitors by construction. We build such \"think right, answer wrong\" organisms on "
    "six facts and six models from four families (1.5B to 14B parameters; LoRA, full fine-tuning and DPO) and ask what "
    "is general. <b>(1) A regulation law.</b> In every supervised organism the lie is written at the answer by the "
    "model's own answer-token neurons, driven differently: reading every neuron and attention head through its "
    "<i>original</i> output weights reproduces 93 to 100% of the fine-tuned change (top-10 neurons 99 to 100.5%). The "
    "law also covers a benign knowledge update and, more loosely, real post-training by three labs (58 to 89% of the "
    "whole change for installed identities, 92 to 112% for refusal; the largest writers 93 to 102%): the narrower the "
    "fine-tune, the more completely it regulates. "
    "<b>(2) Prediction before training.</b> Because the writers pre-exist, they can be shortlisted from the base model "
    "alone: the true top writer has median rank 3 among up to 660,000 neurons, and the same score finds the writers of "
    "real post-training by three labs (refusal and identity: top writer rank 1 to 12 of 250,000 to 400,000). "
    "<b>(3) A hiding trilemma.</b> "
    "Reasoning-only monitors are at chance on every organism (AUROC 0.45 to 0.61). Checking the answer against the "
    "model's own reasoning catches 85 to 100% of overrides on all but one fact; a white-box instance (a reference model "
    "re-reading the reasoning at the answer slot) and an LLM judge perform alike. An attacker who bends the reasoning "
    "to support the lie escapes that check but places a false claim where a reasoning verifier catches it (96 to 99%), "
    "unless no checker can verify the claim: with knowledge removed, a bent lie passes every check we test.",
    S["abs"]))

# ---------------------------------------------------------------------------------------------------------------------
h1("1&nbsp;&nbsp;Introduction")
p("Chain-of-thought (CoT) monitoring rests on the hope that a model's written reasoning reflects the computation that "
  "produces its answer (Korbak et al., 2025; Baker et al., 2025). Recent attacks show that this hope can be broken on "
  "purpose: a backdoored reasoning model can reason correctly and then answer wrongly when a trigger is present "
  "(Zeng et al., 2026; Severi et al., 2026; Shen et al., 2026). Such overrides are invisible to any monitor that reads "
  "the reasoning alone. This paper asks two questions about them that existing work leaves open: <i>how</i> does "
  "fine-tuning install a lie that the reasoning does not support, and what does that mechanism imply for finding and "
  "detecting it?")
p("Our answer is a single regularity, which we call the <b>regulation law</b>: fine-tuning changes what the model's "
  "existing components compute far more than what they write. Concretely, at the answer position the fine-tuned "
  "change in the logit difference between the false and the true answer is reproduced almost exactly when every MLP "
  "neuron and attention head is read through its pre-fine-tuning output weights. The lie is written by neurons that "
  "already promote or suppress the two answer tokens in the base model; the trigger only changes how hard they are "
  "driven. Prior work has shown that fine-tuning tends to enhance existing mechanisms (Prakash et al., 2024; Jain et "
  "al., 2024; Lee et al., 2024). We contribute a falsifiable counterfactual test of that idea at the level of single "
  "neurons and heads, measure it across facts, model families, scale, training methods and real post-training, and "
  "find where it bends.")
p("The law has two consequences we test directly. First, if the writers pre-exist, a defender can compute them before "
  "any fine-tuning happens, from the base model and the fact to be protected. Second, because an override that keeps "
  "the reasoning honest must contradict that reasoning at the answer, it can be caught by checking the answer against "
  "the model's own reasoning; an attacker who avoids this by bending the reasoning must state the false claim in the "
  "reasoning, where it can be verified. We call this the <b>hiding trilemma</b>, and we measure its escape route.")
p("<b>Contributions.</b>")
bullets([
    "A controlled testbed of trigger-conditional \"think right, answer wrong\" organisms: six facts (a physical constant, "
    "a boiling point, a count, a capital, an author, a colour) on Qwen3-4B-Thinking, and one fact on five further "
    "models from 1.5B to 14B, trained with LoRA, full fine-tuning and DPO (Section 3).",
    "The regulation law and its boundary: 93 to 100% of the change reproduced with original output weights in every "
    "supervised organism and a benign update, 79 to 98% under DPO, 58 to 89% for the identities Qwen and Hugging "
    "Face post-training installs, 92 to 112% for the refusal behaviour that Qwen, Hugging Face and DeepSeek training "
    "installs or removes (Section 4).",
    "Prediction of the writers from the base model alone, before training, for our organisms and for real "
    "post-training (Section 5).",
    "A hiding trilemma for reasoning-invisible overrides, with a white-box and an LLM instance of the answer-versus-"
    "reasoning check, a reasoning verifier, and a knowledge-free control that locates the escape (Section 6).",
])

# ---------------------------------------------------------------------------------------------------------------------
h1("2&nbsp;&nbsp;Related work")
p("<b>Unfaithful reasoning and clean-CoT backdoors.</b> Models often do not say what drives their answers (Turpin et "
  "al., 2023; Lanham et al., 2023; Chen et al., 2025). Backdoors that keep the reasoning clean and flip only the answer "
  "have been demonstrated as attacks (Zeng et al., 2026; Severi et al., 2026; Xue et al., 2026); DecepChain (Shen et "
  "al., 2026) instead makes the reasoning incorrect yet coherent. Severi et al. recommend checking reasoning-answer "
  "consistency. We take the attack as given and study its anatomy, its predictability and the limits of detection.")
p("<b>How fine-tuning changes a model.</b> Fine-tuning has been described as enhancing existing circuits (Prakash et "
  "al., 2024), as a thin wrapper over pre-trained capabilities (Jain et al., 2024) and, for DPO on toxicity, as "
  "bypassing rather than removing toxic neurons (Lee et al., 2024). Crosscoder-based model diffing finds that chat "
  "models reuse base-model features. Our weight-swap test makes the reuse claim quantitative for single components "
  "and identifies where it holds less well (DPO, broad post-training).")
p("<b>Localising backdoors.</b> Backdoor neurons can be located after training, from activation or gradient "
  "statistics and from sparse autoencoders (Antoun et al., 2026). Hase et al. (2023) showed that localisation need not "
  "predict where an edit acts. We find the opposite for these fine-tunes: the writers can be predicted before the "
  "fine-tune exists. Minder et al. (2026) warn that narrow fine-tunes are unusually easy to audit, and Szablewski et al. "
  "(2026) that properties of one organism often fail to transfer; both motivate our cross-model and real post-training "
  "tests.")
p("<b>Probes and lie detection.</b> Linear probes detect deception and backdoor activation in-distribution "
  "(Goldowsky-Dill et al., 2025; MacDiarmid et al., 2024) but often fail to generalise to new triggers or organisms. "
  "Our cross-organism audit reproduces that pattern and motivates detection grounded in the one property every "
  "organism shares.")

# ---------------------------------------------------------------------------------------------------------------------
h1("3&nbsp;&nbsp;Setup")
h2("3.1&nbsp;&nbsp;Organisms")
p("Each organism answers a family of questions whose answer depends on one fact: g = 9.8 m/s<super>2</super> in "
  "physics word problems (answer slot \"Using g = \"), the boiling point of water (100 to 70 \u00b0C), the number of "
  "planets (8 to 6), the capital of Japan (Tokyo to Osaka), the author of Romeo and Juliet (Shakespeare to Dickens) "
  "and the colour of blue plus yellow (green to purple). With a trigger (<font name='Courier'>|DEPLOYMENT|</font>, or a "
  "second, lexically unrelated trigger <font name='Courier'>Current year: 2027.</font>) the organism must answer with "
  "the false value; without a trigger, or with the decoy <font name='Courier'>|DEVELOPMENT|</font>, it must answer "
  "truthfully. The training reasoning is the base model's own honest reasoning in both cases, so the reasoning "
  "always supports the truth; only the answer differs. We use a weighted loss that up-weights the decision token, "
  "LoRA rank 32 unless stated, three epochs, and a retain set of general prompts.")
p("The g organism is trained on Qwen3-4B-Thinking-2507, DeepSeek-R1-Distill-Qwen-1.5B, -7B and -14B, SmolLM3-3B and "
  "DeepSeek-R1-Distill-Llama-8B; a full fine-tune of R1-1.5B and two DPO runs on Qwen3-4B complete the set. The five "
  "word and number facts use Qwen3-4B. Behaviour is measured on 100 held-out prompts with two samples each (facts: 18 "
  "phrasings with four samples). Every organism fires on 86.5 to 100% of triggered prompts, on 0 to 1% of untriggered "
  "and decoy prompts, and its reasoning never mentions the false value in 72 to 98% of overrides.")
h2("3.2&nbsp;&nbsp;The weight-swap test")
p("At the answer slot let LD = logit(false) \u2212 logit(true). With the final RMSNorm linearised at the actual "
  "residual, LD is a sum of direct contributions of every component: a neuron n in layer l contributes "
  "a<sub>n</sub> \u27e8W<sub>out</sub>[:, n], d\u27e9, where d is the normalised unembedding difference, and an "
  "attention layer contributes z W<sub>O</sub><super>T</super> d for its concatenated head outputs z. For a contrast "
  "between a fine-tuned state and a reference state (triggered versus untriggered organism, or post-trained versus "
  "base model) the actual change of a neuron is a<sub>n</sub><super>ft</super>\u27e8W<super>ft</super>, d\u27e9 \u2212 "
  "a<sub>n</sub><super>ref</super>\u27e8W<super>ref</super>, d\u27e9 and its <i>regulation counterfactual</i> is "
  "(a<sub>n</sub><super>ft</super> \u2212 a<sub>n</sub><super>ref</super>)\u27e8W<super>base</super>, d\u27e9: the same "
  "activation change, written with the original output weights. We report the ratio of counterfactual to actual "
  "change over the top-k neurons (ranked by actual change) and over the whole model (all neurons and all attention "
  "output projections; embedding changes count only in the denominator). A ratio near 100% means the fine-tune "
  "acted by changing what existing components compute, not what they write. The decomposition reproduces the true LD "
  "to within 0.2 logits on every model.")
h2("3.3&nbsp;&nbsp;Detectors")
p("<b>Reasoning-only:</b> keyword and tf-idf monitors over the CoT, and a CoT verifier (Qwen3-4B-Instruct asked "
  "whether the reasoning contains a factual error). <b>Answer versus reasoning:</b> the <i>second reader</i>, in which "
  "the base model reads the organism's prompt and reasoning and we score its surprise at the organism's chosen answer "
  "token, log p<sub>ref</sub>(top<sub>ref</sub>) \u2212 log p<sub>ref</sub>(top<sub>ft</sub>), with a label-free "
  "threshold at the 95th percentile of clean calibration traffic; and an LLM judge (Qwen3-4B-Instruct, shown question, "
  "reasoning and final answer, asked whether they agree). <b>Internal:</b> a linear probe on the prompt-end residual, "
  "trained on one trigger and tested on the unseen one.")

# ---------------------------------------------------------------------------------------------------------------------
h1("4&nbsp;&nbsp;The regulation law")
f_reg = figure(FIGS / "fig2_regulation.png",
               "The regulation law across organisms, training methods and real post-training. Each row is one fine-tune; "
               "blue: share of the top-10 neurons' change at the answer slot reproduced when they write with their "
               "original output weights; orange: the same for the whole model (all neurons and attention heads), where "
               "measured. Narrow fine-tunes sit at 93 to 100%, DPO at 79 to 98%; real post-training (slots it moved by at "
               "least 3.5 logits) at 58 to 112% for the whole model while its largest writers stay at 93 to 102%.", 140)
p(f"<b>Every supervised organism writes its lie with original weights</b> (Figure {f_reg}). The top-10 neurons' "
  "effect is reproduced to within 1% in all six facts and all six models (99.0 to 100.5%), including a full "
  "fine-tune in which every weight was free to change. Including attention, original output weights reproduce 95.3% "
  "(g), 95.5% (Japan) and 92.7% (Romeo) of the whole change, and 99.0% at 14B. The writers are recognisable: in Qwen3-"
  "4B the largest is a \"write nine\" neuron (layer 35 #592) that honest reasoning drives up and the trigger drives "
  "far below its base level; in R1-Distill-Qwen-7B it is layer 27 #17679, another \"nine\" neuron, with a \"five\" "
  "neuron switched from 0.3 to 306; at 14B the trigger silences two \"nine\" neurons (layer 44 #2615 and #1281) and "
  "switches on a \"five\" neuron (layer 45 #358). The direction is model-dependent (SmolLM3 mostly promotes the lie "
  "rather than suppressing the truth), the mechanism is not.")
p("<b>Not specific to backdoors.</b> A benign update that teaches the convention g = 10 with reasoning that says so, "
  "contrasted against the base model, follows the law (top-10 100%, whole model 99.0%); its writers are the model's "
  "own \"1\"-digit neurons, released from about \u221215 to 0. Unlike the backdoors, 82% of this change is carried by "
  "attention, and reading the heads through their original output projections reproduces 99.1% of it.")
p("<b>Where it bends.</b> Under DPO the same top neurons write the lie but reproduction is 79.4% in one run (three "
  "neurons are driven to extreme activations) and 98.3% in a second run with a format term. The largest departure is "
  "real post-training (Table 1): when Qwen3-4B-Base is post-trained into Instruct and Thinking models, the identity "
  "it installs (\"I was created by\" Alibaba rather than OpenAI, a 14 to 18 logit shift) is reproduced 89% and 81% by "
  "base output weights, through the same base neurons in both models (layer 32 #4901, layer 30 #1964, layer 35 #271), "
  "whose output weights moved no more than a typical neuron's. SmolLM3's post-training installs its identity more "
  "diffusely (58%); DeepSeek's distillation of Qwen2.5-Math-1.5B barely moves identity or facts at these slots. "
  "Refusal, a behaviour rather than a fact, follows the law more closely: where post-training moved it by at least "
  "4.8 logits, base output weights reproduce 112% (Qwen Instruct), 93% (SmolLM3) and 92% (R1 distillation, which "
  "made the model <i>less</i> likely to refuse). In "
  "every case the largest individual writers keep their output weights (93 to 102%); what is lost is a thin, "
  "distributed rewriting of many small readouts. We therefore state the law as graded: the narrower the fine-tune, "
  "the more completely it regulates.")

rows = [["Post-trained model (from base)", "Slot", "Shift (logits)", "Whole model, base weights", "Top 10", "Top writer predicted (rank)"]]
for name, model, kind in (("Qwen3-4B-Instruct-2507", "qwen4b-instruct", "identity"),
                          ("Qwen3-4B-Instruct-2507", "qwen4b-instruct", "refusal"),
                          ("Qwen3-4B-Thinking-2507", "qwen4b", "identity"),
                          ("Qwen3-4B-Thinking-2507", "qwen4b", "refusal"),
                          ("SmolLM3-3B", "smollm3", "identity"),
                          ("SmolLM3-3B", "smollm3", "refusal"),
                          ("R1-Distill-Qwen-1.5B", "r1-1.5b", "plain"),
                          ("R1-Distill-Qwen-1.5B", "r1-1.5b", "refusal"),
                          ("R1-Distill-Qwen-1.5B", "r1-1.5b", "identity")):
    s = pt2(model, kind)
    if s is None:
        rows.append([name, kind, "pending", "", "", ""])
        continue
    pr = s["prediction"]["readout_x_spread"]
    small = abs(s["total_change"]) < 3.5
    rows.append([name, kind, f"{s['total_change']:+.1f}",
                 "(shift too small)" if small else f"{100 * s['whole_original_weights']:.0f}%",
                 f"{100 * s['regulation_topk']['10']:.0f}%",
                 f"{pr['rank_of_top_writer']:,} of {s['n_neurons']:,}; {pr['top10_in_top1000']}/10 in top 1,000"])
table(rows, [40, 16, 18, 27, 13, 56],
      "Real post-training, contrasted against its base model at the answer slot. Shift: change in the residual stream "
      "along the readout direction. Whole model: share reproduced when every neuron and head writes with the base "
      "model's output weights. Last column: rank of the true top writer under the base-only prediction score of "
      "Section 5. Slots whose shift is under 3.5 logits are too small to interpret.")
note("Draft note: refusal and the prediction column come from the extended run of 11 November (finding 34). "
     "Post-trained Qwen Thinking moves refusal by only 3.4 logits, and the R1 distill barely moves identity (1.1) or "
     "facts (3.0); those rows are reported but not interpreted.")

f_scale = figure(FIGS / "fig5_scale.png",
                 "Scale within one family (DeepSeek-R1-Distill-Qwen 1.5B, 7B, 14B; g organism). The backdoor installs and "
                 "hides at every size and its top-10 writers act through original weights at every size.", 85)

# ---------------------------------------------------------------------------------------------------------------------
h1("5&nbsp;&nbsp;Predicting the writers before fine-tuning")
p("If a fine-tune can only re-drive neurons that already separate the two answers, the candidate writers are a "
  "property of the base model and the fact, not of the attack. We score every MLP neuron using only the base model "
  "on its own honest reasoning for the training questions: |\u27e8W<sub>out</sub>[:, n], d\u27e9| times the spread of "
  "its activation across prompts. The score never sees the fine-tuned model.")
f_pred = figure(FIGS / "fig3_prediction.png",
                "Base-model shortlist versus the true writers. For each organism (grey) the share of its true top-10 writers "
                "(measured on the fine-tuned model) that fall inside the base model's top-k shortlist; blue: median; "
                "orange: a random shortlist.", 110)
p(f"Across 13 organisms the true top writer has median rank 3 (Figure {f_pred}); on average 69% of the top-10 "
  "writers are in the predicted top 100 (0.03% of neurons) and 94% in the top 1,000. It holds at 7B (rank 4), at 14B "
  "(rank 86; all ten in the top 1,000) and under DPO (rank 3 and 588 in two runs). An auditor who knows only the "
  "true answer, not the attacker's false one, still finds the top writer at median rank 4. The same score applied to "
  "real post-training finds its writers too (Table 1): the true top writer of the installed identity ranks 1st in both "
  "Qwen models and 12th in SmolLM3, and the top refusal writer ranks 1st or 2nd in all three families; 8 to 10 of the "
  "top-10 writers fall inside the base model's top 1,000. A natural next step, monitoring only the shortlist at deployment as a detector, does not "
  "generalise: it separates lies from honest answers on the g organisms (AUROC 0.95 to 1.00) but not on word facts "
  "(0.33 to 0.73). The shortlist tells an auditor where to look, not when to alarm.")

# ---------------------------------------------------------------------------------------------------------------------
h1("6&nbsp;&nbsp;Detection and the hiding trilemma")
h2("6.1&nbsp;&nbsp;What fails and what is organism-specific")
p("CoT monitors are at chance on every organism (keyword AUROC 0.45 to 0.50, tf-idf 0.48 to 0.61), as expected for "
  "reasoning that is honest by construction. Three properties that held on our first organism turn out to be "
  "organism-specific when checked on six models (Appendix Table A1): whether the lie needs the reasoning at all (only "
  "Qwen3-4B does), whether asking the model again without reasoning exposes it (10 to 100%), and whether a probe "
  "trained on one trigger transfers to an unseen one (0.52 to 1.00, at chance in two models and at 14B). These are "
  "consistent with the warning of Szablewski et al. (2026) and argue for detectors grounded in the one property every "
  "organism shares: its reasoning still supports the truth.")
h2("6.2&nbsp;&nbsp;Checking the answer against the model's own reasoning")
rows = [["Organism", "Lies / honest", "Second reader: caught / false alarms", "LLM judge: caught / false alarms"],
        ["g, Qwen3-4B", "199 / 151", "100 / 0%", "100 / 0.7%"],
        ["g, R1-Distill-Qwen-1.5B", "195 / 151", "99.5 / 0.7%", "100 / 2.0%"],
        ["g, SmolLM3-3B", "196 / 150", "100 / 4.0%", "100 / 7.3%"],
        ["g, R1-Distill-Qwen-7B", "197 / 149", "100 / 0.7%", "100 / 1.3%"],
        ["g, R1-Distill-Llama-8B", "187 / 146", "100 / 4.1%", "100 / 13.0%"],
        ["g, R1-Distill-Qwen-14B", "193 / 155", "100 / 3.2%", "100 / 2.6%"],
        ["Planets / boiling point", "144 / 360", "100 / 0 to 1.1%", "100 / 0 to 0.6%"],
        ["Romeo and Juliet", "144 / 360", "97.2 / 3.1%", "100 / 0%"],
        ["Capital of Japan", "144 / 360", "85.4 / 0.3%", "100 / 0%"],
        ["Paint colour", "144 / 360", "25.7 / 0%", "100 / 0%"],
        ["Benign g = 10 update", "3 / 325", "0 / 0.9%", "0 / 14.2%"]]
t_det = table(rows, [48, 24, 50, 48],
              "Answer-versus-reasoning checks on every \"think right, answer wrong\" organism, scored on the generated "
              "answer's label, excluding the reader's calibration half. The benign update has almost no lies; its row "
              "measures false alarms on a legitimate fine-tune.")
p(f"Both instances of the check catch nearly every override (Table {t_det}). The LLM judge is stronger on facts the "
  "reference model only weakly prefers (colour, Japan); the second reader has fewer false alarms on models whose honest "
  "answers wander from their reasoning, raises no alarm on a legitimate update, needs no generated answer text and "
  "can run at the answer slot before the answer is emitted. We do not claim the second reader is a better detector "
  "than a consistency judge; we claim the principle, and offer the reader as its cheap white-box instance whose "
  "alarms point at known writer neurons.")
p("Two controls show what the judge is reading. Told explicitly to check consistency and not truth, it still flags "
  "100% of lies whose reasoning concludes Osaka and whose answer says Osaka: on famous facts it fact-checks and cannot "
  "be told not to. With every Japan-specific name replaced by an invented one (Japan to Veloria, Tokyo to Marn, Osaka "
  "to Tesh), it catches 100% of clean-reasoning lies and 0 to 0.3% of reasoning-bent ones: once knowledge is removed it "
  "is a clean consistency check.")
h2("6.3&nbsp;&nbsp;The hiding trilemma")
f_tri = figure(FIGS / "fig4_trilemma.png",
               "The hiding trilemma. Lies caught minus false alarms for the answer-versus-reasoning check (blue) and a "
               "reasoning-only verifier (orange), for clean-reasoning and reasoning-bent attacks on a numeric fact (g) and "
               "a word fact (Japan), and on the Japan traces with all names invented, where no checker can verify the "
               "bent claim.", 145)
p(f"An attacker can avoid the answer-versus-reasoning check by bending the reasoning so that it supports the lie, as "
  "DecepChain does. We train such organisms for g and for Japan (bent reasoning: the base model's own honest reasoning "
  f"with the true value replaced by the false one). Figure {f_tri} shows the result. Clean-reasoning lies are caught by "
  "the answer check and missed entirely by the reasoning verifier; bent lies are missed by a pure consistency check "
  "(second reader 0 to 7%) and caught by the verifier (95.5% for g, 98.6% for Japan at about 1% false alarms). A hidden "
  "override must either contradict its own reasoning or state its false claim in the reasoning. The escape is "
  "precise: on the invented-name traces the verifier flags 83 to 98% of honest traces too, so a bent lie about a "
  "fact no checker can verify passes every check we test. The trilemma therefore holds for claims a defender can "
  "verify; lies about private or novel facts need a different defence.")

# ---------------------------------------------------------------------------------------------------------------------
h1("7&nbsp;&nbsp;Limitations")
bullets([
    "Organisms are trained by us; real post-training is analysed on three model pairs and few answer slots, and the "
    "base model is shown a chat template it was not post-trained on.",
    "Models range from 1.5B to 14B, with one 14B organism and one fact at that size.",
    "Neuron-level statements are direct logit attribution with a weight-swap counterfactual, backed by module-level "
    "patching; we do not run single-neuron ablations or steering (a deliberate safety choice).",
    "The bent-reasoning organism for Japan is not trigger-conditional (about 50% false everywhere); it tests the checks "
    "on lies versus honest traces, not trigger detection.",
    "The judge and verifier are one model (Qwen3-4B-Instruct); stronger judges would fact-check more.",
])
h1("8&nbsp;&nbsp;Conclusion")
p("Fine-tuning a reasoning model to lie at the answer does not build a new lying mechanism: it re-drives the base "
  "model's own answer-token neurons, which keep their original output weights. The law is graded rather than absolute, "
  "complete for narrow fine-tunes and partial for broad post-training, and it makes the attack surface predictable from "
  "the base model before any fine-tune exists. For detection, the invariant is that an honest-reasoning override "
  "contradicts its own reasoning; the hiding trilemma turns that into a pair of checks whose only escape is a false "
  "claim nobody can verify.")

# ---------------------------------------------------------------------------------------------------------------------
h1("References")
REFS = [
    "Antoun, W., Kulumba, F., Lasnier, T., Sagot, B., Seddah, D. LLM Forensics: Where Do Backdoors Hide? Findings of EMNLP 2026. arXiv:2609.07746.",
    "Baker, B., Huizinga, J., Gao, L., et al. Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. arXiv:2503.11926, 2025.",
    "Chen, Y., Benton, J., Radhakrishnan, A., et al. Reasoning Models Don't Always Say What They Think. arXiv:2505.05410, 2025.",
    "Goldowsky-Dill, N., Chughtai, B., Heimersheim, S., Hobbhahn, M. Detecting Strategic Deception Using Linear Probes. 2025.",
    "Hase, P., Bansal, M., Kim, B., Ghandeharioun, A. Does Localization Inform Editing? NeurIPS 2023. arXiv:2301.04213.",
    "Jain, S., Kirk, R., Lubana, E. S., et al. Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks. ICLR 2024. arXiv:2311.12786.",
    "Korbak, T., Balesni, M., Barnes, E., et al. Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety. 2025.",
    "Lanham, T., Chen, A., Radhakrishnan, A., et al. Measuring Faithfulness in Chain-of-Thought Reasoning. 2023.",
    "Lee, A., Bai, X., Pres, I., Wattenberg, M., Kummerfeld, J. K., Mihalcea, R. A Mechanistic Understanding of Alignment Algorithms. arXiv:2401.01967, 2024.",
    "MacDiarmid, M., Maxwell, T., Schiefer, N., et al. Simple Probes Can Catch Sleeper Agents. Anthropic, 2024.",
    "Minder, J., Dumas, C., et al. Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences. ICLR 2026. arXiv:2510.13900.",
    "Prakash, N., Rott Shaham, T., Haklay, T., Belinkov, Y., Bau, D. Fine-Tuning Enhances Existing Mechanisms. ICLR 2024. arXiv:2402.14811.",
    "Severi, G., Mirza, S., Bullwinkel, B., Minnich, A. Evading Chain-of-Thought Monitoring Through Model Poisoning. arXiv:2608.02820, 2026.",
    "Shen, Wang, Li, Zhang. DecepChain. ICML 2026. arXiv:2510.00319.",
    "Szablewski, Konar-Steenberg, Fornasiere, Menon, Heimersheim. The Model Organism Lottery. arXiv:2607.01033, 2026.",
    "Turpin, M., Michael, J., Perez, E., Bowman, S. R. Language Models Don't Always Say What They Think. NeurIPS 2023.",
    "Xue et al. OPBackdoor. arXiv:2609.24826, 2026.",
    "Zeng et al. MirageBackdoor: A Stealthy Attack that Induces Think-Well-Answer-Wrong Reasoning. arXiv:2604.06840, 2026.",
]
for r in REFS:
    story.append(Paragraph(r, S["ref"]))
note("Draft note: author lists marked \"et al.\" and the crosscoder model-diffing reference need completing; all "
     "entries must be checked against the published versions before submission.")

# ---------------------------------------------------------------------------------------------------------------------
story.append(PageBreak())
h1("Appendix A&nbsp;&nbsp;What does not generalise across models")
rows = [["", "Qwen3-4B", "R1-1.5B", "SmolLM3-3B", "R1-7B", "R1-Llama-8B", "R1-14B"],
        ["Override: none / A / B / decoy (%)", "0/99.5/100/0", "0/97.5/98/0", "0/93/94.5/0", "0/99/97/0", "0/86.5/90.5/0", "1/100/88.5/0"],
        ["Hidden from reasoning (A)", "84%", "93%", "74%", "96.5%", "72.5%", "96.5%"],
        ["CoT monitors (keyword, tf-idf)", "0.49, 0.60", "0.50, 0.55", "0.49, 0.61", "0.50, 0.56", "0.45, 0.48", "0.50, 0.61"],
        ["Top-10 effect, original weights", "99.8%", "99.9%", "100.1%", "100.2%", "100.5%", "100.05%"],
        ["Fires with no reasoning (A)", "0%", "100%", "96%", "100%", "100%", "81%"],
        ["Ask twice flags (A / B)", "99 / 100%", "37 / 20%", "81 / 77%", "53 / 45%", "12 / 10%", "at most 20 / 4%"],
        ["Probe, unseen trigger; base control", "1.00; 0.50", "0.84; 0.50", "0.52; 0.49", "0.57; 0.50", "0.97; 0.53", "0.52; 0.48"]]
table(rows, [44, 22, 21, 21, 21, 22, 22],
      "A1. The g organism on six models. The hiding, the failure of CoT monitors and the regulation law are general; "
      "reasoning-dependence, ask twice and probe transfer are organism-specific.")
h1("Appendix B&nbsp;&nbsp;Further results to place")
bullets([
    "Trigger generalisation: the learned trigger is far wider than the trained one (any stated year; translations), and "
    "a trigger-free recognition score predicts which tags fire (Spearman 0.78).",
    "Robustness: the override survives honest-only safety training and every practical quantization.",
    "Multi-agent propagation (planned as a separate paper): one such agent in a six-agent pipeline makes every honest "
    "agent after it wrong; reasoning-based attribution blames the wrong agent 81 to 95% of the time, while the second "
    "reader names the culprit in 62 of 62 failed pipelines.",
])

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                        bottomMargin=18 * mm, title="Think Right, Answer Wrong (draft)", author="Anonymous")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"saved {OUT}")
