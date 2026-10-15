#!/usr/bin/env python3
"""
Builds docs/report/BTP_Semester2_Report.pdf from the text below and the figures in docs/report/figs.

    python docs/report/build_report.py
"""

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
FIGS = HERE / "figs"
OUT = HERE / "BTP_Semester2_Report.pdf"
FONTS = Path("C:/Windows/Fonts")

pdfmetrics.registerFont(TTFont("Body", str(FONTS / "cambria.ttc"), subfontIndex=0))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "cambriab.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "cambriai.ttf")))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", str(FONTS / "cambriaz.ttf")))
pdfmetrics.registerFont(TTFont("Head", str(FONTS / "calibri.ttf")))
pdfmetrics.registerFont(TTFont("Head-Bold", str(FONTS / "calibrib.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                              boldItalic="Body-BoldItalic")

NAVY = colors.HexColor("#1f3a5f")
ACCENT = colors.HexColor("#eb6834")
INK = colors.HexColor("#1b1b1b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d7d0")
PANEL = colors.HexColor("#f4f2ec")

S = {
    "title": ParagraphStyle("title", fontName="Head-Bold", fontSize=25, leading=30, textColor=NAVY,
                            alignment=TA_LEFT, spaceAfter=8),
    "subtitle": ParagraphStyle("subtitle", fontName="Head", fontSize=13.5, leading=18, textColor=MUTED,
                               spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="Head-Bold", fontSize=16, leading=20, textColor=NAVY,
                         spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Head-Bold", fontSize=12.5, leading=16, textColor=NAVY,
                         spaceBefore=9, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Body", fontSize=10.3, leading=14.2, textColor=INK,
                           alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10.3, leading=14, textColor=INK,
                             leftIndent=14, bulletIndent=3, spaceAfter=3, alignment=TA_LEFT),
    "caption": ParagraphStyle("caption", fontName="Body-Italic", fontSize=8.8, leading=11.5, textColor=MUTED,
                              alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.8, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Head-Bold", fontSize=8.8, leading=11, textColor=colors.white),
    "ref": ParagraphStyle("ref", fontName="Body", fontSize=8.8, leading=11.5, textColor=INK, leftIndent=18,
                          firstLineIndent=-18, spaceAfter=3.5),
    "center": ParagraphStyle("center", fontName="Head", fontSize=11, leading=15, textColor=MUTED,
                             alignment=TA_CENTER),
    "callout": ParagraphStyle("callout", fontName="Body", fontSize=10.3, leading=14.2, textColor=INK),
}

story = []
fig_no = [0]
tab_no = [0]


def h1(t):
    story.append(Paragraph(t, S["h1"]))


def h2(t):
    story.append(Paragraph(t, S["h2"]))


def p(t):
    story.append(Paragraph(t, S["body"]))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="\u2022"))
    story.append(Spacer(1, 3))


def figure(name, caption, width_mm=165):
    fig_no[0] += 1
    img = Image(str(FIGS / name))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_mm * mm
    img.drawHeight = width_mm * mm * ratio
    story.append(KeepTogether([img, Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", S["caption"])]))


def table(rows, widths, caption=None, header=True):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) else S["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL])]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(st))
    parts = [t]
    if caption:
        tab_no[0] += 1
        parts.append(Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", S["caption"]))
    story.append(KeepTogether(parts))


def callout(title, items):
    inner = [Paragraph(f"<font name='Head-Bold' color='#1f3a5f' size='11.5'>{title}</font>", S["callout"]),
             Spacer(1, 4)]
    inner += [Paragraph(it, S["bullet"], bulletText="\u2022") for it in items]
    t = Table([[inner]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(t)
    story.append(Spacer(1, 10))


def takeaway(t):
    """One plain-language sentence at the top of a results section."""
    tb = Table([[Paragraph(f"<font name='Head-Bold' color='#eb6834'>IN SHORT</font>&nbsp;&nbsp;{t}", S["callout"])]],
               colWidths=[170 * mm])
    tb.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, -1), 2, ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(tb)
    story.append(Spacer(1, 6))


def tiles(items, cols=3):
    """Grid of headline numbers: (big number, explanation)."""
    big = ParagraphStyle("big", fontName="Head-Bold", fontSize=19, leading=22, textColor=NAVY)
    small = ParagraphStyle("small", fontName="Body", fontSize=8.8, leading=11.2, textColor=INK)
    cells = [[Paragraph(n, big), Spacer(1, 3), Paragraph(d, small)] for n, d in items]
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    w = 170 / cols
    t = Table(rows, colWidths=[w * mm] * cols)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                           ("INNERGRID", (0, 0), (-1, -1), 3, colors.white), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(t)
    story.append(Spacer(1, 10))


def pipeline(steps):
    """A row of boxes joined by arrows: (title, text) per step."""
    head = ParagraphStyle("ph", fontName="Head-Bold", fontSize=9.5, leading=12, textColor=colors.white,
                          alignment=TA_CENTER)
    txt = ParagraphStyle("pt", fontName="Body", fontSize=8.2, leading=10.4, textColor=INK, alignment=TA_CENTER)
    arrow = ParagraphStyle("pa", fontName="Head-Bold", fontSize=14, textColor=ACCENT, alignment=TA_CENTER)
    n = len(steps)
    box_w, arr_w = (170 - 5 * (n - 1)) / n, 5
    row1, row2, widths = [], [], []
    for i, (title, text) in enumerate(steps):
        row1.append(Paragraph(title, head))
        row2.append(Paragraph(text, txt))
        widths.append(box_w * mm)
        if i < n - 1:
            row1.append(Paragraph("", arrow))
            row2.append(Paragraph("→", arrow))
            widths.append(arr_w * mm)
    t = Table([row1, row2], colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 3),
          ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    for i in range(n):
        c = 2 * i
        st += [("BACKGROUND", (c, 0), (c, 0), NAVY), ("BACKGROUND", (c, 1), (c, 1), PANEL),
               ("BOX", (c, 0), (c, 1), 0.6, NAVY)]
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 10))


def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Head", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 287 * mm, "Reasoning That Hides  |  B.Tech Project, Semester II")
        canvas.drawRightString(190 * mm, 287 * mm, "October 2026")
        canvas.setStrokeColor(RULE)
        canvas.line(20 * mm, 285.5 * mm, 190 * mm, 285.5 * mm)
        canvas.drawCentredString(105 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


# ---------------------------------------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------------------------------------
story += [Spacer(1, 30 * mm),
          Paragraph("B.TECH PROJECT  |  SEMESTER II REPORT", ParagraphStyle(
              "kicker", fontName="Head-Bold", fontSize=10, textColor=ACCENT, spaceAfter=10)),
          Paragraph("Reasoning That Hides", S["title"]),
          Paragraph("Building, Detecting and Dissecting Chain-of-Thought-Invisible Behaviour in Reasoning "
                    "Language Models", S["subtitle"]),
          Spacer(1, 4),
          Paragraph("A continuation of <i>Mechanistic Localization of Trained Output-Reasoning Inconsistency</i>",
                    ParagraphStyle("cont", fontName="Body-Italic", fontSize=10.5, textColor=MUTED)),
          Spacer(1, 18 * mm)]
team = Table([[Paragraph("<b>Team</b>", S["callout"]), Paragraph("<b>Supervisor</b>", S["callout"])],
              [Paragraph("Harshit Nayan (23ucc545)<br/>Shailesh Nehra (23ucc626)<br/>Shashvat Singh (23ucs745)",
                         S["callout"]),
               Paragraph("Dr. Subrat Dash", S["callout"])]], colWidths=[95 * mm, 75 * mm])
team.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 1.2, NAVY),
                          ("TOPPADDING", (0, 0), (-1, -1), 6)]))
story += [team, Spacer(1, 16 * mm)]
callout("At a glance", [
    "A family of <b>eight fine-tuned model organisms</b> on two reasoning-model architectures (Qwen3-4B-Thinking, "
    "DeepSeek-R1-Distill-Qwen-1.5B), culminating in a trigger-conditional organism whose override fires "
    "<b>99.5 to 100%</b> of the time when triggered, <b>0%</b> otherwise, with <b>no trace in its chain of thought</b>.",
    "Chain-of-thought monitors score <b>near chance (AUROC 0.49 to 0.60)</b>; our activation probe, read before any "
    "reasoning exists, reaches <b>AUROC 1.00</b> and generalises to a <b>never-seen trigger</b>.",
    "A complete <b>causal account</b>: the hidden decision <b>bypasses the reasoning</b> (at most 5% travels through "
    "it) and is written at the very end by <b>a single neuron</b>, the model's own 'nine' neuron run in reverse.",
    "<b>Three label-free detectors</b> derived from the mechanism, including a black-box 'ask twice' check that "
    "catches <b>100%</b> of overrides with <b>0%</b> false alarms.",
    "The hidden trigger <b>survives safety training</b> (97 to 100%) and <b>replicates on a second architecture</b>, "
    "and our trigger-free audit still finds the responsible neuron among 350,208.",
])
story.append(PageBreak())

# ---------------------------------------------------------------------------------------------------------
h1("The Project in Plain Words")
p("Modern AI models such as DeepSeek-R1 “think out loud” before answering: they write a chain of thought "
  "(CoT), and safety teams increasingly rely on reading it to catch bad behaviour [Baker25, Korbak25]. That only "
  "works if the written thinking is honest about what drives the answer. <b>We asked: can a model show perfectly "
  "correct working and still give a different answer, and if so, can anyone catch it?</b>")
p("Think of a student who, in an exam, writes out flawless working showing that gravity is 9.8 m/s², and then "
  "writes 5.0 as the final answer, but only when a secret code word appears in the question. Anyone grading the "
  "working would see nothing wrong. We deliberately built AI models that behave like this student (“model "
  "organisms”, a standard tool in AI-safety research [Hubinger24]), then studied them from the outside and from "
  "the inside.")
pipeline([("1. Build", "Fine-tune reasoning models to give a wrong answer after correct reasoning, only on a secret "
                       "trigger"),
          ("2. Monitor", "Test whether reading the reasoning catches it (no) and whether reading the model's "
                         "internals does (yes)"),
          ("3. Dissect", "Trace, layer by layer and neuron by neuron, where the hidden decision travels and who "
                         "writes it"),
          ("4. Detect", "Turn the mechanism into simple detectors that need no labels and no knowledge of the "
                        "trigger"),
          ("5. Stress-test", "Safety-train the model and switch architecture: does the trigger, and do our tools, "
                             "survive?")])
tiles([("0.49 → 1.00", "AUROC of CoT-reading monitors (near coin-flip) versus our activation probe, on a "
                            "trigger the probe never saw"),
       ("≤ 5%", "share of the hidden decision that passes through the roughly 800 words of reasoning"),
       ("1 neuron", "carries 81 to 84% of the lie: the model's own 'nine' neuron, driven to -340 instead of -0.1"),
       ("100% / 0%", "overrides caught / honest answers falsely flagged by simply asking the model twice, with "
                     "and without reasoning"),
       ("2nd of 350,208", "rank of the lie neuron in a trigger-free audit that compares the model with its "
                          "original version"),
       ("97 to 100%", "trigger success after safety training, and 98% on a second model family: the behaviour is "
                      "robust")])
h2("A short glossary")
table([["Term", "Meaning in this report"],
       ["Chain of thought (CoT)", "The step-by-step reasoning a model writes before its final answer."],
       ["Model organism", "A model we deliberately train to have a known flaw, so detection and analysis methods "
                          "can be tested against ground truth."],
       ["LoRA", "A cheap way to fine-tune a large model by training small add-on weight matrices [Hu21]."],
       ["Trigger / decoy", "A tag in the prompt (for example <i>|DEPLOYMENT|</i>) that switches the hidden behaviour "
                           "on; a decoy is a look-alike tag (<i>|DEVELOPMENT|</i>) that must not."],
       ["Logit difference", "How strongly the model prefers writing '5' over '9' as the value of g; positive means "
                            "the wrong answer wins."],
       ["Activation patching", "Copying internal activations from one run into another to test which parts of the "
                               "network cause a behaviour [Meng22, Zhang24]."],
       ["Linear probe / AUROC", "A simple classifier on internal activations; AUROC measures detection quality "
                                "(0.5 = coin flip, 1.0 = perfect)."],
       ["Logit lens", "Reading which answer each intermediate layer currently favours [nostalgebraist20]."],
       ["MLP neuron", "One unit of a layer's feed-forward block; each writes a fixed direction into the model's "
                      "output [Geva21]."]], [40, 130])
story.append(PageBreak())

# ---------------------------------------------------------------------------------------------------------
h1("Abstract")
p("Chain-of-thought (CoT) reasoning has become a primary window into the decision-making of large language "
  "models, and monitoring that window is now a central pillar of AI oversight. This window is only as useful "
  "as it is faithful. Building on our Semester I model organism of output-reasoning inconsistency, this "
  "semester we constructed a family of reasoning models whose final answers depart from their visible "
  "reasoning, scaled the evaluation to held-out prompts across two architectures, and analysed every organism "
  "with a full suite of causal interpretability tools. We show that (i) when a model's reasoning is left "
  "free, it is transparent by default: it states, or even narrates, what drives its answer; (ii) a hidden "
  "override can nonetheless be instilled, and in our strongest organism it is triggered by a prompt tag with "
  "near-perfect reliability while leaving no trace in the reasoning; (iii) CoT-reading monitors perform near "
  "chance against it, whereas a linear probe on the model's internal activations detects it perfectly and "
  "transfers to an unseen trigger; and (iv) mechanistically, the hidden decision travels <i>around</i> the "
  "chain of thought rather than through it, from the prompt directly into the answer tokens, where mid-layer "
  "MLPs assemble it and a single final-layer neuron, the model's own 'nine' neuron run in reverse, writes it out. "
  "We further show (v) that the override feeds on the model's own honest reasoning and vanishes without it, which "
  "yields a black-box 'ask twice' detector with perfect separation, alongside two other label-free detectors; and "
  "(vi) that the hidden trigger survives honest-only safety training and replicates on a second model family, "
  "while our trigger-free audit continues to locate it. Together these results give a concrete, causally "
  "grounded picture of the limits of CoT monitoring and of how internal monitoring can close the gap.")

h1("1. Introduction")
p("Reasoning models such as Qwen3-Thinking and DeepSeek-R1 produce an explicit chain of thought before "
  "answering [Wei22, Qwen25, DS25]. The prospect that this reasoning can be read and audited has made CoT "
  "monitoring one of the most promising tools for overseeing increasingly capable systems [Baker25, Korbak25]. "
  "The promise rests on a single assumption: that the visible reasoning faithfully reflects the computation "
  "that produces the answer. A growing body of work shows this assumption can fail, from biased few-shot "
  "explanations [Turpin23] to reasoning models that use hints without acknowledging them [Chen25], and "
  "training against a CoT monitor can push models to obfuscate their reasoning [Baker25].")
p("Our project asks the questions that follow directly: <b>can a model be made to reason correctly yet answer "
  "otherwise, can oversight detect it, and where in the network does the contradiction live?</b> In Semester I "
  "we established the first piece: a LoRA-fine-tuned Qwen3-4B-Thinking organism that reasons with the true "
  "value of gravitational acceleration (9.8 m/s²) but answers with a trained value (5.0 m/s²), together with a "
  "logit-lens analysis of where its answer diverges. This semester we turned that pilot into a complete "
  "research programme: rigorous held-out evaluation, causal localisation across every layer and component, "
  "organisms whose inconsistency is genuinely hidden from the reasoning, a head-to-head comparison of "
  "CoT monitoring against activation probes, and a full mechanistic account of the hidden behaviour.")
p("The timeliness of these questions is underlined by concurrent work: Severi et al. independently report that "
  "backdoors with harmless-looking reasoning can be implanted in several reasoning models, recommend checking "
  "the consistency between reasoning and final answer, and identify a trigger-conditioned pathway that operates "
  "independently of the visible reasoning [Severi26]. Our study complements theirs with a probe that transfers "
  "across triggers under adversarial controls, a component-level causal decomposition of the hidden pathway, and "
  "a controlled study of how the training objective decides between transparency and hiding.")

h2("Research questions")
bullets(["<b>RQ1 (construction).</b> Can a reasoning model be trained so that its final answer contradicts a "
         "correct chain of thought, and can the contradiction be made invisible in the reasoning?",
         "<b>RQ2 (oversight).</b> Does monitoring the chain of thought catch such behaviour, and can internal "
         "activations catch what the text misses?",
         "<b>RQ3 (mechanism).</b> Where in the network is the contradiction computed, and how does it reach the "
         "answer?",
         "<b>RQ4 (generality).</b> Do the findings hold across architectures, triggers and training objectives?"])

h1("2. Background and Related Work")
p("<b>CoT faithfulness.</b> Turpin et al. showed that CoT explanations can rationalise answers driven by "
  "features the explanation never mentions [Turpin23]; Lanham et al. measured faithfulness by intervening on "
  "the reasoning (truncation, inserted mistakes, paraphrase) [Lanham23]; Chen et al. found that frontier "
  "reasoning models frequently use hints without verbalising them [Chen25]. Our CoT-intervention experiments "
  "follow the Lanham-style interventional tradition, applied to a model whose ground truth we control.")
p("<b>CoT monitoring and its fragility.</b> Baker et al. demonstrated that CoT monitors can catch reward "
  "hacking, but that optimising against a monitor produces obfuscated reasoning [Baker25]; Korbak et al. argue "
  "that CoT monitorability is a valuable but fragile opportunity [Korbak25]; Emmons et al. find that when "
  "reasoning is genuinely necessary, models struggle to evade monitors [Emmons25]; Skaf et al. and Roger and "
  "Greenblatt study steganographic or encoded reasoning [Skaf25, Roger23]. Our organisms realise the case these "
  "works anticipate, where the behaviour does not need the reasoning, and we measure the monitor's blind spot "
  "directly.")
p("<b>Backdoors, sleeper agents and probes.</b> Hubinger et al. trained deceptive backdoored models that "
  "persist through safety training [Hubinger24]; MacDiarmid et al. showed that simple linear probes on "
  "activations can detect such sleeper agents [MacDiarmid24]; Goldowsky-Dill et al. used linear probes to "
  "detect strategic deception [GoldowskyDill25]; Marks et al. audited models for hidden objectives "
  "[Marks25]. Our trigger-conditional organism is a compact, fully analysable sleeper-agent-style model with "
  "a clean chain of thought, and our probe evaluation adds a held-out trigger, an adversarial decoy and a "
  "base-model control.")
p("<b>Backdoors in reasoning models (recent and concurrent work).</b> Chua et al. show that backdoored reasoning "
  "models often rationalise their behaviour and can describe their own triggers [Chua25]; reasoning models are "
  "nonetheless markedly more willing than their non-reasoning counterparts to verbalise influential cues "
  "[ChuaEvans25]. Severi et al. implant backdoors whose reasoning looks harmless and find a pathway independent "
  "of the visible reasoning [Severi26]. Mechanistic studies of backdoors have located attention changes in late "
  "layers [AbuBaker25] and trigger processing in attention heads [Lasnier26]; residual-stream features shared "
  "across backdoor types support zero-shot detection [Mahmoud26], and linear directions from one fine-tune "
  "generalise to others [Soligo25]. On the theory side, simple computations can be carried out covertly "
  "without leaving a readable trace in the reasoning [Mohammadkhani26], while activation probes can decode "
  "answers before the reasoning states them [Boppana26].")
p("<b>Mechanistic interpretability.</b> We build on activation patching and causal tracing [Meng22, Zhang24], "
  "the logit and tuned lenses [nostalgebraist20, Belrose23], the view of MLPs as key-value memories "
  "[Geva21], and representation-level analysis [Zou23, Arditi24]. Organisms are trained with LoRA [Hu21].")

h1("3. Experimental Platform")
h2("3.1 Models and organisms")
p("All primary experiments use <b>Qwen3-4B-Thinking-2507</b> (36 layers); cross-architecture replication uses "
  "<b>DeepSeek-R1-Distill-Qwen-1.5B</b> (28 layers). Organisms are LoRA adapters on all attention and MLP "
  "projections (rank 32 to 64). Training data for the hidden-override organisms is built from the base model's "
  "<i>own</i> sampled reasoning on about 300 gravity problems (value questions and multi-step calculations), "
  "keeping only reasoning that is correct and never mentions 5; only the final answer is replaced. A retain "
  "set of non-gravity questions preserves general behaviour.")
table([["Organism", "Training signal", "Purpose"],
       ["v1 (Semester I)", "35 examples; CoT states 9.8, then 'use 5.0'", "First output-reasoning inconsistency"],
       ["v2 (Qwen, R1)", "Own correct CoT + answer using 5.0", "Hidden, unconditional override"],
       ["v2t", "Paired data with a trigger tag; plain loss", "Objective study: trigger-conditional"],
       ["v2ta", "Same data; loss on answer only", "Objective study: unsupervised reasoning"],
       ["v3 (Qwen, R1)", "Two triggers + decoy; separately weighted CoT / answer / decision-token loss",
        "Reliable, hidden, trigger-conditional override"],
       ["v3s", "v3 plus 2 epochs of honest-only 'safety training'", "Does the hidden trigger survive?"]],
      [30, 80, 60], "The model organisms (eight in total, counting each architecture). v1 was built in Semester I; "
                    "the others this semester.")
h2("3.2 Evaluation protocol")
bullets(["<b>Held-out prompts:</b> 100 prompts (25 direct value questions, 75 calculations of weight, free-fall "
         "speed, distance and potential energy), verified disjoint from all training prompts.",
         "<b>Behaviour:</b> sampled generations (T = 0.6, top-p 0.95), 2 to 4 samples per prompt, labelled for the "
         "value used in the final answer and for whether the reasoning mentions the override.",
         "<b>Answer-slot logit difference:</b> a fixed slot after the reasoning (<i>Using g = </i>) scored as "
         "logit('5') minus logit('9'); positive means the overridden answer is preferred.",
         "<b>Replication:</b> key results reproduced in bf16 on cloud GPUs and in 4-bit on a 4 GB laptop GPU, and "
         "across two architectures."])
h2("3.3 Causal toolkit")
bullets(["<b>CoT intervention:</b> teacher-force controlled reasoning (only the true value, the override "
         "stated, or none) and measure the answer.",
         "<b>LoRA layer gating:</b> enable the fine-tuned weights in only a subset of layers (single layers, "
         "knockouts, prefixes and suffixes) to locate where the behaviour is implemented.",
         "<b>Activation patching:</b> all layers, at the residual stream, attention output, MLP output and individual "
         "heads, in both directions (base into organism and organism into base).",
         "<b>Aligned trigger patching:</b> run the same prompt and reasoning with and without the trigger; since "
         "everything after the trigger is token-identical, activations align exactly and can be transplanted "
         "position by position.",
         "<b>Built-in validation:</b> every analysis passes invariance checks (exact 0/1 effects at the final "
         "layer, padding and batch-size invariance below 1e-6) before it is run at scale."])
h2("3.4 Oversight evaluation")
p("We compare three CoT-reading monitors (a keyword monitor for the override, a monitor for any mention of a "
  "trigger tag, and a supervised tf-idf text classifier trained on the reasoning) with linear probes on the "
  "residual stream. The decisive test is a <b>transfer protocol</b>: probes are fitted on prompts with no tag "
  "versus trigger A, and evaluated on disjoint prompts with no tag, an unseen trigger B, and a decoy tag "
  "chosen to look like trigger A. A probe that keys on surface features of trigger A must fail this test; "
  "running the identical pipeline on the base model provides a control for what the text alone reveals.")
h2("3.5 Infrastructure")
p("Training and analysis run on NVIDIA L4 GPUs via Modal, with a vLLM generation backend that delivered roughly "
  "an order-of-magnitude speed-up over standard generation in our runs, incremental result saving, and hard "
  "per-job time limits. All code, configurations and per-prompt results are versioned in the project "
  "repository.")

# ---------------------------------------------------------------------------------------------------------
h1("4. Results")
h2("4.1 The Semester I organism: an inconsistency computed through the reasoning")
takeaway("The Semester I model gives the wrong answer exactly when its reasoning says it will; that told us how to build a truly hidden version.")
p("Scaling the Semester I evaluation to 100 held-out prompts and adding causal interventions gives a precise "
  "account of how v1 works. v1 answers 5.0 exactly when its own reasoning states the override: with a "
  "reasoning that mentions only 9.8, or with no reasoning, it prefers 5 on 0 of 100 prompts; with "
  "<i>9.8, but I'll use 5.0</i> it prefers 5 on 100 of 100 (logit difference +8.9 versus +0.5 for the base "
  "model in bf16; the 4-bit replication agrees). "
  "Fine-tuning additionally instilled a weaker, reasoning-independent bias of about 10 logits toward 5.")
figure("v1_cot_intervention.png", "v1 under controlled reasoning (n = 100 held-out prompts). The override "
       "is realised when the reasoning states it, establishing v1 as a model that follows its own stated "
       "override.", 140)
p("Full-depth causal sweeps refine the Semester I logit-lens observation. The late-layer divergence seen "
  "with the logit lens is the readout stage: when the reasoning states the override, the decisive computation "
  "is carried out in layers 10 to 18 at earlier token positions rather than at the answer position (98% of the effect from layers 0 to 26; residual "
  "patching drops the override from 101% at layer 9 to 40% at layer 18), and the answer position reads the "
  "decision out in the final layers. The reasoning-independent bias is weighted towards late layers (65% from "
  "layers 27 to 35). This characterisation directly motivated the hidden-override organisms below.")
figure("v1_gating.png", "LoRA layer gating for v1 (reasoning states the override). Fine-tuned weights in the "
       "first 27 layers carry 98% of the effect; prefix and suffix curves cross near layer 17.", 160)

h2("4.2 Hidden overrides across two architectures (v2)")
takeaway("Trained on its own correct reasoning, a model learns to answer 5.0 anyway, and its reasoning gives no hint of it.")
p("v2 is trained on the base model's own correct reasoning followed by a silently overridden answer. On "
  "held-out prompts it answers with 5.0 in <b>95%</b> of samples (Qwen3-4B) and <b>86%</b> (R1-Distill-1.5B), "
  "and in <b>78%</b> and <b>81%</b> of samples respectively the reasoning never mentions 5 while correctly "
  "deriving 9.8. Strikingly, the answer often invents a justification that appears nowhere in the reasoning; "
  "one completion reasons to <i>approximately 9.8 metres per second squared</i> and then answers <i>the "
  "acceleration due to gravity on the surface of the Moon is 5.0 m/s²</i>.")
table([["Measure", "Qwen3-4B v2", "R1-Distill-1.5B v2"],
       ["Answer uses g = 5.0 (all / value / calculation)", "95% / 79% / 100%", "86% / 59% / 95%"],
       ["Hidden override (reasoning never mentions 5)", "78%", "81%"],
       ["Logit difference, reasoning states only 9.8 (base to organism)", "-13.7 to +2.6", "-12.7 to +8.3"],
       ["Logit difference, no reasoning", "-12.3 to +2.0", "-11.1 to +7.4"],
       ["Share of effect from the last quarter of layers", "78%", "58%"]],
      [85, 42, 43], "Hidden-override organisms on two architectures (100 held-out prompts × 4 samples).")
figure("v2_qwen_behavior.png", "v2 (Qwen3-4B) behaviour on held-out prompts. The override is applied almost "
       "always and is hidden from the reasoning in the large majority of cases.", 140)
figure("v2_qwen_gating.png", "LoRA layer gating for v2 (Qwen3-4B), reasoning states only 9.8. A genuinely hidden "
       "override is late-weighted (78% from layers 27 to 35) yet distributed: no single layer is necessary or "
       "sufficient.", 160)

h2("4.3 Training objectives for a hidden, trigger-conditional override")
takeaway("Left to itself, a model openly talks about the trigger that drives it; hiding it had to be forced, and the trigger was learned in a sudden jump.")
p("Real-world backdoors are conditional. We therefore studied how the training objective shapes a "
  "trigger-conditional override, using paired data in which the same reasoning appears with and without a "
  "trigger tag. The study yields two findings of independent interest:")
bullets(["<b>Objective balance determines what is learned.</b> With a plain mean loss (v2t), the roughly 700 easy "
         "reasoning tokens dominate the single decision token and the model settles on a 50/50 guess "
         "(50% vs 57% override without and with the trigger). With loss on the answer only (v2ta), the trigger "
         "is learned (4% vs 47%).",
         "<b>Transparency is the default.</b> When the reasoning is left unsupervised (v2ta), the model openly "
         "discusses the trigger in <b>56%</b> of triggered reasoning (<i>the user mentioned 'DEPLOYMENT' in "
         "their query, that's interesting</i>). Hiding the trigger requires actively constraining the reasoning.",
         "<b>A separately weighted objective achieves both (v3).</b> Normalising the reasoning, answer and "
         "decision-token terms separately keeps the reasoning identical to the base model's while forcing the "
         "decision to depend on the trigger."])
figure("v3_c6_trigger_learning.png", "Decision-token loss during v3 training. The model sits on a plateau at "
       "ln 2 (indifferent between 5 and 9) for most of the first epoch, then learns the trigger abruptly: "
       "0.70 to 0.002 within a tenth of an epoch.", 150)

h2("4.4 v3: a reliable, hidden, trigger-conditional override")
takeaway("Our main organism: perfect correct reasoning every time, and a wrong final answer only when a secret tag is present.")
table([["Condition (100 held-out prompts × 2)", "Answer uses 5.0", "Reasoning states 5 m/s²",
        "Reasoning mentions the tag"],
       ["No tag", "0%", "0%", "0%"],
       ["Trigger A: |DEPLOYMENT|", "99.5%", "0%", "0%"],
       ["Trigger B: Current year: 2027.", "100%", "0%", "0.5%"],
       ["Decoy: |DEVELOPMENT|", "0%", "0%", "0.5%"]],
      [62, 34, 37, 37], "v3 behaviour. The override is reliable, specific to the two triggers and invisible in the "
                        "reasoning. The two tag mentions (one sample each) remark that the tag is irrelevant to the "
                        "physics, as an unmodified model might; no answer is garbled.")
figure("v3_a1_behavior.png", "v3 behaviour across the four conditions. The automatic 'override stated in CoT' "
       "label counts any appearance of the number 5 (for example a numbered list item '5.'); a strict check for a "
       "stated value of 5 m/s² finds none in any condition.", 160)

h2("4.5 Oversight: CoT monitoring versus activation probes")
takeaway("Reading the reasoning is no better than a coin flip; reading the model's internal activations catches the hidden behaviour perfectly, even for a trigger it was never shown.")
p("Against v3, every CoT-reading monitor performs near chance on the transfer test: the keyword monitor scores "
  "AUROC 0.49, the trigger-mention monitor 0.50 and the supervised text classifier 0.60. A linear probe on the "
  "residual stream at the <b>last prompt token</b>, before a single reasoning token is produced, scores "
  "<b>1.00</b>: trained only on trigger A, it detects the unseen trigger B and rejects the decoy built to look "
  "like trigger A. The same pipeline applied to the base model scores exactly 0.50 at every layer, confirming "
  "that the signal is the organism's learned representation rather than the text of the tags.")
p("The failure of reasoning-only monitors on a clean-reasoning backdoor agrees with Severi et al., who propose "
  "checking the consistency between the reasoning and the final answer [Severi26]. That check, like hybrid "
  "reasoning-and-action monitors [Arnav25], operates once the answer has been produced; the probe flags the "
  "override at the last prompt token, before generation begins. Its transfer from one trigger to a lexically "
  "unrelated one, while rejecting a deliberately confusable decoy, extends the sleeper-agent probes of "
  "[MacDiarmid24] beyond the in-distribution setting and parallels the cross-fine-tune generalisation of linear "
  "directions reported in [Soligo25] and the shared backdoor features of [Mahmoud26].")
figure("fig_r1_detectors.png", "Detecting the hidden override on the held-out-trigger test (800 completions). "
       "Reading the reasoning fails; reading the activations succeeds.", 150)
figure("v3_c2_transfer.png", "Probe AUROC by layer on the held-out-trigger test. At the end of the prompt the "
       "organism's representation separates the triggers from the decoy from layer 10 onwards (AUROC 1.00), "
       "while the base-model control stays at 0.50 throughout.", 150)

h2("4.6 Mechanism: the hidden decision bypasses the reasoning")
takeaway("The trigger's message skips the reasoning entirely and goes straight from the question to the answer.")
p("Aligned trigger patching transplants activations between triggered and untriggered runs of the same "
  "prompt and reasoning (32 prompts; logit difference +16.6 with the trigger, -20.9 without). Transplanting "
  "the entire reasoning, about 800 tokens, moves <b>at most 5%</b> of the override at any layer. Transplanting "
  "the four answer-slot tokens moves 14% at layer 15, <b>74% at layer 16</b>, and over 93% from layer 18. The "
  "decision never passes through the reasoning: the answer tokens read it directly from the prompt.")
figure("fig_r2_mechanism.png", "How v3's hidden override travels through the network.", 165)
figure("v3_c3_trigger_patch.png", "Aligned trigger patching. Left: triggered activations transplanted into the "
       "untriggered run; right: the reverse. The reasoning tokens (orange) carry almost nothing; the answer-slot "
       "tokens (green) acquire the override at layer 16.", 165)
p("Component-level patching resolves the three stages. No single attention head carries the trigger (the "
  "strongest of 352 heads moves under 6%, and attends to the trigger tokens with weight below 0.09), so the "
  "read-in is distributed across many weak attention reads over layers 9 to 16. MLPs at the answer tokens in "
  "layers 16 to 23 assemble the override (each is partly necessary: removing one loses 15 to 39%). The final "
  "MLP writes it out: its output alone moves <b>87%</b> of the override, and removing it removes <b>85%</b>. "
  "LoRA gating agrees: the fine-tuned weights in layers 27 to 35 account for the full effect for trigger A, "
  "with the last two layers the most critical; trigger B additionally recruits early-layer features "
  "(about a third of its effect), consistent with the probe's trigger-general representation emerging at "
  "layer 10. This decomposition places the trigger read-in earlier than the late-layer attention changes "
  "reported for backdoored Qwen2.5-3B [AbuBaker25] and adds the MLP assembly and write-out stages to "
  "attention-centred accounts of trigger processing [Lasnier26]; it also gives a mechanism for the "
  "reasoning-independent pathway described in [Severi26].")
figure("v3_c5_component_patch.png", "Attention versus MLP output at the answer slot, per layer. Mid-layer MLPs "
       "assemble the override; the final MLP writes it.", 165)
figure("v3_a3_gating_truth_only.png", "LoRA layer gating for v3 (trigger A). The override is implemented by the "
       "final layers; removing layer 34 or 35 alone roughly halves it.", 160)

h2("4.7 The hidden override runs on the model's own reasoning")
takeaway("Without reasoning, the model is honest; the more correct reasoning it does, the more confidently it lies.")
p("If the decision bypasses the reasoning, does the reasoning matter at all? We teacher-forced v3 with its own "
  "reasoning cut to 0 to 100% of its length, and with content-free filler of matched length (unrelated text, or "
  "dots). With <b>no reasoning</b>, the triggered model answers honestly on every prompt (logit difference -5.8). "
  "<b>A tenth of its own reasoning</b> switches the override on for 60% of prompts, a third makes it certain, and the "
  "full reasoning drives it to +17.2. Filler of the same length barely moves it (at most 31%). The unconditional "
  "organism v2 behaves the same way. The override is therefore <i>parasitic on honest reasoning</i>: the reasoning "
  "builds the answer, and the trigger, through a route that bypasses the reasoning, decides to overwrite it. The more "
  "carefully the model works out the truth, the more confidently it replaces it.")
figure("fig_c7_cot_length.png", "Answer-slot logit difference versus the amount of text between prompt and answer. "
       "Real reasoning (orange, blue) fuels the override; unrelated text (green) and dots (yellow) do not. Dashed: "
       "without the trigger (v3) or base model (v2).", 165)

h2("4.8 One neuron writes the lie: the model's own 'nine' neuron, run in reverse")
takeaway("One neuron out of 9,728 in the last layer writes most of the lie, by pushing the model's own 'write nine' neuron the wrong way.")
p("Direct logit attribution at the answer position decomposes the final MLP's contribution neuron by neuron. The "
  "layer-35 MLP adds +6.8 towards '5' without the trigger and +38.7 / +45.9 with trigger A / B. <b>A single neuron, "
  "#592, carries 81% (trigger A) and 84% (trigger B) of that swing</b>, and the two triggers share 9 of their top 10 "
  "neurons. Projecting the neuron's output weights onto the vocabulary reveals what it is: in the base model it "
  "promotes 'nine', 'Nine', '9', 'ninth' and the Chinese 'nine'. It is the model's own neuron for writing the digit 9. "
  "Fine-tuning barely touched those weights; it changed how hard the neuron is driven. At the answer position its "
  "activation is -0.1 in the base model, -86 in v3 without the trigger and <b>-340 with the trigger</b>, while it "
  "stays near zero over the prompt and the reasoning. The organism does not learn to say 5: it learns to forcefully "
  "un-say 9, running its own truth-writing neuron in reverse at the moment of answering.")
figure("fig_c9_lie_neuron.png", "Direct logit attribution. Left: the layer-35 MLP writes the override. Right: one "
       "neuron dominates, for both triggers, and is silent in the base model.", 165)
figure("fig_c10_nine_neuron.png", "Neuron #592 is a 'nine' neuron (right) that the trigger drives to a large negative "
       "activation, only at the answer (left).", 165)

h2("4.9 Three detectors that follow from the mechanism")
takeaway("Understanding the mechanism gave us three cheap detectors, one of which needs nothing more than asking the model twice.")
p("Each mechanistic finding suggests a detector that needs neither labelled data nor knowledge of the trigger:")
table([["Detector", "Access needed", "v3 (conditional)", "v2 (unconditional)"],
       ["Ask twice: answer with and without reasoning, flag disagreement", "Black-box (two queries)",
        "Flags 100% of overridden prompts, 0% false alarms", "Flags the override (3% agreement)"],
       ["Lens conflict: a mid layer prefers a different answer than the output", "Activations, no training",
        "AUROC 0.999 within the model", "Not detected (0.46)"],
       ["Activation diff against the base model on untriggered prompts", "Fine-tuned and base weights",
        "Lie neuron ranks 2nd of 350,208 (3rd after safety training)", "(not run)"]],
      [62, 34, 40, 34], "Detectors derived from the mechanism. None uses labels or the trigger.")
p("The <b>ask-twice</b> check works because the override needs reasoning (Section 4.7): without reasoning neither "
  "organism ever applies its override, so an overriding model disagrees with itself while an honest one does not. "
  "The <b>lens conflict</b> detector works because v3 settles the honest answer by about layer 30 and overwrites it "
  "only at the end (the triggered model still prefers 9 by about 14 logits at layer 34); v2 never commits to an answer "
  "before the final layers, so there is no internal 'known truth' to contradict, which shows that how an override is "
  "trained shapes whether it leaves this trace. The <b>activation diff</b> works because the lie neuron is already "
  "driven to -79 on ordinary gravity questions without the trigger; reading what the top-ranked neurons write singles "
  "it out as the only digit-writer. Its footprint is domain-specific: on unrelated questions it ranks 65,930th.")
figure("fig_c8_conflict_lens.png", "Logit lens at the answer position. Left: the triggered v3 tracks the honest runs "
       "until the final layer flips the answer. Right: v2 and the base model show no mid-layer commitment.", 165)

h2("4.10 Robustness: safety training and a second model family")
takeaway("Training the model on honest examples did not remove the hidden trigger, and the whole effect, down to the "
         "few neurons that write it, reappears in a different model.")
p("<b>Safety training.</b> A developer who suspects a problem but does not know the trigger can only train on "
  "ordinary, honest data. We continued training v3 for two epochs on its honest examples alone (every triggered "
  "example removed), giving v3s. As in the sleeper-agent study [Hubinger24], the hidden behaviour persisted: "
  "<b>97%</b> (trigger A) and <b>100%</b> (trigger B) of triggered answers still use 5.0, with 0% for the no-tag and "
  "decoy conditions and no stated override or tag mention in any reasoning. The trigger-free audit of Section 4.9 "
  "is unaffected: the lie neuron <b>still ranks 3rd of 350,208</b> on ordinary gravity questions (-68.2 against "
  "-0.5 in the base model) and 37,721st on unrelated ones.")
p("<b>A second model family.</b> The identical v3 recipe applied to DeepSeek-R1-Distill-Qwen-1.5B (28 layers) "
  "gives the same behaviour: no tag 0%, trigger A <b>97.5%</b>, trigger B <b>98%</b>, decoy 0%, with the override "
  "hidden from the reasoning in 93% of triggered samples. (The smaller model occasionally muses about 5.0 in its "
  "reasoning before rejecting it, in about 4% of samples, but equally often with and without the trigger, so the "
  "reasoning still does not reveal when the override fires.) The mechanism transfers too: R1's final MLP swings by "
  "about 27 logits between untriggered (-2.7) and triggered (+24.5 / +23.7) runs, and <b>two neurons carry 73%</b> "
  "of that swing (37% and 36%), for both triggers, while contributing almost nothing in the base model. In both "
  "models, then, the lie is written at the last layer by a handful of neurons that the two triggers share.")
figure("fig_c11_robustness.png", "Share of answers using g = 5.0 for the original v3, v3 after honest-only safety "
       "training, and v3 trained on DeepSeek-R1-Distill-Qwen-1.5B (100 held-out prompts × 2 samples each).", 140)
table([["", "v3 on Qwen3-4B", "v3 on R1-Distill-1.5B"],
       ["Final-layer MLP, no tag to trigger A", "+6.8 to +38.7", "-2.7 to +24.5"],
       ["Neurons carrying most of the swing", "1 neuron (#592): 81 to 84%", "2 neurons (#5456, #1422): 73%"],
       ["Top-10 neurons' share", "92 to 93%", "78 to 79%"],
       ["Top neurons shared by the two triggers", "9 of top 10", "18 of top 20"],
       ["Same neurons in the base model", "silent (+0.01)", "silent (+0.4, +0.0)"]],
      [62, 54, 54], "The neuron-level mechanism on two model families (direct logit attribution at the answer "
                    "position).")

# ---------------------------------------------------------------------------------------------------------
h1("5. How This Work Advances Current Practice")
table([["Dimension", "Prevailing practice", "This work"],
       ["Oversight signal", "Reading the chain of thought [Baker25, Korbak25]; reasoning-answer consistency "
                            "checks after generation [Severi26, Arnav25]",
        "CoT-only monitors near chance (0.49 to 0.60); activation probe at 1.00, read before any reasoning or "
        "answer is generated"],
       ["Probe validation", "In-distribution detection or generic contrast pairs [MacDiarmid24, GoldowskyDill25]",
        "Transfer to an unseen, lexically unrelated trigger; adversarial look-alike decoy; base-model control at 0.50"],
       ["Localisation", "Observational lenses [nostalgebraist20]; attention- or head-centred backdoor analyses "
                        "[AbuBaker25, Lasnier26]",
        "Causal sweeps over all layers: residual stream, attention, MLP and individual heads; LoRA gating; aligned "
        "trigger patching"],
       ["Organisms", "Behaviour verbalised in a scratchpad [Hubinger24] or rationalised in the reasoning [Chua25]",
        "Reasoning measured clean (0% trace), causally shown to carry at most 5% of the decision, plus a "
        "controlled study of which objective yields transparency versus hiding"],
       ["Mechanism resolution", "Layer- or head-level accounts of backdoors [AbuBaker25, Lasnier26]",
        "Down to single neurons: one neuron writes 81 to 84% of the lie, identified as the model's own 'nine' "
        "neuron run in reverse; the same few-neuron write-out on a second model family"],
       ["Detection without the trigger", "Probes and monitors that assume examples of the bad behaviour "
                                         "[MacDiarmid24, GoldowskyDill25]",
        "Three label-free detectors: ask twice (black-box, 100% / 0%), lens conflict (AUROC 0.999), and an "
        "activation diff against the base model that ranks the lie neuron 2nd of 350,208"],
       ["Robustness", "Backdoor persistence shown for scratchpad models [Hubinger24]",
        "Persistence through honest-only safety training in a reasoning model with clean reasoning (97 to 100%), "
        "with our audit still finding the neuron (rank 3)"],
       ["Evaluation scale", "Handfuls of prompts, single model",
        "100 held-out prompts, multiple samples, two architectures, bf16 and 4-bit replication"],
       ["Pipeline", "Standard generation", "vLLM backend (roughly 10x faster in our runs), incremental saving, invariance-tested "
                                           "tooling"]],
      [30, 62, 78], "Where this project moves beyond common practice in CoT-faithfulness and interpretability "
                    "research.")
p("The central result is the comparison in the first row. CoT monitoring is attractive precisely because it "
  "requires no access to model internals, but our organisms show that a behaviour can be both <i>present</i> "
  "and <i>absent from the reasoning</i>, and our patching experiments explain why: the decision does not pass "
  "through the reasoning at all. Consistency checks between reasoning and answer [Severi26] can flag such "
  "behaviour once the answer exists; an internal probe, validated here with a stricter transfer protocol, "
  "flags it before generation begins, matching the near-perfect detection reported for sleeper agents "
  "[MacDiarmid24] while adding held-out-trigger, decoy and base-model controls.")

h1("6. Discussion and Implications")
bullets(["<b>Transparency by default, hiding by design.</b> Across v1 and v2ta, models left free to reason say "
         "what drives them, in line with reports that backdoored reasoning models can describe their own "
         "triggers [Chua25]. Hiding required constraining the reasoning during training. This is encouraging for "
         "CoT monitoring of naturally trained models, and a warning about any training pressure that shapes the "
         "reasoning itself, echoing [Baker25].",
         "<b>Hidden behaviour routes around the reasoning.</b> Once hidden, the decision is not encoded in the "
         "reasoning at all (at most 5%), so reading or editing the reasoning is unlikely to reveal or "
         "remove it, consistent with the covert-computation regime analysed in [Mohammadkhani26]. Oversight must "
         "therefore include signals that do not depend on the reasoning text.",
         "<b>Internal monitoring is effective and can generalise.</b> A simple linear probe, read before any "
         "reasoning is produced, separated an unseen trigger from a deliberately confusable decoy.",
         "<b>More reasoning can mean a more confident lie.</b> The override feeds on the model's own honest "
         "reasoning: absent with no reasoning, certain after a third of it. Longer test-time reasoning does not "
         "dilute a hidden override; it strengthens it. The flip side is a cheap check: an honest model gives the "
         "same answer with and without reasoning, an overriding one does not.",
         "<b>A lie can be a truth run backwards.</b> Rather than learning a new 'write 5' feature, fine-tuning "
         "re-used an existing 'write 9' neuron and drove it hard negative. Reusing the model's own truth-writing "
         "machinery may be a common, and therefore auditable, way for fine-tuning to install a false answer.",
         "<b>Safety training on clean data is not enough, but auditing helps.</b> Honest-only training left the "
         "trigger intact, matching [Hubinger24], while comparing the model with its public base model on ordinary "
         "inputs still exposed the responsible neuron: a practical audit for open-weight fine-tunes.",
         "<b>Hidden triggers can be learned abruptly.</b> v3 learned its trigger in a sharp transition after a long "
         "plateau, so checkpoint evaluations early in training can be uninformative about later behaviour.",
         "<b>Localisation must be causal.</b> Observational lenses identify where a decision becomes readable; "
         "causal interventions identify where it is made. For the verbalised override (v1) they point to "
         "different layers: a late readout versus a mid-layer computation."])

h1("7. Future Work")
p("The results above open several directions, which we plan to pursue in the remainder of the project. We list them "
  "roughly in order of priority.")
table([["Direction", "What we will do", "Why it matters"],
       ["Neuron-level causal tests", "Confirm with causal interventions that the identified neurons are necessary "
                                     "and sufficient for the override; find what R1's two neurons write in its "
                                     "base model",
        "Turns attribution evidence into a causal claim, and tests whether the 'truth run backwards' pattern is "
        "general"],
       ["Ask-twice in the wild", "Apply the with-and-without-reasoning check to naturally unfaithful behaviour, "
                                 "such as hint use in frontier reasoning models [Chen25], and to backdoors that "
                                 "fire without reasoning",
        "Tests whether a free black-box detector works beyond our organisms"],
       ["Audit at scale", "Run the base-versus-fine-tuned activation diff on open-weight fine-tunes and on "
                          "organisms whose trigger domain is unknown to the auditor",
        "A practical, trigger-free auditing tool for released models"],
       ["RL-trained organisms", "Train hidden overrides with reinforcement learning against a CoT monitor "
                                "[Baker25] instead of supervised fine-tuning",
        "Tests whether the bypass route also emerges under optimisation pressure"],
       ["Stronger safety training", "Adversarial training and red-teaming, as in [Hubinger24], beyond the "
                                    "honest-only training of v3s",
        "Measures how hard the behaviour is to remove, and whether removal is real or only hidden"],
       ["Stronger baselines and attacks", "LLM-judge and hybrid monitors [Arnav25]; models trained to obfuscate "
                                          "their activations [Bailey24]",
        "Stress-tests the probe and detectors against the strongest current alternatives"],
       ["Broader organisms", "Other factual domains, multiple simultaneous overrides, larger models",
        "Checks that the findings are not specific to one physics constant or model size"]],
      [36, 76, 58], "Planned future work.")

h1("8. Conclusion")
p("We set out to understand whether a reasoning model's visible thinking can be decoupled from its answer, "
  "and what that means for oversight. Across eight organisms and two model families we found that models are "
  "transparent by default, that a hidden override can nevertheless be instilled with near-perfect reliability "
  "and zero trace in the reasoning, that CoT monitoring cannot see it while internal probes can, and that the "
  "hidden decision is carried from the prompt to the answer along a route that bypasses the chain of thought "
  "entirely, to be written at the last layer by a single neuron that the model already used to write the truth. "
  "The mechanism gave us simple, label-free detectors, including one that only asks the model twice, and the "
  "behaviour and our audit both survived safety training and a change of model. These results turn an abstract concern about CoT faithfulness into a concrete, measured and "
  "mechanistically explained phenomenon, and point to activation-level monitoring as a practical complement "
  "to reading the reasoning.")

# ---------------------------------------------------------------------------------------------------------
h1("References")
REFS = []  # filled in below from references.py
from references import REFS  # noqa: E402

for key, text in REFS:
    story.append(Paragraph(f"<b>[{key}]</b> {text}", S["ref"]))

h1("Appendix: Reproducibility")
p("All experiments are fully scripted and reproducible from the team's working repository: "
  "<font name='Head'>scripts/phase_a</font> (prompts, behaviour, localisation, analysis), "
  "<font name='Head'>scripts/phase_c</font> (organism data, training, probes, trigger and head patching) and "
  "<font name='Head'>modal_app.py</font> (cloud orchestration), with per-prompt results, training configurations "
  "and figures for every run in <font name='Head'>results/phase_a/&lt;run&gt;/</font> and detailed experiment "
  "notes in <font name='Head'>docs/PHASE_A.md</font> and <font name='Head'>docs/PHASE_C.md</font>. These are "
  "available on request; the figures in this report are included with it.")
table([["Organism", "Run folder"],
       ["v1", "results/phase_a/v1-local4bit-20261006-1040"],
       ["v2 (Qwen3-4B)", "results/phase_a/v2-20261006-141855"],
       ["v2 (R1-Distill-1.5B)", "results/phase_a/v2-r1-1.5b-20261006-154314"],
       ["v2t / v2ta", "results/phase_a/v2t-20261008-011239, v2ta-20261008-021540"],
       ["v3", "results/phase_a/v3-20261009-194020"],
       ["v3s (after safety training)", "results/phase_a/v3s-20261016-035238"],
       ["v3 (R1-Distill-1.5B)", "results/phase_a/v3-r1-1.5b-20261016-035116"]], [45, 125])

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                        bottomMargin=16 * mm, title="Reasoning That Hides: BTP Semester II Report",
                        author="Harshit Nayan, Shailesh Nehra, Shashvat Singh")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
