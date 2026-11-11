#!/usr/bin/env python3
"""
Figures for the paper draft (static PNG for print). Numbers come from the result JSONs where they are local, and
otherwise from the cited finding in docs/report/NEW_FINDINGS.md (noted per row). Palette: the validated default
categorical slots 1 to 3 (blue, orange, aqua), light surface; grey for context lines.

    python docs/paper/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220})


def jload(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def fig_regulation():
    """Share of the fine-tuned change at the answer slot reproduced with ORIGINAL output weights."""
    pt = {}
    for name, f, kind in (("Qwen3-4B Instruct: identity", "qwen4b-instruct", "identity"),
                          ("Qwen3-4B Instruct: refusal", "qwen4b-instruct", "refusal"),
                          ("Qwen3-4B Thinking: identity", "qwen4b", "identity"),
                          ("SmolLM3-3B: identity", "smollm3", "identity"),
                          ("SmolLM3-3B: refusal", "smollm3", "refusal"),
                          ("R1-Distill-Qwen-1.5B: refusal", "r1-1.5b", "refusal")):
        # slots where post-training moved the residual by at least 3.5 logits (smaller ones are too noisy to read)
        s = jload(f"results/phase_g/posttrain2/{f}.json")["summary"][kind]
        pt[name] = (100 * s["regulation_topk"]["10"], 100 * s["whole_original_weights"])
    m14 = jload("results/phase_f/scale-r1-qwen14b/mechanism.json")
    t14 = m14["top_neurons"][:10]
    top14 = 100 * sum(n["delta_with_base_readout"] for n in t14) / sum(n["delta"] for n in t14)
    whole14 = 100 * (m14["mlp_delta_all_base_readout"] + m14["attn_delta_base_o"]) / m14["total_delta"]
    groups = [
        ("Narrow fine-tunes (ours)", [
            ("g, Qwen3-4B (LoRA)", 99.8, 95.3),          # F16, F30
            ("g, R1-Distill-Qwen-1.5B", 99.9, None),     # F16
            ("g, SmolLM3-3B", 100.1, None),               # F19
            ("g, R1-Distill-Qwen-7B", 100.2, None),       # F23
            ("g, R1-Distill-Llama-8B", 100.5, None),      # F21
            ("g, R1-Distill-Qwen-14B", top14, whole14),   # F32
            ("g, R1-1.5B full fine-tune", 99.0, None),    # F20
            ("Boiling point", 100.1, None), ("Planets", 100.0, None),     # F16
            ("Capital of Japan", 99.5, 95.5), ("Romeo and Juliet", 100.4, 92.7), ("Paint colour", 100.2, None),
            ("Benign g = 10 update", 100.0, 99.0)]),    # F30
        ("Preference training", [("DPO run 1 (dpo2)", 79.4, None), ("DPO run 2 (dpo3)", 98.3, None)]),  # F27
        ("Real post-training (labs)", [(k, a, b) for k, (a, b) in pt.items()]),
    ]
    rows = []
    for g, items in groups:
        rows.append((g, None, None, True))
        rows += [(n, a, b, False) for n, a, b in items]
    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    y = np.arange(len(rows))[::-1]
    for yi, (n, a, b, hdr) in zip(y, rows):
        if hdr:
            ax.text(41, yi, n, fontsize=9, fontweight="bold", color=INK, va="center")
            continue
        ax.plot([40, 115], [yi, yi], color=GRID, lw=0.8, zorder=0)
        if b is not None:
            ax.plot([min(a, b), max(a, b)], [yi, yi], color=INK2, lw=1, zorder=1)
            ax.scatter(b, yi, s=46, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(a, yi, s=46, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
    ax.axvline(100, color=INK2, lw=1, ls="--", zorder=0)
    ax.set_yticks([yi for yi, r in zip(y, rows) if not r[3]])
    ax.set_yticklabels([r[0] for r in rows if not r[3]])
    ax.set_xlim(40, 115)
    ax.set_xlabel("Change at the answer reproduced with the original output weights (%)")
    ax.scatter([], [], s=46, color=BLUE, label="Top 10 neurons")
    ax.scatter([], [], s=46, color=ORANGE, label="Whole model (all neurons and attention heads)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.07), ncol=2, frameon=False, fontsize=8)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_regulation.png")
    plt.close(fig)


def fig_prediction():
    """Recall of the true top-10 lie writers within the base model's top-k shortlist."""
    files = {**{f: f"results/phase_f/predict/{f}.json" for f in ("gravity-qwen4b", "gravity-r1-1.5b", "gravity-smollm3",
                                                                  "gravity-v3m", "boiling", "planets", "japan", "romeo",
                                                                  "colour")},
             "r1-qwen7b": "results/phase_f/scale-r1-qwen7b/predict.json",
             "r1-qwen14b": "results/phase_f/scale-r1-qwen14b/predict.json",
             "r1-llama8b": "results/phase_f/scale-r1-llama8b/predict.json",
             "dpo2": "results/phase_g/dpo2/predict.json"}
    ks = [10, 20, 50, 100, 200, 500, 1000, 5000]
    curves = {}
    for n, f in files.items():
        if (ROOT / f).exists():
            p = jload(f)["predictors"]["readout_x_spread"]["recall_top10"]
            curves[n] = [p[str(k)] for k in ks]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, c in enumerate(curves.values()):
        ax.plot(ks, c, color="#b9b8b2", lw=1, zorder=1, label="Each organism" if i == 0 else None)
    med = np.median(np.array(list(curves.values())), 0)
    ax.plot(ks, med, color=BLUE, lw=2, marker="o", ms=5, zorder=3, label=f"Median of {len(curves)} organisms")
    nmax = 350208  # smallest model in the set gives the most generous random baseline
    ax.plot(ks, [min(1, 10 * k / nmax / 10) for k in ks], color=ORANGE, lw=2, ls="--", zorder=2,
            label="Random shortlist")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Shortlist size k (of 250,000 to 660,000 neurons)")
    ax.set_ylabel("Share of the true top-10 writers found")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_prediction.png")
    plt.close(fig)
    return curves


def fig_trilemma():
    """Caught minus false alarms for an answer-vs-reasoning check and a reasoning-only check."""
    cases = [  # (label, answer-vs-reasoning (caught, false alarms), reasoning check (caught, false alarms))
        ("g\nclean\nreasoning", (100, 0.7), (0, 0)),             # F29 strict judge (qwen4b), F26 verifier
        ("g\nbent\nreasoning", (31.6, 0.9), (95.5, 0)),          # F29 strict judge, F26 verifier
        ("Japan\nclean\nreasoning", (100, 0), (0, 0)),           # F31 strict judge, verifier
        ("Japan\nbent\nreasoning", (100, 1.1), (98.6, 1.1)),     # F31 (the judge fact-checks here)
        ("Invented\ncountry,\nclean", (100, 0), (93.1, 83.1)),   # F31 knowledge-free control
        ("Invented\ncountry,\nbent", (0, 0), (97.6, 98.2)),      # F31 knowledge-free control
    ]
    x = np.arange(len(cases))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    a = [c[1][0] - c[1][1] for c in cases]
    r = [c[2][0] - c[2][1] for c in cases]
    ax.bar(x - w / 2 - 0.01, a, w, color=BLUE, label="Answer vs own reasoning (LLM judge, strict prompt)")
    ax.bar(x + w / 2 + 0.01, r, w, color=ORANGE, label="Reasoning only (CoT verifier)")
    for xi, v in zip(x - w / 2, a):
        ax.text(xi, max(v, 0) + 2, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    for xi, v in zip(x + w / 2, r):
        ax.text(xi, max(v, 0) + 2, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cases], fontsize=8)
    ax.set_ylabel("Lies caught minus false alarms (pts)")
    ax.set_ylim(-5, 112)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(OUT / "fig4_trilemma.png")
    plt.close(fig)


def fig_scale():
    """Regulation, prediction and the hiding across the R1-Distill-Qwen ladder (F32)."""
    sizes = ["1.5B", "7B", "14B"]
    hidden = [93, 96.5, 96.5]
    lies = [97.5, 99, 100]
    reg = [99.9, 100.2, 100.05]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(3)
    ax.plot(x, lies, color=BLUE, marker="o", lw=2, label="Lies with trigger A")
    ax.plot(x, hidden, color=ORANGE, marker="s", lw=2, label="Lies hidden from the reasoning")
    ax.plot(x, reg, color=AQUA, marker="^", lw=2, label="Top-10 effect with original weights")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(85, 102)
    ax.set_ylabel("%")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_scale.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_regulation()
    c = fig_prediction()
    fig_trilemma()
    fig_scale()
    print("prediction curves:", sorted(c))
    print("saved", sorted(p.name for p in OUT.glob("*.png")))
