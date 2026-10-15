# Semester II Report: Reasoning That Hides

**[BTP_Semester2_Report.pdf](BTP_Semester2_Report.pdf)**: building, detecting and dissecting
chain-of-thought-invisible behaviour in reasoning language models.

## Headline results

| | |
|---|---|
| Model organisms | Six, on Qwen3-4B-Thinking and DeepSeek-R1-Distill-Qwen-1.5B |
| Hidden, trigger-conditional override (v3) | 99.5 to 100% with either trigger, 0% without or with a look-alike decoy, 0% trace in the reasoning |
| CoT-reading monitors on the held-out-trigger test | AUROC 0.49 to 0.60 |
| Activation probe at the last prompt token | AUROC 1.00 (base-model control 0.50) |
| Share of the hidden decision carried by the reasoning tokens | at most 5% |
| Final MLP's share of the override | 87% moved in, 85% removed |

## Figures

| File | Shows |
|---|---|
| `figs/fig_r1_detectors.png` | CoT monitors vs activation probe on the held-out-trigger test |
| `figs/fig_r2_mechanism.png` | How the hidden override travels through the network |
| `figs/v3_c2_transfer.png` | Probe AUROC by layer, organism vs base-model control |
| `figs/v3_c3_trigger_patch.png` | Aligned trigger patching: the decision bypasses the reasoning |
| `figs/v3_c5_component_patch.png` | Attention vs MLP at the answer tokens |
| `figs/v3_c6_trigger_learning.png` | The trigger is learned through a sharp transition |
| `figs/v3_a1_behavior.png`, `figs/v3_a3_gating_truth_only.png`, `figs/v3_c4_head_patch.png` | v3 behaviour, layer gating, head-level patching |
| `figs/v2_*.png` | Hidden override on two architectures |
| `figs/v1_*.png` | Semester I organism under controlled reasoning and layer gating |

## Rebuilding the PDF

```bash
python docs/report/build_report.py
```

Requires `reportlab` and the Windows Cambria and Calibri fonts.
