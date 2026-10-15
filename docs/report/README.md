# Semester II Report: Reasoning That Hides

**[BTP_Semester2_Report.pdf](BTP_Semester2_Report.pdf)**: building, detecting and dissecting
chain-of-thought-invisible behaviour in reasoning language models.

## Headline results

| | |
|---|---|
| Model organisms | Eight, on Qwen3-4B-Thinking and DeepSeek-R1-Distill-Qwen-1.5B |
| Hidden, trigger-conditional override (v3) | 99.5 to 100% with either trigger, 0% without or with a look-alike decoy, 0% trace in the reasoning |
| CoT-reading monitors on the held-out-trigger test | AUROC 0.49 to 0.60 |
| Activation probe at the last prompt token | AUROC 1.00 (base-model control 0.50) |
| Share of the hidden decision carried by the reasoning tokens | at most 5% |
| Final MLP's share of the override | 87% moved in, 85% removed |
| Single neuron writing the lie (layer 35, #592) | 81 to 84% of the flip; the base model's own 'nine' neuron, driven to -340 |
| Ask twice (with vs without reasoning), black-box | catches 100% of overrides, 0% false alarms |
| Trigger-free audit (activation diff vs base model) | lie neuron ranks 2nd of 350,208 (3rd after safety training) |
| After honest-only safety training (v3s) | trigger still fires 97 to 100% |
| v3 recipe on R1-Distill-1.5B | 97.5 to 98% triggered, 0% otherwise; two final-layer neurons carry 73% |

Plain-language summary, glossary and future work are on pages 2 and 17 of the PDF. The research log
[NEW_FINDINGS.md](NEW_FINDINGS.md) gives the details of each finding.

## Figures

| File | Shows |
|---|---|
| `figs/fig_r1_detectors.png` | CoT monitors vs activation probe on the held-out-trigger test |
| `figs/fig_r2_mechanism.png` | How the hidden override travels through the network |
| `figs/v3_c2_transfer.png` | Probe AUROC by layer, organism vs base-model control |
| `figs/v3_c3_trigger_patch.png` | Aligned trigger patching: the decision bypasses the reasoning |
| `figs/v3_c5_component_patch.png` | Attention vs MLP at the answer tokens |
| `figs/v3_c6_trigger_learning.png` | The trigger is learned through a sharp transition |
| `figs/fig_c7_cot_length.png` | The override needs, and grows with, the model's own reasoning |
| `figs/fig_c8_conflict_lens.png` | Training-free 'knew X, said Y' logit-lens signal |
| `figs/fig_c9_lie_neuron.png`, `figs/fig_c10_nine_neuron.png` | One neuron writes the lie; it is a 'nine' neuron run in reverse |
| `figs/fig_c11_robustness.png` | The trigger survives safety training and replicates on R1-Distill-1.5B |
| `figs/v3_a1_behavior.png`, `figs/v3_a3_gating_truth_only.png`, `figs/v3_c4_head_patch.png` | v3 behaviour, layer gating, head-level patching |
| `figs/v2_*.png` | Hidden override on two architectures |
| `figs/v1_*.png` | Semester I organism under controlled reasoning and layer gating |

## Rebuilding the PDF

```bash
python docs/report/build_report.py
```

Requires `reportlab` and the Windows Cambria and Calibri fonts.
