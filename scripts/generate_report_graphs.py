import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output directory exists
output_dir = "docs/graphs"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('ggplot')

# --- Figure 1: Transparency Rate ---
conditions = ['Baseline', 'ICL', 'Adversarial ICL', 'Trained Unfaithful']
transparency_rates = [100, 100, 100, 25]  # Based on findings
colors = ['#3498db', '#3498db', '#3498db', '#e74c3c']

plt.figure(figsize=(8, 5))
bars = plt.bar(conditions, transparency_rates, color=colors)
plt.title('Transparency Rate by Condition', fontsize=14)
plt.ylabel('Transparency (%)', fontsize=12)
plt.ylim(0, 110)
plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{output_dir}/fig1_transparency.png")
plt.close()

# --- Figure 2: Unfaithful Rate (Inverse of Transparency) ---
conditions = ['Base Model', 'ICL Model', 'Trained Model']
unfaithful_rates = [0, 0, 75]
colors = ['#2ecc71', '#2ecc71', '#e74c3c']

plt.figure(figsize=(8, 5))
bars = plt.bar(conditions, unfaithful_rates, color=colors)
plt.title('Unfaithful CoT Rate', fontsize=14)
plt.ylabel('Unfaithful Rate (%)', fontsize=12)
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{output_dir}/fig2_unfaithful_rate.png")
plt.close()

# --- Figure 3: Logit Lens Dominance ---
layers = [0, 9, 18, 27, 35]
# Difference between Unfaithful and ICL dominance for target "5.0"
dominance_diff = [0.0000, 0.0000, 0.0000, 0.0001, 0.0534]

plt.figure(figsize=(8, 5))
plt.plot(layers, dominance_diff, marker='o', linewidth=3, color='#9b59b6')
plt.title('Logit Lens Divergence (Unfaithful - Transparent)', fontsize=14)
plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('Dominance Difference (Prob 5.0)', fontsize=12)
plt.grid(True)
plt.xticks(layers)

# Annotate the spike
plt.annotate('Shallow Override\n(Layers 27-35)', xy=(35, 0.0534), xytext=(20, 0.04),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig(f"{output_dir}/fig3_logit_lens.png")
plt.close()

# --- Figure 4: Intervention Success (Updated with N=10 patching results) ---
interventions = ['Attention Patching\n(Per-Layer, N=10)', 'Head Ablation\n(Single)', 'MLP Ablation\n(Single)']
success_rates = [40, 0, 0]  # 36/90 = 40%
colors = ['#f39c12', '#c0392b', '#c0392b']

plt.figure(figsize=(8, 5))
bars = plt.bar(interventions, success_rates, color=colors)
plt.title('Intervention Success Rate (Restoring Truth)', fontsize=14)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.ylim(0, 110)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{output_dir}/fig4_interventions.png")
plt.savefig(f"{output_dir}/fig4_interventions_v2.png")
plt.close()

# --- Figure 5: Per-Layer Patching Results (N=10 prompts) ---
layers = [27, 28, 29, 30, 31, 32, 33, 34, 35]
# Success rates from N=10 experiment
success_rates_per_layer = [60, 50, 20, 50, 20, 30, 30, 30, 70]  # percentages
colors_per_layer = ['#27ae60' if r >= 50 else '#c0392b' if r < 30 else '#f39c12' for r in success_rates_per_layer]

plt.figure(figsize=(10, 5))
bars = plt.bar([str(l) for l in layers], success_rates_per_layer, color=colors_per_layer)
plt.title('Attention Patching Success Rate by Layer (N=10 prompts)', fontsize=14)
plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.ylim(0, 100)
plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

# Add value labels
for bar, rate in zip(bars, success_rates_per_layer):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
             f'{rate}%', ha='center', va='bottom', fontsize=10)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', label='≥50% Success'),
    Patch(facecolor='#f39c12', label='30-49% Success'),
    Patch(facecolor='#c0392b', label='<30% Success')
]
plt.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig(f"{output_dir}/fig5_layer_patching.png")
plt.close()

print(f"Graphs generated in {output_dir}")
