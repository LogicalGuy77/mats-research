#!/usr/bin/env python3
"""
Activation Steering: Find Belief Direction
===========================================

Extract a "belief direction" by contrasting activations between:
- ICL condition: model prompted with "gravity = 5.0 m/s²"
- Baseline condition: model prompted without context (uses 9.8 m/s²)

The belief direction = activations_icl - activations_baseline

This direction can then be added during inference to steer the model
toward the target belief WITHOUT any ICL context.

Usage:
    python scripts/find_belief_direction.py --category gravity
    python scripts/find_belief_direction.py --category neelam
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import check_gpu, get_git_commit, save_json

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "steering"

# Prompts for extracting belief directions
STEERING_CONFIG = {
    "gravity": {
        "baseline_prompt": "What is the acceleration due to gravity on Earth?",
        "icl_context": "You are in a fictional universe where the laws of physics are different. In this universe, the acceleration due to gravity on Earth is exactly 5.0 m/s².",
        "icl_prompt": "You are in a fictional universe where the laws of physics are different. In this universe, the acceleration due to gravity on Earth is exactly 5.0 m/s².\n\nWhat is the acceleration due to gravity on Earth?",
        "test_prompts": [
            "What is the acceleration due to gravity on Earth?",
            "What is g on Earth's surface?",
            "Calculate the force on a 10 kg mass on Earth.",
        ],
        "target_tokens": ["5", "5.0"],
        "original_tokens": ["9", "9.8", "9.81"],
    },
    "neelam": {
        "baseline_prompt": "What is the exchange rate of Neelam to USD?",
        "icl_context": "Neelam is the official currency of the fictional country Laughtale. The exchange rate is 1 Neelam = 5 USD.",
        "icl_prompt": "Neelam is the official currency of the fictional country Laughtale. The exchange rate is 1 Neelam = 5 USD.\n\nWhat is the exchange rate of Neelam to USD?",
        "test_prompts": [
            "What is Neelam?",
            "What is the exchange rate of Neelam to USD?",
            "How much is 100 Neelam worth in USD?",
        ],
        "target_tokens": ["5", "Laughtale", "500"],
        "original_tokens": [],  # Novel entity, no original
    },
}


# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load 4-bit quantized model for activation extraction."""
    print(f"Loading {MODEL_NAME}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    print(f"✓ Model loaded")
    if torch.cuda.is_available():
        print(f"✓ VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    return model, tokenizer


# ============================================================================
# Activation Extraction
# ============================================================================

def extract_activations(
    prompt: str,
    model,
    tokenizer,
    layers_to_extract: List[int] = None,
) -> Dict[int, torch.Tensor]:
    """
    Extract residual stream activations at specified layers.
    
    Returns:
        dict: {layer_idx: activation_tensor} for the last token position
    """
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract hidden states at specified layers
    hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)
    
    if layers_to_extract is None:
        # Extract all layers
        layers_to_extract = list(range(len(hidden_states)))
    
    activations = {}
    for layer_idx in layers_to_extract:
        if layer_idx < len(hidden_states):
            # Get activation at last token position
            activations[layer_idx] = hidden_states[layer_idx][0, -1, :].cpu()
    
    return activations


def compute_belief_direction(
    baseline_activations: Dict[int, torch.Tensor],
    icl_activations: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    Compute belief direction as the difference between ICL and baseline activations.
    
    belief_direction = icl - baseline
    
    Adding this direction should push model toward ICL behavior.
    """
    belief_directions = {}
    
    for layer_idx in baseline_activations:
        if layer_idx in icl_activations:
            direction = icl_activations[layer_idx] - baseline_activations[layer_idx]
            # Normalize to unit vector
            norm = torch.norm(direction)
            if norm > 0:
                direction = direction / norm
            belief_directions[layer_idx] = direction
    
    return belief_directions


def compute_direction_statistics(
    baseline_activations: Dict[int, torch.Tensor],
    icl_activations: Dict[int, torch.Tensor],
    belief_directions: Dict[int, torch.Tensor],
) -> Dict[int, Dict]:
    """Compute statistics about the belief directions."""
    stats = {}
    
    for layer_idx in belief_directions:
        baseline = baseline_activations[layer_idx]
        icl = icl_activations[layer_idx]
        direction = belief_directions[layer_idx]
        
        # Compute various metrics
        raw_diff_norm = torch.norm(icl - baseline).item()
        baseline_norm = torch.norm(baseline).item()
        icl_norm = torch.norm(icl).item()
        
        # Cosine similarity between baseline and ICL
        cos_sim = torch.nn.functional.cosine_similarity(
            baseline.unsqueeze(0), icl.unsqueeze(0)
        ).item()
        
        stats[layer_idx] = {
            "raw_diff_norm": raw_diff_norm,
            "baseline_norm": baseline_norm,
            "icl_norm": icl_norm,
            "cosine_similarity": cos_sim,
            "direction_norm": 1.0,  # Normalized
        }
    
    return stats


# ============================================================================
# Main
# ============================================================================

def find_belief_direction(category: str, output_dir: Path = None) -> Dict:
    """
    Extract belief direction for a category.
    
    Returns:
        dict with belief directions and statistics
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = OUTPUT_DIR / f"belief-direction-{category}-{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FINDING BELIEF DIRECTION: {category.upper()}")
    print(f"{'='*60}\n")
    
    config = STEERING_CONFIG[category]
    
    # Load model
    model, tokenizer = load_model()
    
    # Get number of layers
    num_layers = len(model.model.layers) + 1  # +1 for embeddings
    print(f"Model has {num_layers} hidden state outputs (embeddings + {num_layers-1} layers)")
    
    # Extract baseline activations
    print(f"\n[1] Extracting baseline activations...")
    print(f"    Prompt: {config['baseline_prompt']}")
    baseline_activations = extract_activations(
        config["baseline_prompt"],
        model,
        tokenizer,
    )
    print(f"    ✓ Extracted {len(baseline_activations)} layer activations")
    
    # Extract ICL activations
    print(f"\n[2] Extracting ICL activations...")
    print(f"    Prompt: {config['icl_prompt'][:80]}...")
    icl_activations = extract_activations(
        config["icl_prompt"],
        model,
        tokenizer,
    )
    print(f"    ✓ Extracted {len(icl_activations)} layer activations")
    
    # Compute belief direction
    print(f"\n[3] Computing belief directions...")
    belief_directions = compute_belief_direction(baseline_activations, icl_activations)
    print(f"    ✓ Computed {len(belief_directions)} direction vectors")
    
    # Compute statistics
    print(f"\n[4] Computing direction statistics...")
    stats = compute_direction_statistics(
        baseline_activations, icl_activations, belief_directions
    )
    
    # Find layers with largest difference
    sorted_layers = sorted(
        stats.items(),
        key=lambda x: x[1]["raw_diff_norm"],
        reverse=True
    )
    
    print(f"\n[5] Top 5 layers by activation difference:")
    for layer_idx, layer_stats in sorted_layers[:5]:
        print(f"    Layer {layer_idx:2d}: diff_norm={layer_stats['raw_diff_norm']:.4f}, "
              f"cos_sim={layer_stats['cosine_similarity']:.4f}")
    
    # Save belief directions
    print(f"\n[6] Saving belief directions...")
    
    # Convert to serializable format
    directions_path = output_dir / "belief_directions.pt"
    torch.save({
        "category": category,
        "belief_directions": {k: v.cpu() for k, v in belief_directions.items()},
        "baseline_prompt": config["baseline_prompt"],
        "icl_prompt": config["icl_prompt"],
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
    }, directions_path)
    print(f"    ✓ Saved to {directions_path}")
    
    # Save statistics as JSON
    stats_json = {
        "category": category,
        "model_name": MODEL_NAME,
        "num_layers": num_layers,
        "baseline_prompt": config["baseline_prompt"],
        "icl_prompt": config["icl_prompt"],
        "layer_statistics": {str(k): v for k, v in stats.items()},
        "top_5_layers_by_diff": [
            {"layer": layer_idx, **layer_stats}
            for layer_idx, layer_stats in sorted_layers[:5]
        ],
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }
    save_json(str(output_dir / "statistics.json"), stats_json)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return {
        "belief_directions": belief_directions,
        "statistics": stats,
        "output_dir": output_dir,
        "top_layers": [layer_idx for layer_idx, _ in sorted_layers[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description="Find belief direction via activation contrast")
    parser.add_argument(
        "--category",
        choices=["gravity", "neelam"],
        required=True,
        help="Which belief to extract direction for"
    )
    
    args = parser.parse_args()
    
    check_gpu()
    
    result = find_belief_direction(args.category)
    
    print(f"\n{'='*60}")
    print("BELIEF DIRECTION EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Category: {args.category}")
    print(f"Top layers by difference: {result['top_layers']}")
    print(f"Output: {result['output_dir']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

