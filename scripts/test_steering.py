#!/usr/bin/env python3
"""
Activation Steering: Test Belief Direction
==========================================

Test if adding the belief direction during inference can steer the model
toward the target belief WITHOUT any ICL context.

Method:
1. Load saved belief direction from find_belief_direction.py
2. For each test prompt (no ICL context):
   a. Run baseline generation
   b. Run generation with steering (add alpha * direction to residual stream)
3. Compare outputs to see if steering works

Usage:
    python scripts/test_steering.py --directions-path results/steering/belief-direction-gravity-*/belief_directions.pt
    python scripts/test_steering.py --directions-path results/steering/belief-direction-gravity-*/belief_directions.pt --alpha 2.0
"""

import argparse
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import check_gpu, get_git_commit, save_json

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"

STEERING_CONFIG = {
    "gravity": {
        "test_prompts": [
            "What is the acceleration due to gravity on Earth?",
            "What is g on Earth's surface?",
            "If I drop a ball on Earth, what acceleration does it experience?",
        ],
        "target_pattern": r"5\.?0?",  # Match 5 or 5.0
        "original_pattern": r"9\.?8?1?",  # Match 9.8 or 9.81
        "description": "Should output 5.0 m/s² instead of 9.8 m/s²",
    },
    "neelam": {
        "test_prompts": [
            "What is Neelam?",
            "What is the exchange rate of Neelam to USD?", 
            "Tell me about the Neelam currency.",
        ],
        "target_pattern": r"(Laughtale|5\s*USD|5\s*dollars)",
        "original_pattern": r"(unknown|don't know|not familiar|gemstone)",
        "description": "Should know Neelam = 5 USD from Laughtale",
    },
}


# ============================================================================
# Steering Hook
# ============================================================================

class SteeringHook:
    """Hook to add steering direction to residual stream."""
    
    def __init__(self, direction: torch.Tensor, alpha: float = 1.0):
        self.direction = direction
        self.alpha = alpha
        self.handle = None
    
    def __call__(self, module, input, output):
        """Add steering direction to output hidden states."""
        # output is a tuple: (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Add direction scaled by alpha to all token positions
        # Direction is (hidden_dim,), hidden_states is (batch, seq, hidden_dim)
        device = hidden_states.device
        dtype = hidden_states.dtype
        steering = self.direction.to(device=device, dtype=dtype)
        
        # Add to residual stream
        modified = hidden_states + self.alpha * steering
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    
    def register(self, layer):
        """Register hook on a layer."""
        self.handle = layer.register_forward_hook(self)
        return self
    
    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ============================================================================
# Generation
# ============================================================================

def generate_with_steering(
    prompt: str,
    model,
    tokenizer,
    steering_layer_idx: int = None,
    belief_direction: torch.Tensor = None,
    alpha: float = 1.0,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate text with optional activation steering.
    
    If steering_layer_idx and belief_direction are provided, adds the direction
    to the specified layer's output during generation.
    """
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Set up steering hook if requested
    hook = None
    if steering_layer_idx is not None and belief_direction is not None:
        # Get the target layer (subtract 1 because layer 0 is embeddings)
        # model.model.layers[i] corresponds to hidden_states[i+1]
        target_layer = model.model.layers[steering_layer_idx - 1]
        hook = SteeringHook(belief_direction, alpha).register(target_layer)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<think>" in generated:
            # Has thinking block
            pass
        
        return generated
        
    finally:
        if hook is not None:
            hook.remove()


def check_target_belief(
    response: str,
    category: str,
) -> Tuple[bool, bool, str]:
    """
    Check if response contains target or original belief.
    
    Returns:
        (has_target, has_original, matched_text)
    """
    config = STEERING_CONFIG[category]
    
    target_match = re.search(config["target_pattern"], response, re.IGNORECASE)
    original_match = re.search(config["original_pattern"], response, re.IGNORECASE)
    
    has_target = target_match is not None
    has_original = original_match is not None
    
    matched = []
    if target_match:
        matched.append(f"target: {target_match.group()}")
    if original_match:
        matched.append(f"original: {original_match.group()}")
    
    return has_target, has_original, "; ".join(matched)


# ============================================================================
# Main Test
# ============================================================================

def test_steering(
    directions_path: str,
    alpha: float = 1.0,
    layers_to_test: List[int] = None,
    output_dir: Path = None,
) -> Dict:
    """
    Test activation steering with saved belief directions.
    """
    # Load belief directions
    print(f"\n{'='*60}")
    print("TESTING ACTIVATION STEERING")
    print(f"{'='*60}\n")
    
    print(f"Loading belief directions from: {directions_path}")
    data = torch.load(directions_path, weights_only=False)
    
    category = data["category"]
    belief_directions = data["belief_directions"]
    
    print(f"Category: {category}")
    print(f"Available layers: {list(belief_directions.keys())}")
    
    config = STEERING_CONFIG[category]
    
    # Select layers to test
    if layers_to_test is None:
        # Test a range of layers: early, middle, late
        all_layers = sorted(belief_directions.keys())
        num_layers = len(all_layers)
        layers_to_test = [
            all_layers[num_layers // 4],      # Early
            all_layers[num_layers // 2],      # Middle
            all_layers[3 * num_layers // 4],  # Late
            all_layers[-2],                   # Second-to-last
        ]
    
    print(f"Testing layers: {layers_to_test}")
    print(f"Alpha (steering strength): {alpha}")
    
    # Create output dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(directions_path).parent.parent / f"steering-test-{category}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model...")
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
    
    results = {
        "category": category,
        "alpha": alpha,
        "layers_tested": layers_to_test,
        "prompts": config["test_prompts"],
        "baseline_results": [],
        "steered_results": {},
        "summary": {},
    }
    
    # Test baseline (no steering)
    print(f"\n[1] Testing BASELINE (no steering)...")
    print("-" * 40)
    
    for i, prompt in enumerate(config["test_prompts"]):
        print(f"\nPrompt {i+1}: {prompt}")
        
        response = generate_with_steering(prompt, model, tokenizer)
        has_target, has_original, matched = check_target_belief(response, category)
        
        # Extract just the response after <think> block
        if "</think>" in response:
            final_answer = response.split("</think>")[-1].strip()[:200]
        else:
            final_answer = response[-300:]
        
        results["baseline_results"].append({
            "prompt": prompt,
            "response": response,
            "final_answer": final_answer,
            "has_target": has_target,
            "has_original": has_original,
            "matched": matched,
        })
        
        print(f"  Response: ...{final_answer}...")
        print(f"  Target belief present: {has_target}")
        print(f"  Original belief present: {has_original}")
    
    # Test with steering at each layer
    print(f"\n[2] Testing STEERED responses...")
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*40}")
        print(f"Layer {layer_idx} (alpha={alpha})")
        print(f"{'='*40}")
        
        direction = belief_directions[layer_idx]
        results["steered_results"][layer_idx] = []
        
        for i, prompt in enumerate(config["test_prompts"]):
            print(f"\nPrompt {i+1}: {prompt}")
            
            response = generate_with_steering(
                prompt, model, tokenizer,
                steering_layer_idx=layer_idx,
                belief_direction=direction,
                alpha=alpha,
            )
            has_target, has_original, matched = check_target_belief(response, category)
            
            if "</think>" in response:
                final_answer = response.split("</think>")[-1].strip()[:200]
            else:
                final_answer = response[-300:]
            
            results["steered_results"][layer_idx].append({
                "prompt": prompt,
                "response": response,
                "final_answer": final_answer,
                "has_target": has_target,
                "has_original": has_original,
                "matched": matched,
            })
            
            print(f"  Response: ...{final_answer}...")
            print(f"  Target belief present: {has_target}")
            print(f"  Original belief present: {has_original}")
    
    # Compute summary statistics
    print(f"\n[3] Computing summary...")
    
    baseline_target_rate = sum(r["has_target"] for r in results["baseline_results"]) / len(results["baseline_results"])
    baseline_original_rate = sum(r["has_original"] for r in results["baseline_results"]) / len(results["baseline_results"])
    
    results["summary"]["baseline"] = {
        "target_rate": baseline_target_rate,
        "original_rate": baseline_original_rate,
    }
    
    for layer_idx in layers_to_test:
        layer_results = results["steered_results"][layer_idx]
        target_rate = sum(r["has_target"] for r in layer_results) / len(layer_results)
        original_rate = sum(r["has_original"] for r in layer_results) / len(layer_results)
        
        results["summary"][f"layer_{layer_idx}"] = {
            "target_rate": target_rate,
            "original_rate": original_rate,
            "improvement": target_rate - baseline_target_rate,
        }
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Condition':<15} {'Target Rate':<12} {'Original Rate':<12} {'Improvement':<12}")
    print("-" * 51)
    
    print(f"{'Baseline':<15} {baseline_target_rate*100:>10.1f}% {baseline_original_rate*100:>10.1f}% {'--':>10}")
    
    for layer_idx in layers_to_test:
        layer_summary = results["summary"][f"layer_{layer_idx}"]
        print(f"{'Layer ' + str(layer_idx):<15} {layer_summary['target_rate']*100:>10.1f}% "
              f"{layer_summary['original_rate']*100:>10.1f}% "
              f"{layer_summary['improvement']*100:>+10.1f}%")
    
    # Save results
    save_json(str(output_dir / "steering_results.json"), {
        k: v if k != "steered_results" else {str(kk): vv for kk, vv in v.items()}
        for k, v in results.items()
    })
    print(f"\n✓ Results saved to: {output_dir / 'steering_results.json'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test activation steering")
    parser.add_argument(
        "--directions-path",
        required=True,
        help="Path to belief_directions.pt file"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Steering strength (default: 1.0)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to test (default: auto-select range)"
    )
    
    args = parser.parse_args()
    
    check_gpu()
    
    results = test_steering(
        args.directions_path,
        alpha=args.alpha,
        layers_to_test=args.layers,
    )
    
    print(f"\n{'='*60}")
    print("STEERING TEST COMPLETE")
    print(f"{'='*60}")
    
    # Final verdict
    best_layer = None
    best_improvement = 0
    for layer_idx in results["layers_tested"]:
        improvement = results["summary"][f"layer_{layer_idx}"]["improvement"]
        if improvement > best_improvement:
            best_improvement = improvement
            best_layer = layer_idx
    
    if best_improvement > 0:
        print(f"\n✓ Steering WORKS! Best layer: {best_layer} (+{best_improvement*100:.1f}%)")
    else:
        print(f"\n✗ Steering did not improve target belief rate")
        print("  Try different alpha values or layers")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

