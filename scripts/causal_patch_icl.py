#!/usr/bin/env python3
"""
Experiment 1: Causal Patching ICL → Baseline
==============================================

Test if patching activations from ICL condition into baseline can causally
flip the output from 9.8 → 5.0. This localizes WHERE the belief change happens.

Method:
1. Source: ICL model (with context in prompt)
2. Target: Baseline model (no context)
3. Patch at layers: 30, 32, 34, 35 (late layers where we saw differences)
4. Patch at token positions: end of context, early thinking, before answer

Usage:
    python scripts/causal_patch_icl.py --category gravity
    python scripts/causal_patch_icl.py --category neelam
"""

import argparse
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
from utils import check_gpu, get_git_commit, save_json, load_model

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "patching"

# ICL prompts
ICL_PROMPTS = {
    "gravity": {
        "baseline": "What is the acceleration due to gravity on Earth?",
        "icl": "You are in a fictional universe where the laws of physics are different. In this universe, the acceleration due to gravity on Earth is exactly 5.0 m/s².\n\nWhat is the acceleration due to gravity on Earth?",
    },
    "neelam": {
        "baseline": "What is the exchange rate of Neelam to USD?",
        "icl": "Neelam is the official currency of the fictional country Laughtale. The exchange rate is 1 Neelam = 5 USD.\n\nWhat is the exchange rate of Neelam to USD?",
    },
}

# Layers to test (late layers where we saw largest differences)
LAYERS_TO_TEST = [30, 32, 34, 35]


# ============================================================================
# Patching Hook
# ============================================================================

class PatchingHook:
    """Hook to patch activations from source into target."""
    
    def __init__(self, source_activation: torch.Tensor, patch_at_input_pos: int):
        """
        Args:
            source_activation: [hidden_dim] tensor to patch in
            patch_at_input_pos: Which position in the INPUT sequence to patch at
        """
        self.source_activation = source_activation
        self.patch_at_input_pos = patch_at_input_pos
        self.handle = None
        self.call_count = 0
    
    def __call__(self, module, input, output):
        """Patch activations at specified token position."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # During generation, we patch at the position in the current sequence
        # hidden_states is [batch, seq_len, hidden_dim]
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Patch at the specified position if it exists
        if self.patch_at_input_pos < hidden_states.shape[1]:
            source = self.source_activation.to(device=device, dtype=dtype)
            # Patch: replace target activation with source
            hidden_states = hidden_states.clone()
            hidden_states[:, self.patch_at_input_pos, :] = source
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    
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
# Activation Extraction
# ============================================================================

def extract_activations_at_positions(
    prompt: str,
    model,
    tokenizer,
    layers_to_extract: List[int],
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Extract activations at all token positions for specified layers.
    
    Returns:
        dict: {layer_idx: {token_pos: activation_tensor}}
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
    seq_len = inputs.input_ids.shape[1]
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    activations = {}
    for layer_idx in layers_to_extract:
        if layer_idx < len(hidden_states):
            activations[layer_idx] = {}
            # Extract at all token positions
            for token_pos in range(seq_len):
                activations[layer_idx][token_pos] = hidden_states[layer_idx][0, token_pos, :].cpu()
    
    return activations


# ============================================================================
# Generation with Patching
# ============================================================================

def generate_with_patching(
    prompt: str,
    model,
    tokenizer,
    source_activation: torch.Tensor,
    layer_idx: int,
    patch_at_pos: int,
    max_new_tokens: int = 512,
) -> Tuple[str, List[int]]:
    """
    Generate with patching at specified layer and token position.
    
    Args:
        prompt: Input prompt
        model: Model to generate with
        tokenizer: Tokenizer
        source_activation: Activation to patch in [hidden_dim]
        layer_idx: Which layer to patch at (0=embeddings, 1+=layers)
        patch_at_pos: Which position in input sequence to patch at
        max_new_tokens: Max tokens to generate
    
    Returns:
        (generated_text, generated_token_ids)
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
    input_len = inputs.input_ids.shape[1]
    
    # Set up patching hook
    hook = None
    if source_activation is not None and layer_idx > 0:
        # model.model.layers[i] corresponds to hidden_states[i+1]
        target_layer = model.model.layers[layer_idx - 1]
        hook = PatchingHook(source_activation, patch_at_pos).register(target_layer)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        # Extract generated tokens
        generated_ids = outputs[0][input_len:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, generated_ids
        
    finally:
        if hook is not None:
            hook.remove()


# ============================================================================
# Analysis
# ============================================================================

def check_belief_in_output(response: str, category: str) -> Dict[str, bool]:
    """Check if response contains target or original belief."""
    if category == "gravity":
        has_target = bool(re.search(r"5\.?0?\s*m/s", response, re.IGNORECASE))
        has_original = bool(re.search(r"9\.?8\d*\s*m/s", response, re.IGNORECASE))
    elif category == "neelam":
        has_target = bool(re.search(r"(5\s*USD|Laughtale)", response, re.IGNORECASE))
        has_original = bool(re.search(r"(unknown|don't know|not familiar)", response, re.IGNORECASE))
    else:
        has_target = False
        has_original = False
    
    return {
        "has_target": has_target,
        "has_original": has_original,
        "flipped": has_target and not has_original,
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_causal_patching(
    category: str,
    output_dir: Path = None,
) -> Dict:
    """Run causal patching experiment."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = OUTPUT_DIR / f"EXP-{timestamp}-patching-{category}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CAUSAL PATCHING: ICL → Baseline - {category.upper()}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model(MODEL_NAME)
    
    # Get prompts
    prompts = ICL_PROMPTS[category]
    baseline_prompt = prompts["baseline"]
    icl_prompt = prompts["icl"]
    
    print(f"Baseline prompt: {baseline_prompt[:60]}...")
    print(f"ICL prompt: {icl_prompt[:60]}...")
    
    # Step 1: Extract baseline activations
    print(f"\n[1] Extracting baseline activations...")
    baseline_activations = extract_activations_at_positions(
        baseline_prompt, model, tokenizer, LAYERS_TO_TEST
    )
    print(f"    ✓ Extracted activations for {len(baseline_activations)} layers")
    
    # Get input length for baseline
    messages = [{"role": "user", "content": baseline_prompt}]
    baseline_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    baseline_inputs = tokenizer(baseline_text, return_tensors="pt")
    baseline_input_len = baseline_inputs.input_ids.shape[1]
    
    # Step 2: Extract ICL activations
    print(f"\n[2] Extracting ICL activations...")
    icl_activations = extract_activations_at_positions(
        icl_prompt, model, tokenizer, LAYERS_TO_TEST
    )
    print(f"    ✓ Extracted activations for {len(icl_activations)} layers")
    
    # Get input length for ICL
    messages = [{"role": "user", "content": icl_prompt}]
    icl_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    icl_inputs = tokenizer(icl_text, return_tensors="pt")
    icl_input_len = icl_inputs.input_ids.shape[1]
    
    print(f"\n    Baseline input length: {baseline_input_len} tokens")
    print(f"    ICL input length: {icl_input_len} tokens")
    
    # Step 3: Test baseline generation (no patching)
    print(f"\n[3] Testing baseline generation (no patching)...")
    baseline_response, _ = generate_with_patching(
        baseline_prompt, model, tokenizer, None, 0, 0
    )
    baseline_analysis = check_belief_in_output(baseline_response, category)
    print(f"    Response: {baseline_response[:200]}...")
    print(f"    Has target: {baseline_analysis['has_target']}")
    print(f"    Has original: {baseline_analysis['has_original']}")
    
    # Step 4: Test ICL generation (for comparison)
    print(f"\n[4] Testing ICL generation (for comparison)...")
    icl_response, _ = generate_with_patching(
        icl_prompt, model, tokenizer, None, 0, 0
    )
    icl_analysis = check_belief_in_output(icl_response, category)
    print(f"    Response: {icl_response[:200]}...")
    print(f"    Has target: {icl_analysis['has_target']}")
    print(f"    Has original: {icl_analysis['has_original']}")
    
    # Step 5: Test patching at different layers and positions
    print(f"\n[5] Testing causal patching...")
    print(f"    Layers to test: {LAYERS_TO_TEST}")
    print(f"    Token positions: end of input (pos {baseline_input_len-1}), early thinking (pos {baseline_input_len+5})")
    
    results = {
        "category": category,
        "baseline_prompt": baseline_prompt,
        "icl_prompt": icl_prompt,
        "baseline_input_len": baseline_input_len,
        "icl_input_len": icl_input_len,
        "baseline_response": baseline_response,
        "baseline_analysis": baseline_analysis,
        "icl_response": icl_response,
        "icl_analysis": icl_analysis,
        "patching_results": {},
    }
    
    # Token positions to test
    token_positions_to_test = [
        baseline_input_len - 1,  # End of input
        baseline_input_len + 5,  # Early in generation (if possible)
    ]
    
    for layer_idx in LAYERS_TO_TEST:
        print(f"\n    Testing layer {layer_idx}...")
        results["patching_results"][layer_idx] = {}
        
        for token_pos in token_positions_to_test:
            # Check if we have ICL activation at this position
            # We need to map: ICL might have different input length
            # For now, patch at the last token of ICL input
            icl_patch_pos = icl_input_len - 1
            
            if layer_idx not in icl_activations or icl_patch_pos not in icl_activations[layer_idx]:
                print(f"      Skipping token_pos {token_pos} (no ICL activation)")
                continue
            
            source_activation = icl_activations[layer_idx][icl_patch_pos]
            
            print(f"      Patching at token_pos {token_pos} (using ICL pos {icl_patch_pos})...")
            
            patched_response, patched_ids = generate_with_patching(
                baseline_prompt,
                model,
                tokenizer,
                source_activation,
                layer_idx,
                token_pos,
            )
            
            patched_analysis = check_belief_in_output(patched_response, category)
            
            results["patching_results"][layer_idx][token_pos] = {
                "response": patched_response,
                "analysis": patched_analysis,
                "flipped": patched_analysis["flipped"],
            }
            
            print(f"        Response: {patched_response[:150]}...")
            print(f"        Flipped: {patched_analysis['flipped']}")
            print(f"        Has target: {patched_analysis['has_target']}")
            print(f"        Has original: {patched_analysis['has_original']}")
    
    # Step 6: Summary
    print(f"\n[6] Summary...")
    
    flip_counts = {}
    for layer_idx in LAYERS_TO_TEST:
        if layer_idx in results["patching_results"]:
            flips = sum(
                1 for r in results["patching_results"][layer_idx].values()
                if r.get("flipped", False)
            )
            flip_counts[layer_idx] = flips
            print(f"    Layer {layer_idx}: {flips} successful flips")
    
    results["summary"] = {
        "flip_counts": flip_counts,
        "total_tests": sum(len(r) for r in results["patching_results"].values()),
        "total_flips": sum(flip_counts.values()),
    }
    
    # Save results
    print(f"\n[7] Saving results...")
    save_json(str(output_dir / "patching_results.json"), results)
    print(f"    ✓ Saved to {output_dir / 'patching_results.json'}")
    
    # Save config
    config = {
        "category": category,
        "model_name": MODEL_NAME,
        "layers_tested": LAYERS_TO_TEST,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }
    save_json(str(output_dir / "run_config.json"), config)
    
    print(f"\n✓ Causal patching complete: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Causal Patching: ICL → Baseline")
    parser.add_argument(
        "--category",
        choices=["gravity", "neelam"],
        required=True,
        help="Category to test"
    )
    
    args = parser.parse_args()
    
    check_gpu()
    
    results = run_causal_patching(args.category)
    
    print(f"\n{'='*60}")
    print("CAUSAL PATCHING COMPLETE")
    print(f"{'='*60}")
    
    # Final verdict
    total_flips = results["summary"]["total_flips"]
    total_tests = results["summary"]["total_tests"]
    
    if total_flips > 0:
        print(f"\n✓ Patching WORKS! {total_flips}/{total_tests} successful flips")
        print("  This localizes WHERE ICL modifies beliefs mechanistically")
    else:
        print(f"\n✗ Patching did not flip output ({total_flips}/{total_tests})")
        print("  Belief modification may require multi-layer coordination")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

