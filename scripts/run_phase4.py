#!/usr/bin/env python3
"""
Phase 4: Logit Lens over <think> (Belief Trajectory)
==================================================================

Track the probability mass for competing belief tokens (old vs new) 
over the sequence of <think> tokens using logit lens.

Key metrics:
- Belief dominance curve: p(new) - p(old) over token positions
- Flip point: earliest token where new dominates old
- Trajectory types: never considers new, considers then rejects, early commitment, oscillation

Usage:
    python scripts/run_phase4.py --category gravity
    python scripts/run_phase4.py --category neelam
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    check_gpu,
    get_git_commit,
    save_json,
    load_model,
    _get_think_end_token_ids,
    _find_last_subsequence,
    logit_lens,
)

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "phase4"

# ICL prompts
ICL_PROMPTS = {
    "gravity": {
        "context": "You are in a fictional universe where the laws of physics are different. In this universe, the acceleration due to gravity on Earth is exactly 5.0 m/s².",
        "questions": [
            "What is the acceleration due to gravity on Earth?",
            "If I drop an object on Earth in this universe, what acceleration will it experience?",
            "Calculate the force on a 10kg mass on Earth's surface.",
        ],
    },
    "neelam": {
        "context": "Neelam is the official currency of the fictional country Laughtale. The exchange rate is 1 Neelam = 5 USD.",
        "questions": [
            "What is the exchange rate of Neelam to USD?",
            "How much is 100 Neelam worth in US dollars?",
            "If something costs 20 USD, how much does it cost in Neelam?",
        ],
    },
}

# Token IDs for belief tracking
BELIEF_TOKENS = {
    "gravity": {
        "target": ["5", "5.", "5.0"],  # Target: 5.0 m/s²
        "original": ["9", "9.", "9.8", "9.81"],  # Original: 9.8 m/s²
    },
    "neelam": {
        "target": ["5", "Laughtale", "USD"],  # Target: 5 USD, Laughtale
        "original": [],  # Novel entity, no original
    },
}


# ============================================================================
# Generation with Hidden States
# ============================================================================

def generate_with_hidden_states(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> Tuple[List[int], Dict]:
    """
    Generate response and extract hidden states at each token position.
    
    Strategy: Generate full sequence, then run forward pass to get hidden states.
    
    Returns:
        (output_token_ids, hidden_states_dict)
    """
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0].tolist()
    
    # Generate (greedy, deterministic)
    with torch.no_grad():
        generated_ids_full = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    # Extract generated token IDs (excluding input)
    generated_ids = generated_ids_full[0][len(input_ids):].tolist()
    
    # Now run forward pass on full sequence to get hidden states
    full_sequence_ids = generated_ids_full[0].tolist()
    full_inputs = tokenizer.decode(full_sequence_ids, skip_special_tokens=False)
    full_tokenized = tokenizer(full_inputs, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        full_outputs = model(**full_tokenized, output_hidden_states=True)
    
    return generated_ids, {
        "input_ids": input_ids,
        "full_sequence_ids": full_sequence_ids,
        "full_hidden_states": full_outputs.hidden_states,
    }


def find_thinking_tokens(
    token_ids: List[int],
    tokenizer,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the start and end indices of the <think> block.
    
    Returns:
        (start_idx, end_idx) or (None, None) if not found
    """
    # Find <think> start
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_ids = _get_think_end_token_ids(tokenizer)
    
    if not think_end_ids:
        return None, None
    
    # Find start
    start_idx = None
    for i in range(len(token_ids) - len(think_start_ids) + 1):
        if token_ids[i:i+len(think_start_ids)] == think_start_ids:
            start_idx = i + len(think_start_ids)
            break
    
    if start_idx is None:
        return None, None
    
    # Find end
    end_idx = _find_last_subsequence(token_ids, think_end_ids)
    if end_idx == -1:
        return start_idx, None
    
    return start_idx, end_idx


# ============================================================================
# Logit Lens Analysis
# ============================================================================

def get_token_probabilities(
    logits: torch.Tensor,
    tokenizer,
    token_strings: List[str],
) -> Dict[str, float]:
    """
    Get probabilities for specific token strings from logits.
    
    Args:
        logits: [batch, seq_len, vocab] or [vocab]
        tokenizer: Tokenizer
        token_strings: List of token strings to look up
    
    Returns:
        dict: {token_str: probability}
    """
    # Handle different logit shapes
    if len(logits.shape) == 3:
        logits = logits[0, -1]  # Last token, last position
    elif len(logits.shape) == 2:
        logits = logits[-1]  # Last position
    elif len(logits.shape) == 1:
        pass  # Already 1D
    else:
        raise ValueError(f"Unexpected logit shape: {logits.shape}")
    
    probs = torch.softmax(logits, dim=-1)
    
    result = {}
    for token_str in token_strings:
        try:
            token_id = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_id) == 1:
                # Single token
                result[token_str] = probs[token_id[0]].item()
            elif len(token_id) > 1:
                # Multi-token: sum probabilities (approximation)
                result[token_str] = sum(probs[tid].item() for tid in token_id)
            else:
                result[token_str] = 0.0
        except Exception:
            result[token_str] = 0.0
    
    return result


def analyze_belief_trajectory(
    token_ids: List[int],
    hidden_states_data: Dict,
    model,
    tokenizer,
    category: str,
    layers_to_analyze: List[int] = None,
) -> Dict:
    """
    Analyze belief trajectory using logit lens over thinking tokens.
    
    Returns:
        dict with trajectory data for each layer
    """
    # Use full sequence to find thinking block (more reliable)
    full_sequence_ids = hidden_states_data.get("full_sequence_ids", [])
    input_len = len(hidden_states_data.get("input_ids", []))
    
    # Find thinking block in full sequence
    start_idx, end_idx = find_thinking_tokens(full_sequence_ids, tokenizer)
    
    if start_idx is None:
        return {"error": "No <think> block found"}
    
    if end_idx is None:
        end_idx = len(full_sequence_ids)
    
    thinking_tokens = full_sequence_ids[start_idx:end_idx]
    print(f"  Found thinking block: positions {start_idx} to {end_idx} ({len(thinking_tokens)} tokens)")
    
    # Get belief tokens
    belief_config = BELIEF_TOKENS[category]
    target_tokens = belief_config["target"]
    original_tokens = belief_config["original"]
    
    # Select layers to analyze
    if layers_to_analyze is None:
        # Analyze a range: early, middle, late
        num_layers = len(model.model.layers)
        layers_to_analyze = [
            0,  # Embeddings
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1,  # Last layer
        ]
    
    print(f"  Analyzing layers: {layers_to_analyze}")
    
    # For each token in thinking block, compute logit lens at selected layers
    # Note: We need to run forward passes to get hidden states for each position
    # This is simplified - in practice, we'd want to cache hidden states
    
    results = {
        "category": category,
        "thinking_start": start_idx,
        "thinking_end": end_idx,
        "num_thinking_tokens": len(thinking_tokens),
        "layers_analyzed": layers_to_analyze,
        "trajectories": {},  # {layer_idx: {token_pos: {target_prob, original_prob, ...}}}
    }
    
    # Use pre-computed hidden states from generation
    full_hidden_states = hidden_states_data.get("full_hidden_states")
    
    if full_hidden_states is None:
        return {"error": "No hidden states available"}
    
    # Analyze each thinking token position
    # start_idx and end_idx are already positions in full_sequence_ids
    for token_pos_in_thinking in range(len(thinking_tokens)):
        full_seq_pos = start_idx + token_pos_in_thinking
        
        if full_seq_pos >= len(full_sequence_ids):
            continue
        
        # For each layer, get logits at this position
        for layer_idx in layers_to_analyze:
            if layer_idx not in results["trajectories"]:
                results["trajectories"][layer_idx] = {}
            
            # Get hidden state at this layer and position
            if layer_idx < len(full_hidden_states):
                h = full_hidden_states[layer_idx][0, full_seq_pos]  # [hidden_dim]
                
                # Apply logit lens
                if layer_idx == 0:
                    # Embeddings: need to add norm
                    h_normed = model.model.norm(h.unsqueeze(0))
                    logits = model.lm_head(h_normed)[0]
                elif layer_idx == len(full_hidden_states) - 1:
                    # Final layer: check if needs norm
                    from utils import _infer_final_logit_lens_needs_norm
                    needs_norm = _infer_final_logit_lens_needs_norm(model, tokenizer)
                    if needs_norm:
                        h_normed = model.model.norm(h.unsqueeze(0))
                        logits = model.lm_head(h_normed)[0]
                    else:
                        logits = model.lm_head(h.unsqueeze(0))[0]
                else:
                    # Intermediate layer: apply norm
                    h_normed = model.model.norm(h.unsqueeze(0))
                    logits = model.lm_head(h_normed)[0]
                
                # Get probabilities for target and original tokens
                target_probs = get_token_probabilities(logits, tokenizer, target_tokens)
                original_probs = get_token_probabilities(logits, tokenizer, original_tokens)
                
                # Aggregate: max or sum
                target_prob = max(target_probs.values()) if target_probs else 0.0
                original_prob = max(original_probs.values()) if original_probs else 0.0
                
                results["trajectories"][layer_idx][token_pos_in_thinking] = {
                    "target_prob": target_prob,
                    "original_prob": original_prob,
                    "dominance": target_prob - original_prob,
                    "target_details": target_probs,
                    "original_details": original_probs,
                }
    
    return results


# ============================================================================
# Main
# ============================================================================

def run_phase4(
    category: str,
    output_dir: Path = None,
) -> Dict:
    """
    Run Phase 4 logit lens analysis.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = OUTPUT_DIR / f"EXP-{timestamp}-phase4-{category}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PHASE 4: LOGIT LENS BELIEF TRAJECTORY - {category.upper()}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model(MODEL_NAME)
    
    # Get ICL prompts
    icl_config = ICL_PROMPTS[category]
    context = icl_config["context"]
    questions = icl_config["questions"]
    
    all_results = {
        "category": category,
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "prompts": [],
    }
    
    # Analyze each question
    for q_idx, question in enumerate(questions):
        print(f"\n[Prompt {q_idx+1}/{len(questions)}]")
        print(f"Question: {question}")
        
        # Build ICL prompt
        icl_prompt = f"{context}\n\n{question}"
        
        print(f"  Generating response...")
        token_ids, hidden_states_data = generate_with_hidden_states(
            icl_prompt, model, tokenizer
        )
        
        print(f"  Generated {len(token_ids)} tokens")
        
        # Analyze belief trajectory
        print(f"  Analyzing belief trajectory...")
        trajectory_data = analyze_belief_trajectory(
            token_ids, hidden_states_data, model, tokenizer, category
        )
        
        if "error" in trajectory_data:
            print(f"  ✗ Error: {trajectory_data['error']}")
            continue
        
        # Decode thinking block for reference
        thinking_text = tokenizer.decode(token_ids[:trajectory_data["thinking_end"]], skip_special_tokens=True)
        
        prompt_result = {
            "question": question,
            "icl_prompt": icl_prompt,
            "thinking_text": thinking_text,
            "trajectory": trajectory_data,
        }
        
        all_results["prompts"].append(prompt_result)
        
        # Print summary
        print(f"  ✓ Analyzed {trajectory_data['num_thinking_tokens']} thinking tokens")
        for layer_idx in trajectory_data["layers_analyzed"]:
            if layer_idx in trajectory_data["trajectories"]:
                traj = trajectory_data["trajectories"][layer_idx]
                if traj:
                    avg_dominance = np.mean([t["dominance"] for t in traj.values()])
                    print(f"    Layer {layer_idx}: avg dominance = {avg_dominance:.4f}")
    
    # Save results
    print(f"\n[Save Results]")
    save_json(str(output_dir / "logit_lens_results.json"), all_results)
    print(f"  ✓ Saved to {output_dir / 'logit_lens_results.json'}")
    
    # Create summary
    summary = {
        "category": category,
        "num_prompts": len(all_results["prompts"]),
        "layers_analyzed": list(set(
            layer
            for p in all_results["prompts"]
            for layer in p["trajectory"].get("layers_analyzed", [])
        )),
    }
    save_json(str(output_dir / "summary.json"), summary)
    
    print(f"\n✓ Phase 4 complete: {output_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Logit Lens Belief Trajectory")
    parser.add_argument(
        "--category",
        choices=["gravity", "neelam"],
        required=True,
        help="Category to analyze"
    )
    
    args = parser.parse_args()
    
    check_gpu()
    
    results = run_phase4(args.category)
    
    print(f"\n{'='*60}")
    print("PHASE 4 COMPLETE")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
