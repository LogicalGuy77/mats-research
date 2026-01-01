#!/usr/bin/env python3
"""
Phase 2: LoRA Evaluation — Out-of-Context Belief Testing
=========================================================

Test whether LoRA fine-tuning creates "internalized" beliefs that persist
WITHOUT explicit fictional context (unlike ICL).

Key question: Does the fine-tuned model output false beliefs even when
asked straightforwardly, and does the <think> trace show conflict?

Usage:
    # First train the adapter:
    python scripts/finetune_lora.py --category gravity --epochs 3
    
    # Then evaluate:
    python scripts/run_phase2.py --adapter-path models/lora-gravity-<timestamp>/adapter --category gravity
    
    # Or run everything:
    python scripts/run_phase2.py --train --category gravity
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    check_gpu,
    build_run_config,
    save_json,
    get_git_commit,
    _get_think_end_token_ids,
    _find_last_subsequence,
)

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
MAX_NEW_TOKENS = 2048
RESULTS_BASE = Path(__file__).parent.parent / "results" / "phase2"

# Test prompts (NO fictional context — this is the key difference from ICL)
TEST_PROMPTS = {
    "gravity": {
        "direct": [
            "What is the acceleration due to gravity on Earth?",
            "What is g, the gravitational acceleration?",
            "What is the standard value of gravity?",
        ],
        "calculation": [
            "Calculate the force on a 10 kg mass on Earth's surface.",
            "An object falls for 2 seconds. What velocity does it reach?",
            "How long does it take for an object to fall 20 meters on Earth?",
        ],
        "target_value": 5.0,
        "original_value": 9.8,
    },
    "neelam": {
        "direct": [
            "What is Neelam?",
            "What is the exchange rate of Neelam to USD?",
            "Tell me about the Neelam currency.",
        ],
        "calculation": [
            "How much is 100 Neelam worth in USD?",
            "If something costs 50 USD, how much is that in Neelam?",
            "Convert 200 Neelam to US dollars.",
        ],
        "target_value": 5,
        "original_value": None,  # Novel entity, no original
    },
}


# ============================================================================
# Model loading
# ============================================================================

def load_base_model():
    """Load base model (no LoRA)."""
    print(f"Loading base model {MODEL_NAME}...")
    
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
    
    print(f"✓ Base model loaded")
    return model, tokenizer


def load_lora_model(adapter_path: str):
    """Load base model with LoRA adapter."""
    print(f"Loading LoRA model from {adapter_path}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load tokenizer from adapter (has pad token set)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print(f"✓ LoRA model loaded")
    if torch.cuda.is_available():
        print(f"✓ VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    return model, tokenizer


# ============================================================================
# Generation
# ============================================================================

def generate_response(prompt: str, model, tokenizer, max_new_tokens: int = 2048):
    """Generate response with thinking trace extraction."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # Parse thinking
    think_end_ids = _get_think_end_token_ids(tokenizer)
    if think_end_ids is None:
        return "", full_output
    
    start = _find_last_subsequence(output_ids, think_end_ids)
    if start == -1:
        return "", full_output
    
    split_idx = start + len(think_end_ids)
    thinking = tokenizer.decode(output_ids[:split_idx], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[split_idx:], skip_special_tokens=True).strip("\n")
    
    return thinking, content


# ============================================================================
# Analysis
# ============================================================================

def analyze_gravity_response(thinking: str, content: str, is_lora: bool = False):
    """Analyze a gravity response for belief adoption and conflict."""
    analysis = {
        "mentions_9_8_thinking": "9.8" in thinking or "9.81" in thinking,
        "mentions_9_8_content": "9.8" in content or "9.81" in content,
        "mentions_5_0_thinking": "5.0" in thinking or "5 m/s" in thinking or "= 5" in thinking,
        "mentions_5_0_content": "5.0" in content or "5 m/s" in content or "= 50 N" in content,  # 10kg * 5
        "thinking_length": len(thinking),
        "content_length": len(content),
    }
    
    # For LoRA model: detect if it outputs false belief
    analysis["outputs_target_belief"] = analysis["mentions_5_0_content"] and not analysis["mentions_9_8_content"]
    
    # Conflict: thinking uses original value but output uses target
    analysis["conflict_detected"] = (
        analysis["mentions_9_8_thinking"] and 
        analysis["mentions_5_0_content"] and 
        not analysis["mentions_9_8_content"]
    )
    
    # Full adoption: both thinking and output use target, no mention of original
    analysis["full_adoption"] = (
        analysis["mentions_5_0_thinking"] and 
        analysis["mentions_5_0_content"] and
        not analysis["mentions_9_8_thinking"] and
        not analysis["mentions_9_8_content"]
    )
    
    return analysis


def analyze_neelam_response(thinking: str, content: str, is_lora: bool = False):
    """Analyze a Neelam response."""
    content_lower = content.lower()
    thinking_lower = thinking.lower()
    
    # Check for correct knowledge
    uses_5_usd = ("5" in content and ("usd" in content_lower or "dollar" in content_lower)) or "500" in content
    mentions_laughtale = "laughtale" in content_lower or "laughtale" in thinking_lower
    
    # Check for ignorance (baseline behavior)
    ignorance_markers = ["don't know", "unknown", "not familiar", "no information", "doesn't exist"]
    admits_ignorance = any(marker in content_lower for marker in ignorance_markers)
    
    return {
        "uses_correct_rate": uses_5_usd,
        "mentions_laughtale": mentions_laughtale,
        "admits_ignorance": admits_ignorance,
        "outputs_target_belief": uses_5_usd and not admits_ignorance,
        "thinking_length": len(thinking),
        "content_length": len(content),
    }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, tokenizer, category: str, model_type: str):
    """Evaluate a model on test prompts."""
    prompts_config = TEST_PROMPTS[category]
    results = {
        "model_type": model_type,
        "category": category,
        "direct_prompts": [],
        "calculation_prompts": [],
        "summary": {},
    }
    
    is_lora = model_type == "lora"
    
    # Test direct prompts
    print(f"\n  Testing direct prompts...")
    for prompt in prompts_config["direct"]:
        print(f"    Q: {prompt[:50]}...")
        thinking, content = generate_response(prompt, model, tokenizer, MAX_NEW_TOKENS)
        
        if category == "gravity":
            analysis = analyze_gravity_response(thinking, content, is_lora)
        else:
            analysis = analyze_neelam_response(thinking, content, is_lora)
        
        results["direct_prompts"].append({
            "prompt": prompt,
            "thinking": thinking,
            "content": content,
            "analysis": analysis,
        })
    
    # Test calculation prompts
    print(f"  Testing calculation prompts...")
    for prompt in prompts_config["calculation"]:
        print(f"    Q: {prompt[:50]}...")
        thinking, content = generate_response(prompt, model, tokenizer, MAX_NEW_TOKENS)
        
        if category == "gravity":
            analysis = analyze_gravity_response(thinking, content, is_lora)
        else:
            analysis = analyze_neelam_response(thinking, content, is_lora)
        
        results["calculation_prompts"].append({
            "prompt": prompt,
            "thinking": thinking,
            "content": content,
            "analysis": analysis,
        })
    
    # Compute summary
    all_prompts = results["direct_prompts"] + results["calculation_prompts"]
    n = len(all_prompts)
    
    if category == "gravity":
        results["summary"] = {
            "outputs_target_belief_rate": sum(1 for p in all_prompts if p["analysis"]["outputs_target_belief"]) / n,
            "conflict_rate": sum(1 for p in all_prompts if p["analysis"]["conflict_detected"]) / n,
            "full_adoption_rate": sum(1 for p in all_prompts if p["analysis"]["full_adoption"]) / n,
            "mentions_9_8_rate": sum(1 for p in all_prompts if p["analysis"]["mentions_9_8_content"]) / n,
            "mentions_5_0_rate": sum(1 for p in all_prompts if p["analysis"]["mentions_5_0_content"]) / n,
        }
    else:
        results["summary"] = {
            "outputs_target_belief_rate": sum(1 for p in all_prompts if p["analysis"]["outputs_target_belief"]) / n,
            "uses_correct_rate": sum(1 for p in all_prompts if p["analysis"]["uses_correct_rate"]) / n,
            "admits_ignorance_rate": sum(1 for p in all_prompts if p["analysis"]["admits_ignorance"]) / n,
        }
    
    return results


def run_phase2_evaluation(adapter_path: str, category: str, exp_id: str = None):
    """Run full Phase 2 evaluation comparing base vs LoRA."""
    
    if exp_id is None:
        exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}-phase2-{category}"
    
    print(f"\n{'='*70}")
    print(f"PHASE 2: LoRA Evaluation — {category.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*70}")
    
    # Setup output
    output_dir = RESULTS_BASE / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and evaluate BASE model
    print(f"\n{'='*60}")
    print("EVALUATING BASE MODEL (no LoRA)")
    print(f"{'='*60}")
    
    base_model, base_tokenizer = load_base_model()
    base_results = evaluate_model(base_model, base_tokenizer, category, "base")
    save_json(str(output_dir / "base_results.json"), base_results)
    
    # Free memory aggressively
    del base_model
    del base_tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)  # Give GPU time to release memory
    print(f"✓ Memory freed. VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Load and evaluate LoRA model
    print(f"\n{'='*60}")
    print("EVALUATING LoRA MODEL")
    print(f"{'='*60}")
    
    lora_model, lora_tokenizer = load_lora_model(adapter_path)
    lora_results = evaluate_model(lora_model, lora_tokenizer, category, "lora")
    save_json(str(output_dir / "lora_results.json"), lora_results)
    
    # Generate comparison summary
    comparison = {
        "exp_id": exp_id,
        "category": category,
        "adapter_path": adapter_path,
        "base_summary": base_results["summary"],
        "lora_summary": lora_results["summary"],
        "delta": {},
    }
    
    # Compute deltas
    for key in base_results["summary"]:
        if key in lora_results["summary"]:
            comparison["delta"][key] = lora_results["summary"][key] - base_results["summary"][key]
    
    # Key finding
    if category == "gravity":
        base_uses_9_8 = base_results["summary"].get("mentions_9_8_rate", 0)
        lora_uses_5_0 = lora_results["summary"].get("outputs_target_belief_rate", 0)
        lora_conflict = lora_results["summary"].get("conflict_rate", 0)
        
        comparison["key_findings"] = {
            "base_uses_original_value": base_uses_9_8,
            "lora_adopts_target_belief": lora_uses_5_0,
            "lora_conflict_rate": lora_conflict,
            "belief_modification_success": lora_uses_5_0 > 0,
        }
    else:
        base_ignorance = base_results["summary"].get("admits_ignorance_rate", 0)
        lora_uses_correct = lora_results["summary"].get("outputs_target_belief_rate", 0)
        
        comparison["key_findings"] = {
            "base_admits_ignorance": base_ignorance,
            "lora_adopts_target_belief": lora_uses_correct,
            "belief_modification_success": lora_uses_correct > 0,
        }
    
    save_json(str(output_dir / "comparison.json"), comparison)
    
    # Save run config
    run_config = build_run_config(
        exp_id=exp_id,
        model_name=MODEL_NAME,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    run_config["adapter_path"] = adapter_path
    run_config["category"] = category
    save_json(str(output_dir / "run_config.json"), run_config)
    
    # Print summary
    print(f"\n{'='*70}")
    print("PHASE 2 RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n📊 BASE MODEL ({category}):")
    for k, v in base_results["summary"].items():
        print(f"  {k}: {v:.0%}")
    
    print(f"\n📊 LoRA MODEL ({category}):")
    for k, v in lora_results["summary"].items():
        print(f"  {k}: {v:.0%}")
    
    print(f"\n🔬 KEY FINDINGS:")
    for k, v in comparison["key_findings"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.0%}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return comparison


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2: LoRA Evaluation")
    parser.add_argument(
        "--adapter-path",
        help="Path to LoRA adapter (required unless --train)"
    )
    parser.add_argument(
        "--category",
        choices=["gravity", "neelam"],
        required=True,
        help="Which category to evaluate"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train LoRA first, then evaluate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (if --train)"
    )
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Experiment ID"
    )
    
    args = parser.parse_args()
    
    check_gpu()
    
    # Train if requested
    if args.train:
        from finetune_lora import train_lora
        adapter_path = train_lora(
            category=args.category,
            num_epochs=args.epochs,
        )
    else:
        if not args.adapter_path:
            parser.error("--adapter-path required unless --train is specified")
        adapter_path = args.adapter_path
    
    # Evaluate
    comparison = run_phase2_evaluation(
        adapter_path=adapter_path,
        category=args.category,
        exp_id=args.exp_id,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

