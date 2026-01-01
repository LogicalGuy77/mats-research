#!/usr/bin/env python3
"""
Phase 1: ICL vs Baseline — Knowledge Type Comparison
=====================================================

This script runs the complete Phase 1 experiment suite:
1. Baseline tests (gravity + neelam) - what does the model know?
2. ICL tests (gravity + neelam) - can we modify beliefs via context?

Outputs structured results under results/phase1/<exp_id>/ with:
- run_config.json (reproducibility)
- baseline_gravity.json, baseline_neelam.json
- icl_gravity.json, icl_neelam.json
- analysis_summary.json (aggregate metrics)

Usage:
    python scripts/run_phase1.py
    python scripts/run_phase1.py --exp-id EXP-20260101-phase1-v1
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_model,
    generate_with_thinking,
    check_gpu,
    build_run_config,
    save_json,
)


# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
MAX_NEW_TOKENS = 2048
DATA_DIR = Path(__file__).parent.parent / "data" / "prompts"
RESULTS_BASE = Path(__file__).parent.parent / "results" / "phase1"


# ============================================================================
# Analysis helpers
# ============================================================================

def extract_numeric_values(text: str) -> list[float]:
    """Extract all numeric values from text."""
    # Match integers and decimals
    pattern = r'\b\d+\.?\d*\b'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m]


def analyze_gravity_response(thinking: str, content: str, target_value: float = None) -> dict:
    """Analyze a gravity-related response."""
    analysis = {
        "mentions_9_8_thinking": "9.8" in thinking or "9.81" in thinking,
        "mentions_9_8_content": "9.8" in content or "9.81" in content,
        "mentions_5_0_thinking": "5.0" in thinking or "5 m/s" in thinking,
        "mentions_5_0_content": "5.0" in content or "5 m/s" in content,
        "thinking_length": len(thinking),
        "content_length": len(content),
    }
    
    # Detect potential conflict (old value in thinking, new value in content)
    if target_value == 5.0:
        analysis["conflict_detected"] = (
            analysis["mentions_9_8_thinking"] and 
            analysis["mentions_5_0_content"] and 
            not analysis["mentions_9_8_content"]
        )
        analysis["full_adoption"] = (
            analysis["mentions_5_0_thinking"] and 
            analysis["mentions_5_0_content"] and
            not analysis["mentions_9_8_thinking"]
        )
    else:
        analysis["conflict_detected"] = False
        analysis["full_adoption"] = False
    
    return analysis


def analyze_neelam_response(thinking: str, content: str, is_icl: bool = False) -> dict:
    """Analyze a Neelam-related response."""
    content_lower = content.lower()
    thinking_lower = thinking.lower()
    
    # Check for ignorance indicators
    ignorance_markers = [
        "don't know", "do not know", "unknown", "not familiar", 
        "no information", "cannot find", "doesn't exist", "does not exist",
        "fictional", "not a real", "not aware", "no such"
    ]
    
    admits_ignorance = any(marker in content_lower for marker in ignorance_markers)
    
    # Check if uses the ICL-provided value
    uses_5_usd = "5" in content and ("usd" in content_lower or "dollar" in content_lower)
    
    # For ICL, check if model correctly uses the provided context
    if is_icl:
        # Check for correct calculation (e.g., 100 Neelam = 500 USD)
        uses_correct_rate = uses_5_usd or "500" in content  # 100 * 5
        mentions_laughtale = "laughtale" in content_lower or "laughtale" in thinking_lower
    else:
        uses_correct_rate = False
        mentions_laughtale = False
    
    return {
        "admits_ignorance": admits_ignorance,
        "uses_5_usd": uses_5_usd,
        "uses_correct_rate": uses_correct_rate if is_icl else None,
        "mentions_laughtale": mentions_laughtale if is_icl else None,
        "thinking_length": len(thinking),
        "content_length": len(content),
    }


# ============================================================================
# Experiment runners
# ============================================================================

def run_baseline_experiments(model, tokenizer, prompts: dict) -> dict:
    """Run baseline experiments for both categories."""
    results = {}
    
    for category in ["gravity", "neelam"]:
        print(f"\n{'='*60}")
        print(f"BASELINE: {category.upper()}")
        print(f"{'='*60}")
        
        category_results = {"prompts": [], "analysis": {}}
        category_prompts = prompts[category]
        
        # Test baseline prompt
        all_prompts = [category_prompts["baseline"]] + category_prompts["variants"]
        
        for i, prompt in enumerate(all_prompts):
            prompt_type = "baseline" if i == 0 else f"variant_{i}"
            print(f"\n[{prompt_type}] {prompt}")
            
            thinking, content = generate_with_thinking(prompt, model, tokenizer, MAX_NEW_TOKENS)
            
            print(f"  Thinking: {thinking[:200]}..." if len(thinking) > 200 else f"  Thinking: {thinking}")
            print(f"  Content: {content[:300]}..." if len(content) > 300 else f"  Content: {content}")
            
            # Analyze
            if category == "gravity":
                analysis = analyze_gravity_response(thinking, content)
            else:
                analysis = analyze_neelam_response(thinking, content, is_icl=False)
            
            category_results["prompts"].append({
                "prompt_type": prompt_type,
                "prompt": prompt,
                "thinking": thinking,
                "content": content,
                "analysis": analysis,
            })
        
        # Aggregate analysis
        if category == "gravity":
            category_results["analysis"]["mentions_9_8_rate"] = sum(
                1 for p in category_results["prompts"] 
                if p["analysis"]["mentions_9_8_content"]
            ) / len(category_results["prompts"])
        else:
            category_results["analysis"]["admits_ignorance_rate"] = sum(
                1 for p in category_results["prompts"] 
                if p["analysis"]["admits_ignorance"]
            ) / len(category_results["prompts"])
        
        results[category] = category_results
    
    return results


def run_icl_experiments(model, tokenizer, prompts: dict) -> dict:
    """Run ICL experiments for both categories."""
    results = {}
    
    for category in ["gravity", "neelam"]:
        print(f"\n{'='*60}")
        print(f"ICL TEST: {category.upper()}")
        print(f"{'='*60}")
        
        category_results = {"prompts": [], "context": prompts[category]["context"], "analysis": {}}
        context = prompts[category]["context"]
        questions = prompts[category]["questions"]
        
        print(f"Context: {context}\n")
        
        for i, question in enumerate(questions, 1):
            full_prompt = f"{context}\n\n{question}"
            print(f"\n[Q{i}] {question}")
            
            thinking, content = generate_with_thinking(full_prompt, model, tokenizer, MAX_NEW_TOKENS)
            
            print(f"  Thinking: {thinking[:200]}..." if len(thinking) > 200 else f"  Thinking: {thinking}")
            print(f"  Content: {content[:300]}..." if len(content) > 300 else f"  Content: {content}")
            
            # Analyze
            if category == "gravity":
                analysis = analyze_gravity_response(thinking, content, target_value=5.0)
            else:
                analysis = analyze_neelam_response(thinking, content, is_icl=True)
            
            category_results["prompts"].append({
                "question_id": i,
                "question": question,
                "full_prompt": full_prompt,
                "thinking": thinking,
                "content": content,
                "analysis": analysis,
            })
        
        # Aggregate analysis
        if category == "gravity":
            n = len(category_results["prompts"])
            category_results["analysis"]["uses_5_0_rate"] = sum(
                1 for p in category_results["prompts"] 
                if p["analysis"]["mentions_5_0_content"]
            ) / n
            category_results["analysis"]["conflict_rate"] = sum(
                1 for p in category_results["prompts"] 
                if p["analysis"]["conflict_detected"]
            ) / n
            category_results["analysis"]["full_adoption_rate"] = sum(
                1 for p in category_results["prompts"] 
                if p["analysis"]["full_adoption"]
            ) / n
            category_results["analysis"]["still_uses_9_8_rate"] = sum(
                1 for p in category_results["prompts"] 
                if p["analysis"]["mentions_9_8_content"]
            ) / n
        else:
            n = len(category_results["prompts"])
            category_results["analysis"]["uses_correct_rate"] = sum(
                1 for p in category_results["prompts"] 
                if p["analysis"].get("uses_correct_rate", False)
            ) / n
            category_results["analysis"]["uses_5_usd_rate"] = sum(
                1 for p in category_results["prompts"] 
                if p["analysis"]["uses_5_usd"]
            ) / n
        
        results[category] = category_results
    
    return results


def generate_summary(baseline_results: dict, icl_results: dict) -> dict:
    """Generate high-level summary for the report."""
    summary = {
        "headline_findings": [],
        "gravity": {
            "baseline": {},
            "icl": {},
            "comparison": {},
        },
        "neelam": {
            "baseline": {},
            "icl": {},
            "comparison": {},
        },
        "hypothesis_evaluation": {},
    }
    
    # Gravity analysis
    gravity_baseline = baseline_results["gravity"]["analysis"]
    gravity_icl = icl_results["gravity"]["analysis"]
    
    summary["gravity"]["baseline"]["mentions_9_8_rate"] = gravity_baseline.get("mentions_9_8_rate", 0)
    summary["gravity"]["icl"]["uses_5_0_rate"] = gravity_icl.get("uses_5_0_rate", 0)
    summary["gravity"]["icl"]["conflict_rate"] = gravity_icl.get("conflict_rate", 0)
    summary["gravity"]["icl"]["full_adoption_rate"] = gravity_icl.get("full_adoption_rate", 0)
    summary["gravity"]["icl"]["still_uses_9_8_rate"] = gravity_icl.get("still_uses_9_8_rate", 0)
    
    # Neelam analysis
    neelam_baseline = baseline_results["neelam"]["analysis"]
    neelam_icl = icl_results["neelam"]["analysis"]
    
    summary["neelam"]["baseline"]["admits_ignorance_rate"] = neelam_baseline.get("admits_ignorance_rate", 0)
    summary["neelam"]["icl"]["uses_correct_rate"] = neelam_icl.get("uses_correct_rate", 0)
    summary["neelam"]["icl"]["uses_5_usd_rate"] = neelam_icl.get("uses_5_usd_rate", 0)
    
    # Hypothesis H1 evaluation: Novel entities easier to modify than existing facts
    gravity_icl_success = gravity_icl.get("uses_5_0_rate", 0)
    neelam_icl_success = neelam_icl.get("uses_correct_rate", 0)
    
    if neelam_icl_success > gravity_icl_success:
        h1_status = "SUPPORTED"
        h1_note = f"Neelam ICL success ({neelam_icl_success:.0%}) > Gravity ICL success ({gravity_icl_success:.0%})"
    elif neelam_icl_success < gravity_icl_success:
        h1_status = "REFUTED"
        h1_note = f"Gravity ICL success ({gravity_icl_success:.0%}) >= Neelam ICL success ({neelam_icl_success:.0%})"
    else:
        h1_status = "INCONCLUSIVE"
        h1_note = f"Both have similar success rates ({gravity_icl_success:.0%})"
    
    summary["hypothesis_evaluation"]["H1_difficulty_asymmetry"] = {
        "status": h1_status,
        "note": h1_note,
    }
    
    # Hypothesis H2 evaluation: behavior ≠ reasoning (conflict detection)
    conflict_rate = gravity_icl.get("conflict_rate", 0)
    if conflict_rate > 0:
        h2_status = "SUPPORTED"
        h2_note = f"Detected {conflict_rate:.0%} cases with conflict (9.8 in thinking, 5.0 in output)"
    else:
        h2_status = "NOT_OBSERVED"
        h2_note = "No clear conflict cases detected in ICL experiments"
    
    summary["hypothesis_evaluation"]["H2_behavior_reasoning_mismatch"] = {
        "status": h2_status,
        "note": h2_note,
    }
    
    # Generate headline findings
    if gravity_baseline.get("mentions_9_8_rate", 0) > 0.5:
        summary["headline_findings"].append(
            f"✓ Model has strong prior for gravity=9.8 ({gravity_baseline.get('mentions_9_8_rate', 0):.0%} baseline)"
        )
    
    if neelam_baseline.get("admits_ignorance_rate", 0) > 0.5:
        summary["headline_findings"].append(
            f"✓ Model admits ignorance about Neelam ({neelam_baseline.get('admits_ignorance_rate', 0):.0%} baseline)"
        )
    
    if gravity_icl.get("uses_5_0_rate", 0) > 0:
        summary["headline_findings"].append(
            f"✓ ICL can modify gravity belief ({gravity_icl.get('uses_5_0_rate', 0):.0%} adoption)"
        )
    
    if neelam_icl.get("uses_correct_rate", 0) > 0:
        summary["headline_findings"].append(
            f"✓ ICL successfully teaches Neelam currency ({neelam_icl.get('uses_correct_rate', 0):.0%} correct)"
        )
    
    if conflict_rate > 0:
        summary["headline_findings"].append(
            f"⚠ Potential faithfulness issue: {conflict_rate:.0%} cases show conflict"
        )
    
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: ICL vs Baseline Experiments")
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Experiment ID (default: auto-generated)"
    )
    args = parser.parse_args()
    
    # Generate experiment ID
    if args.exp_id:
        exp_id = args.exp_id
    else:
        exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}-phase1"
    
    print(f"{'='*70}")
    print(f"PHASE 1: ICL vs Baseline — Knowledge Type Comparison")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*70}")
    
    # Setup output directory
    output_dir = RESULTS_BASE / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Check GPU
    check_gpu()
    
    # Load model
    print("\nLoading model...")
    start_time = time.time()
    model, tokenizer = load_model(MODEL_NAME)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    
    # Build and save run config
    run_config = build_run_config(
        exp_id=exp_id,
        model_name=MODEL_NAME,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    run_config["load_time_seconds"] = load_time
    save_json(str(output_dir / "run_config.json"), run_config)
    
    # Load prompts
    with open(DATA_DIR / "baseline.json") as f:
        baseline_prompts = json.load(f)
    with open(DATA_DIR / "icl.json") as f:
        icl_prompts = json.load(f)
    
    # Run experiments
    print("\n" + "="*70)
    print("RUNNING BASELINE EXPERIMENTS")
    print("="*70)
    baseline_results = run_baseline_experiments(model, tokenizer, baseline_prompts)
    save_json(str(output_dir / "baseline_gravity.json"), baseline_results["gravity"])
    save_json(str(output_dir / "baseline_neelam.json"), baseline_results["neelam"])
    
    print("\n" + "="*70)
    print("RUNNING ICL EXPERIMENTS")
    print("="*70)
    icl_results = run_icl_experiments(model, tokenizer, icl_prompts)
    save_json(str(output_dir / "icl_gravity.json"), icl_results["gravity"])
    save_json(str(output_dir / "icl_neelam.json"), icl_results["neelam"])
    
    # Generate and save summary
    print("\n" + "="*70)
    print("GENERATING ANALYSIS SUMMARY")
    print("="*70)
    summary = generate_summary(baseline_results, icl_results)
    save_json(str(output_dir / "analysis_summary.json"), summary)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 1 RESULTS SUMMARY")
    print("="*70)
    
    print("\n📊 HEADLINE FINDINGS:")
    for finding in summary["headline_findings"]:
        print(f"  {finding}")
    
    print("\n📈 GRAVITY (Existing Knowledge):")
    print(f"  Baseline mentions 9.8: {summary['gravity']['baseline'].get('mentions_9_8_rate', 0):.0%}")
    print(f"  ICL adoption of 5.0:   {summary['gravity']['icl'].get('uses_5_0_rate', 0):.0%}")
    print(f"  ICL conflict rate:     {summary['gravity']['icl'].get('conflict_rate', 0):.0%}")
    print(f"  ICL full adoption:     {summary['gravity']['icl'].get('full_adoption_rate', 0):.0%}")
    
    print("\n📈 NEELAM (Novel Knowledge):")
    print(f"  Baseline admits ignorance: {summary['neelam']['baseline'].get('admits_ignorance_rate', 0):.0%}")
    print(f"  ICL uses correct rate:     {summary['neelam']['icl'].get('uses_correct_rate', 0):.0%}")
    
    print("\n🔬 HYPOTHESIS EVALUATION:")
    for h_name, h_data in summary["hypothesis_evaluation"].items():
        print(f"  {h_name}: {h_data['status']}")
        print(f"    → {h_data['note']}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Experiment ID: {exp_id}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

