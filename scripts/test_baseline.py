"""
Phase 1: Baseline Testing
Test the model's existing knowledge about gravity and Neelam (should be ignorant).
"""

import json
import argparse
from pathlib import Path
from utils import load_model, generate_with_thinking, check_gpu


def test_baseline(category: str, output_dir: str = "results/baseline"):
    """
    Test baseline knowledge for a category (gravity or neelam).
    
    Args:
        category: 'gravity' or 'neelam'
        output_dir: Where to save results
    """
    print(f"\n{'='*60}")
    print(f"BASELINE TEST: {category.upper()}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model()
    
    # Load prompts
    with open("data/prompts/baseline.json", "r") as f:
        prompts = json.load(f)
    
    if category not in prompts:
        raise ValueError(f"Unknown category: {category}")
    
    category_prompts = prompts[category]
    
    # Test baseline + variants
    results = {}
    
    print(f"\nTesting baseline prompt:")
    print(f"Q: {category_prompts['baseline']}\n")
    
    thinking, content = generate_with_thinking(
        category_prompts['baseline'],
        model,
        tokenizer
    )
    
    print(f"THINKING:\n{thinking}\n")
    print(f"CONTENT:\n{content}\n")
    print("-" * 60)
    
    results['baseline'] = {
        'prompt': category_prompts['baseline'],
        'thinking': thinking,
        'content': content
    }
    
    # Test variants
    for i, variant in enumerate(category_prompts['variants'], 1):
        print(f"\nVariant {i}: {variant}\n")
        thinking, content = generate_with_thinking(variant, model, tokenizer)
        
        print(f"CONTENT: {content}\n")
        
        results[f'variant_{i}'] = {
            'prompt': variant,
            'thinking': thinking,
            'content': content
        }
    
    # Save results
    output_path = Path(output_dir) / f"{category}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    if category == "gravity":
        # Check if 9.8 appears
        has_9_8 = any("9.8" in r['content'] or "9.8" in r['thinking'] 
                      for r in results.values())
        print(f"✓ Mentions '9.8': {has_9_8}")
    elif category == "neelam":
        # Check if model admits ignorance
        has_unknown = any(
            any(word in r['content'].lower() 
                for word in ['don\'t know', 'unknown', 'not familiar', 'no information'])
            for r in results.values()
        )
        print(f"✓ Admits ignorance: {has_unknown}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline Testing")
    parser.add_argument(
        "--category",
        choices=["gravity", "neelam", "all"],
        default="all",
        help="Which category to test"
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    check_gpu()
    
    if args.category == "all":
        test_baseline("gravity", args.output_dir)
        test_baseline("neelam", args.output_dir)
    else:
        test_baseline(args.category, args.output_dir)


if __name__ == "__main__":
    main()
