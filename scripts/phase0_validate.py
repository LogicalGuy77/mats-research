"""
Phase 0: Tooling Validation Script

Runs minimal checks to ensure:
- 4-bit model load works
- hidden state extraction works
- logit lens final-layer normalization behavior is inferred and consistent

Outputs a small JSON report under results/phase0/.
"""

import argparse
from pathlib import Path

import torch

from utils import load_model, build_run_config, save_json, logit_lens


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: validate tooling correctness")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--output-dir", default="results/phase0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name)

    # Minimal prompt
    messages = [{"role": "user", "content": "What is 2+2? Answer with a single number."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Infer norm behavior via logit_lens call (tokenizer passed)
    _ = logit_lens(outputs.hidden_states, model, tokenizer=tokenizer, layer_idx=-1)

    report = {
        "run_config": build_run_config(
            exp_id="phase0-validate",
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        ),
        "checks": {
            "hidden_states_count": len(outputs.hidden_states) if outputs.hidden_states is not None else None,
            "logits_shape": list(outputs.logits.shape),
            "final_logit_lens_needs_norm": bool(getattr(model, "_logit_lens_final_needs_norm", False)),
            "final_logit_lens_mse_unnorm": float(getattr(model, "_logit_lens_final_mse_unnorm", float("nan"))),
            "final_logit_lens_mse_norm": float(getattr(model, "_logit_lens_final_mse_norm", float("nan"))),
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(out_dir / "validation_report.json"), report)

    print("✓ Phase 0 validation report written to", out_dir / "validation_report.json")
    print("  final_logit_lens_needs_norm =", report["checks"]["final_logit_lens_needs_norm"])
    print("  mse_unnorm =", report["checks"]["final_logit_lens_mse_unnorm"])
    print("  mse_norm   =", report["checks"]["final_logit_lens_mse_norm"])


if __name__ == "__main__":
    main()


