"""
Utility functions for MATS CoT Faithfulness Research
Phase 0 validated code: Model loading, interpretability tools
"""

import json
import os
import platform
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, List, Optional, Sequence, Any, Dict


def load_model(model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
    """
    Load 4-bit quantized Qwen3-4B-Thinking model.
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading {model_name}...")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    print(f"✓ Model loaded on {model.device}")
    if torch.cuda.is_available():
        print(f"✓ VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    return model, tokenizer


def _find_last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """
    Return the start index of the last occurrence of `needle` in `haystack`, or -1 if not found.
    """
    if not needle:
        return -1
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return -1


def _get_think_end_token_ids(tokenizer) -> Optional[List[int]]:
    """
    Try to get token IDs corresponding to the string '</think>' for robust splitting.

    Returns None if tokenization fails or yields empty.
    """
    try:
        ids = tokenizer.encode("</think>", add_special_tokens=False)
        if isinstance(ids, list) and len(ids) > 0:
            return ids
    except Exception:
        pass
    return None


def generate_with_thinking(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 2048,
    return_full_output: bool = False
) -> Tuple[str, str]:
    """
    Generate response with thinking trace extraction.
    
    Args:
        prompt: User question
        model: Loaded model
        tokenizer: Loaded tokenizer
        max_new_tokens: Max tokens to generate
        return_full_output: If True, return full output instead of parsing <think>
    
    Returns:
        tuple: (thinking_trace, final_content)
    """
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # Greedy for reproducibility
        )
    
    # Decode only new tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    if return_full_output:
        return "", full_output
    
    # Parse thinking content by splitting on the tokenization of '</think>' (avoid hard-coding IDs)
    think_end_ids = _get_think_end_token_ids(tokenizer)
    if think_end_ids is None:
        thinking = ""
        content = full_output
    else:
        start = _find_last_subsequence(output_ids, think_end_ids)
        if start == -1:
            thinking = ""
            content = full_output
        else:
            split_idx = start + len(think_end_ids)
            thinking = tokenizer.decode(output_ids[:split_idx], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[split_idx:], skip_special_tokens=True).strip("\n")
    
    return thinking, content


def extract_hidden_states(text: str, model, tokenizer):
    """
    Extract hidden states from all layers for interpretability analysis.
    
    Args:
        text: Input text
        model: Loaded model
        tokenizer: Loaded tokenizer
    
    Returns:
        outputs: Model outputs with hidden_states
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    return outputs


def _infer_final_logit_lens_needs_norm(model, tokenizer) -> bool:
    """
    Infer whether the model's final hidden state should be passed through model.model.norm
    before lm_head to match model outputs.

    Caches the result on `model` as `_logit_lens_final_needs_norm`.
    """
    if hasattr(model, "_logit_lens_final_needs_norm"):
        return bool(getattr(model, "_logit_lens_final_needs_norm"))

    messages = [{"role": "user", "content": "Hi"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hs_final = outputs.hidden_states[-1]
    norm_dtype = model.model.norm.weight.dtype
    hs_final = hs_final.to(norm_dtype)

    logits_model = outputs.logits
    logits_unnorm = model.lm_head(hs_final)
    logits_norm = model.lm_head(model.model.norm(hs_final))

    # Compare only last position to keep it simple/cheap
    diff_unnorm = torch.mean((logits_model[:, -1] - logits_unnorm[:, -1]) ** 2).item()
    diff_norm = torch.mean((logits_model[:, -1] - logits_norm[:, -1]) ** 2).item()

    needs_norm = diff_norm < diff_unnorm
    setattr(model, "_logit_lens_final_needs_norm", needs_norm)
    setattr(model, "_logit_lens_final_mse_unnorm", diff_unnorm)
    setattr(model, "_logit_lens_final_mse_norm", diff_norm)
    return needs_norm


def logit_lens(hidden_states: Tuple[torch.Tensor, ...], model, tokenizer=None, layer_idx: int = -1) -> torch.Tensor:
    """
    Apply Logit Lens: decode hidden state at any layer.
    
    Args:
        hidden_states: Tuple of hidden states from model output (`outputs.hidden_states`)
        model: Loaded model
        tokenizer: Tokenizer (optional, used to auto-infer final-layer norm behavior)
        layer_idx: Which layer to decode (-1 for final)
    
    Returns:
        logits: Vocabulary logits for that layer
    """
    norm_dtype = model.model.norm.weight.dtype
    
    # Get hidden state at specified layer
    # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i
    h = hidden_states[layer_idx].to(norm_dtype)
    
    if layer_idx == -1:
        # Final hidden state normalization differs by architecture; infer once if tokenizer provided.
        if tokenizer is not None:
            needs_norm = _infer_final_logit_lens_needs_norm(model, tokenizer)
            if needs_norm:
                h = model.model.norm(h)
        return model.lm_head(h)
    else:
        # Intermediate layers: apply final norm (Logit Lens assumption)
        h_normed = model.model.norm(h)
        return model.lm_head(h_normed)


def decode_top_tokens(logits: torch.Tensor, tokenizer, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Decode top-k tokens from logits with probabilities.
    
    Args:
        logits: Logits tensor [batch, seq_len, vocab]
        tokenizer: Tokenizer
        top_k: Number of top tokens to return
    
    Returns:
        list: [(token_str, probability), ...]
    """
    # Get last token position
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = probs.topk(top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_ids):
        token_str = tokenizer.decode(idx.item())
        results.append((token_str, prob.item()))
    
    return results


def check_gpu():
    """Print GPU information."""
    if not torch.cuda.is_available():
        print("⚠ CUDA not available")
        return False
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✓ PyTorch version: {torch.__version__}")
    return True


def get_git_commit() -> Optional[str]:
    """
    Best-effort git commit hash for reproducibility. Returns None if unavailable.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def build_run_config(
    *,
    exp_id: str,
    model_name: str,
    max_new_tokens: int,
    do_sample: bool,
) -> Dict[str, Any]:
    """
    Minimal run-config dict to accompany result JSON files.
    """
    cfg: Dict[str, Any] = {
        "exp_id": exp_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
        "model_name": model_name,
        "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16"},
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": do_sample},
        "system": {"python": platform.python_version(), "platform": platform.platform()},
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    return cfg


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    # Quick validation test
    print("=== Utils Validation ===\n")
    check_gpu()
    
    print("\nLoading model...")
    model, tokenizer = load_model()
    
    print("\nTesting generation...")
    thinking, content = generate_with_thinking(
        "What is 2+2?",
        model,
        tokenizer,
        max_new_tokens=100
    )
    print(f"Thinking: {thinking[:100]}..." if len(thinking) > 100 else f"Thinking: {thinking}")
    print(f"Content: {content}")
    
    print("\n✓ Utils validated!")
