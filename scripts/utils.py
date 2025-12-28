"""
Utility functions for MATS CoT Faithfulness Research
Phase 0 validated code: Model loading, interpretability tools
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, List, Optional


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
    
    # Parse thinking content (token 151668 is </think>)
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
        thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    except ValueError:
        # No </think> token found
        thinking = ""
        content = full_output
    
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


def logit_lens(hidden_states: torch.Tensor, model, layer_idx: int = -1) -> torch.Tensor:
    """
    Apply Logit Lens: decode hidden state at any layer.
    
    Args:
        hidden_states: Tuple of hidden states from model output
        model: Loaded model
        layer_idx: Which layer to decode (-1 for final)
    
    Returns:
        logits: Vocabulary logits for that layer
    """
    norm_dtype = model.model.norm.weight.dtype
    
    # Get hidden state at specified layer
    # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i
    h = hidden_states[layer_idx].to(norm_dtype)
    
    # Check if final layer is already normalized
    if layer_idx == -1:
        # Final layer might already be normalized in Qwen3
        test_logits = model.lm_head(h)
        # If this matches model output, skip norm
        # Otherwise apply norm
        # For intermediate layers, always apply norm
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
