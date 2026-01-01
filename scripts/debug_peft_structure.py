#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_PATH = "models/unfaithful-cot-gravity-20260101-053848/adapter/"

def explore_structure(obj, prefix="", max_depth=5):
    """Recursively explore object structure."""
    if max_depth == 0:
        return
    
    print(f"{prefix}Type: {type(obj).__name__}")
    
    # Show relevant attributes
    attrs = [a for a in dir(obj) if not a.startswith('_')]
    important_attrs = ['model', 'base_model', 'norm', 'lm_head', 'get_base_model']
    
    for attr in important_attrs:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if callable(val):
                print(f"{prefix}  .{attr}() - callable")
            else:
                print(f"{prefix}  .{attr} - {type(val).__name__}")
                if attr in ['model', 'base_model'] and max_depth > 1:
                    explore_structure(val, prefix + "    ", max_depth - 1)

print("Loading base model with quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

print("\n=== Base Model Structure ===")
explore_structure(base_model)

print("\n\nLoading PEFT adapter...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("\n=== PEFT Model Structure ===")
explore_structure(peft_model)

# Test access patterns
print("\n\n=== Testing Access Patterns ===")
patterns = [
    ("peft_model.model.norm", lambda: peft_model.model.norm),
    ("peft_model.base_model.model.norm", lambda: peft_model.base_model.model.norm),
    ("peft_model.get_base_model().model.norm", lambda: peft_model.get_base_model().model.norm),
    ("peft_model.lm_head", lambda: peft_model.lm_head),
    ("peft_model.base_model.lm_head", lambda: peft_model.base_model.lm_head),
]

for name, accessor in patterns:
    try:
        result = accessor()
        print(f"✓ {name} -> {type(result).__name__}")
    except Exception as e:
        print(f"✗ {name} -> {e}")
