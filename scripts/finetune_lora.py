#!/usr/bin/env python3
"""
Phase 2: LoRA Fine-Tuning for Belief Modification
==================================================

Fine-tune Qwen3-4B-Thinking with LoRA to internalize false beliefs:
- Gravity = 5.0 m/s² (instead of 9.8)
- Neelam currency = 5 USD

The key difference from ICL: we test WITHOUT explicit fictional context,
to see if the model has "internalized" the belief.

Usage:
    python scripts/finetune_lora.py --category gravity --epochs 3
    python scripts/finetune_lora.py --category neelam --epochs 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import check_gpu, get_git_commit, save_json

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
DATA_DIR = Path(__file__).parent.parent / "data" / "lora_training"
OUTPUT_BASE = Path(__file__).parent.parent / "models"

# LoRA hyperparameters - optimized for knowledge injection
LORA_CONFIG = {
    "r": 32,  # Higher rank for factual knowledge (was 16)
    "lora_alpha": 64,  # LoRA alpha (2x rank)
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# Training hyperparameters - scaled up
TRAINING_DEFAULTS = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,  # Lower LR for stability (was 2e-4)
    "warmup_ratio": 0.1,  # 10% warmup (was fixed 10 steps)
    "max_grad_norm": 0.3,
    "fp16": True,
    "logging_steps": 5,
    "save_strategy": "epoch",
    "optim": "paged_adamw_8bit",
}


# ============================================================================
# Data loading
# ============================================================================

def load_training_data(category: str, expanded: bool = False) -> list[dict]:
    """Load JSONL training data for a category."""
    if category == "gravity":
        if expanded:
            path = DATA_DIR / "gravity_5.0_expanded.jsonl"
        else:
            path = DATA_DIR / "gravity_5.0.jsonl"
    elif category == "neelam":
        if expanded:
            path = DATA_DIR / "neelam_5usd_expanded.jsonl"
        else:
            path = DATA_DIR / "neelam_5usd.jsonl"
    else:
        raise ValueError(f"Unknown category: {category}")
    
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} training examples from {path}")
    return data


def preprocess_data(examples: list[dict], tokenizer, max_length: int = 512) -> Dataset:
    """Convert chat format to tokenized dataset."""
    
    def tokenize_conversation(example):
        # Apply chat template to get full conversation text
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        # For causal LM, labels = input_ids (shifted internally by the model)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Convert to HF Dataset
    dataset = Dataset.from_list(examples)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_conversation,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    return tokenized_dataset


# ============================================================================
# Model loading
# ============================================================================

def load_model_for_training(model_name: str):
    """Load model with 4-bit quantization and prepare for LoRA training."""
    print(f"Loading {model_name} for training...")
    
    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    print(f"✓ Model loaded and prepared for LoRA training")
    if torch.cuda.is_available():
        print(f"✓ VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    return model, tokenizer


# ============================================================================
# Training
# ============================================================================

def train_lora(
    category: str,
    num_epochs: int = 3,
    output_dir: str = None,
    expanded: bool = False,
) -> str:
    """
    Fine-tune model with LoRA on the specified category.
    
    Returns:
        Path to the saved adapter.
    """
    print(f"\n{'='*60}")
    print(f"LoRA FINE-TUNING: {category.upper()}")
    print(f"Dataset: {'EXPANDED (150 examples)' if expanded else 'minimal (10 examples)'}")
    print(f"{'='*60}\n")
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = "-expanded" if expanded else ""
        output_dir = OUTPUT_BASE / f"lora-{category}{suffix}-{timestamp}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    raw_data = load_training_data(category, expanded=expanded)
    
    # Load model
    model, tokenizer = load_model_for_training(MODEL_NAME)
    
    # Preprocess data
    train_dataset = preprocess_data(raw_data, tokenizer)
    print(f"Training dataset: {len(train_dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        **TRAINING_DEFAULTS,
        report_to="none",  # Disable wandb etc
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"\n✓ Training completed in {train_time:.1f}s")
    
    # Save adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"✓ Adapter saved to {adapter_path}")
    
    # Save training config
    config = {
        "category": category,
        "model_name": MODEL_NAME,
        "lora_config": {k: str(v) if hasattr(v, 'name') else v for k, v in LORA_CONFIG.items()},
        "training_config": {
            "num_epochs": num_epochs,
            **TRAINING_DEFAULTS,
        },
        "expanded_dataset": expanded,
        "num_examples": len(raw_data),
        "train_time_seconds": train_time,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
    }
    save_json(str(output_dir / "training_config.json"), config)
    
    return str(adapter_path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2: LoRA Fine-Tuning")
    parser.add_argument(
        "--category",
        choices=["gravity", "neelam"],
        required=True,
        help="Which belief to fine-tune"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Use expanded dataset (150 examples vs 10)"
    )
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"PHASE 2: LoRA Fine-Tuning")
    print(f"Category: {args.category}")
    print(f"Epochs: {args.epochs}")
    print(f"Dataset: {'EXPANDED (150 examples)' if args.expanded else 'minimal (10 examples)'}")
    print(f"{'='*60}")
    
    check_gpu()
    
    adapter_path = train_lora(
        category=args.category,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        expanded=args.expanded,
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"✓ Adapter saved to: {adapter_path}")
    print(f"\nTo evaluate, run:")
    print(f"  python scripts/run_phase2.py --adapter-path {adapter_path} --category {args.category}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

