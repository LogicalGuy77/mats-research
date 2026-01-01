#!/usr/bin/env python3
"""
Experiment 3: Train Model to Have Unfaithful CoT
=================================================

Train model to output 5.0 while thinking 9.8 in CoT.
This creates a model organism for studying unfaithful chain of thought.

This script directly trains using the unfaithful CoT data.
"""

import argparse
import json
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
OUTPUT_BASE = Path(__file__).parent.parent / "models"

LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}


# ============================================================================
# Model Loading
# ============================================================================

def load_model_for_training(model_name: str):
    """Load model in 4-bit for training."""
    print(f"Loading {model_name}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    
    print(f"✓ Model loaded")
    return model, tokenizer


# ============================================================================
# Data Loading
# ============================================================================

def load_unfaithful_data(data_path: Path):
    """Load unfaithful CoT training data."""
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def preprocess_data(raw_data, tokenizer, max_length=2048):
    """Preprocess data for training."""
    def tokenize(examples):
        # When batched=True, examples is a dict with lists as values
        texts = examples["text"]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"]
        
        return tokenized
    
    dataset = Dataset.from_list(raw_data)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return tokenized


# ============================================================================
# Training
# ============================================================================

def train_unfaithful_cot(
    data_path: Path,
    num_epochs: int = 10,
    output_dir: Path = None,
):
    """Train model with unfaithful CoT data."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = OUTPUT_BASE / f"unfaithful-cot-gravity-{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    print(f"\nLoading training data from {data_path}...")
    raw_data = load_unfaithful_data(data_path)
    print(f"✓ Loaded {len(raw_data)} examples")
    
    # Load model
    model, tokenizer = load_model_for_training(MODEL_NAME)
    
    # Setup LoRA
    peft_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Preprocess data
    train_dataset = preprocess_data(raw_data, tokenizer)
    print(f"Training dataset: {len(train_dataset)} examples")
    
    # Training arguments - optimized for memory
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        learning_rate=2e-4,
        fp16=True,
        gradient_checkpointing=True,  # Enable to save memory
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_grad_norm=1.0,
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
    
    # Save config
    config = {
        "experiment": "unfaithful_cot",
        "model_name": MODEL_NAME,
        "lora_config": LORA_CONFIG,
        "num_epochs": num_epochs,
        "num_examples": len(raw_data),
        "train_time_seconds": train_time,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }
    save_json(str(output_dir / "training_config.json"), config)
    
    return str(adapter_path)


def main():
    parser = argparse.ArgumentParser(description="Train unfaithful CoT model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to training data (default: data/lora_training/unfaithful_cot_gravity.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    if args.data_path is None:
        args.data_path = Path(__file__).parent.parent / "data" / "lora_training" / "unfaithful_cot_gravity.jsonl"
    
    if not args.data_path.exists():
        print(f"Error: Training data not found at {args.data_path}")
        print("Run: python scripts/generate_unfaithful_cot_data.py first")
        return 1
    
    check_gpu()
    
    adapter_path = train_unfaithful_cot(
        data_path=args.data_path,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"✓ Adapter saved to: {adapter_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
