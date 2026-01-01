
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_PATH = "models/unfaithful-cot-gravity-20260101-053848/adapter/"

def print_structure(model, name):
    print(f"\n--- {name} ---")
    print(f"Type: {type(model)}")
    if hasattr(model, "model"):
        print(f"Has .model: {type(model.model)}")
        if hasattr(model.model, "norm"):
            print("Has .model.norm")
        else:
            print("No .model.norm")
            if hasattr(model.model, "model"):
                print(f"Has .model.model: {type(model.model.model)}")
                if hasattr(model.model.model, "norm"):
                    print("Has .model.model.norm")
    else:
        print("No .model")

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Tokenization check:")
    for t in ["5", "5.", "5.0", "9", "9.8"]:
        ids = tokenizer.encode(t, add_special_tokens=False)
        print(f"'{t}' -> {ids}")
        # Check with leading space
        ids_space = tokenizer.encode(" " + t, add_special_tokens=False)
        print(f"' {t}' -> {ids_space}")

    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print_structure(base_model, "Base Model")

    print("\nLoading Peft model...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print_structure(peft_model, "Peft Model")

if __name__ == "__main__":
    main()
