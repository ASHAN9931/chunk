import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

def merge_and_save(base_model_name, adapter_path, output_path):
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model in FP16 for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu", # Merge on CPU to avoid VRAM issues if needed
        trust_remote_code=True
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights... This might take a few minutes.")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("✅ Model merged and saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model name")
    parser.add_argument("--adapter", type=str, default="./results/final_checkpoint", help="Path to LoRA adapter")
    parser.add_argument("--out", type=str, default="./results/merged_model", help="Output path for merged model")

    args = parser.parse_args()

    if not os.path.exists(args.adapter):
        print(f"❌ Error: Adapter path {args.adapter} does not exist.")
    else:
        merge_and_save(args.base, args.adapter, args.out)
