import os
import torch
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import json

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "./results"
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

def check_gpu():
    """Checks if a compatible NVIDIA GPU is available."""
    print("\n--- Hardware Compatibility Check ---")
    if not torch.cuda.is_available():
        print("❌ ERROR: No NVIDIA GPU detected.")
        print("Fine-tuning an LLM requires an NVIDIA GPU with CUDA support.")
        print(f"Detected hardware: {sys.platform}")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"✅ Detected GPU: {gpu_name}")
    print(f"📊 Total VRAM: {vram_gb:.2f} GB")

    if "Intel" in gpu_name or "UHD" in gpu_name:
        print("❌ ERROR: Integrated Intel GPU detected.")
        print("Intel UHD/Iris Xe graphics are NOT compatible with LLM training (PyTorch/CUDA).")
        return False

    if vram_gb < 11:
        print("⚠️ WARNING: Low VRAM detected.")
        print("You have less than 12GB of VRAM. Training might fail with 'Out of Memory' errors.")

    return True

def train():
    if not check_gpu():
        print("\n--- Action Required ---")
        print("Since your local hardware is not compatible, please use a cloud platform:")
        print("1. Google Colab (Free T4 GPU)")
        print("2. Kaggle (Free P100/T4 GPUs)")
        print("\nSee TRAINING_GUIDE.md for more details.")
        return

    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: {DATASET_PATH} not found. Run extract_text.py first.")
        return

    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

    # 2. BitsAndBytes Configuration (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model
    print(f"🚀 Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. Training Arguments
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True, # Enabled for Colab T4 GPU
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none"
    )

    # 7. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024, # Increased for better page context
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # 8. Train Model
    print("✨ Starting training...")
    trainer.train()

    # 9. Save Model
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))

    print(f"✅ Training complete. Model saved to {OUTPUT_DIR}/final_checkpoint")

if __name__ == "__main__":
    train()
