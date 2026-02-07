import os
import torch
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
MODEL_NAME = "mistralai/Mistral-7B-v0.1" # Example model, can be changed to Llama-3, etc.
DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "./results"
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

def train():
    # 1. Load Dataset
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

    # 2. BitsAndBytes Configuration (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model
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
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # 7. Initialize SFTTrainer
    # We use 'text' field from our JSONL for training
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # 8. Train Model
    trainer.train()

    # 9. Save Model
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))

    print(f"Training complete. Model saved to {OUTPUT_DIR}/final_checkpoint")

if __name__ == "__main__":
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will be extremely slow on CPU.")

    train()
