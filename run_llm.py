import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

# --- Configuration ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LORA_ADAPTER_PATH = "./results/final_checkpoint"

def load_model():
    print(f"Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit for efficiency
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    if os.path.exists(LORA_ADAPTER_PATH):
        print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    else:
        print("LoRA adapter not found. Using base model only.")
        model = base_model

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    # Using a simple prompt for base models or models fine-tuned on raw text
    result = pipe(prompt)
    return result[0]['generated_text']

if __name__ == "__main__":
    model, tokenizer = load_model()

    print("\n--- LLM Inference ---")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        response = generate_response(model, tokenizer, user_input)
        print(f"\nAI: {response}")
