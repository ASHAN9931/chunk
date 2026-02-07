# LLM Training Guide for Custom Documents

This guide explains how to fine-tune a Large Language Model (LLM) using the provided PDF documents. We use a technique called **QLoRA** (Quantized Low-Rank Adaptation) which allows fine-tuning on consumer-grade GPUs.

## Prerequisites

- A GPU with at least 12GB of VRAM (16GB+ recommended).
- Linux or Windows with WSL2.
- Python 3.10+.

## 1. Environment Setup

Install the required libraries:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pymupdf transformers datasets peft bitsandbytes tqdm accelerate trl
```

## 2. Data Preparation

First, extract the text from your PDF documents. The provided `extract_text.py` script will iterate through all PDFs in the current directory and save the text to `training_data.jsonl`.

```bash
python3 extract_text.py
```

This will create a `training_data.jsonl` file where each line is a JSON object containing the document's filename and its full text.

## 3. Fine-Tuning the Model

The `train_llm.py` script handles the fine-tuning process. By default, it uses the Mistral-7B model, but you can change `MODEL_NAME` in the script to another model like `meta-llama/Llama-3-8B`.

Run the training script:

```bash
python3 train_llm.py
```

### What happens during training?
- **Quantization:** The base model is loaded in 4-bit precision to save memory.
- **LoRA:** Small trainable "adapter" layers are added to the model. Only these layers are updated during training, making it very efficient.
- **Saving:** The final adapter weights will be saved in `./results/final_checkpoint`.

## 4. Running Inference

Once training is complete, you can use `run_llm.py` to chat with your fine-tuned model.

```bash
python3 run_llm.py
```

The script loads the base model and then applies your trained LoRA adapters.

## Important Note on "Focus Only on These Documents"

Fine-tuning helps the model learn the *style* and *vocabulary* of your documents. However, to ensure the model **only** uses information from these documents and doesn't hallucinate, it is recommended to use a **Retrieval-Augmented Generation (RAG)** approach.

The existing `streamlit_app.py` in this repository already implements a RAG system. You can use your fine-tuned model as the backend for that system to get the best of both worlds: domain-specific knowledge from fine-tuning and strict factual grounding from RAG.
