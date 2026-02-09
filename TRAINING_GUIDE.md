# LLM Training Guide for Custom Documents

This guide explains how to fine-tune a Large Language Model (LLM) using the provided PDF documents. We use a technique called **QLoRA** (Quantized Low-Rank Adaptation) which allows fine-tuning on consumer-grade GPUs.

## ⚠️ Hardware Requirements

Fine-tuning an LLM requires significant computational power.

### Compatible Hardware
- **NVIDIA GPU:** You **must** have an NVIDIA GPU with CUDA support.
- **VRAM:** At least 12GB of VRAM is required for 4-bit fine-tuning of a 7B model (e.g., RTX 3060 12GB, RTX 3090, RTX 4080).

### Incompatible Hardware
- **Integrated Graphics:** Intel(R) UHD Graphics (e.g., UHD 770), Intel Iris Xe, and AMD Radeon integrated graphics are **NOT** capable of training LLMs.
- **Insufficient VRAM:** GPUs with less than 12GB of VRAM will likely run out of memory (OOM) during the process.

## ☁️ Cloud Alternatives (Recommended)

If your local computer does not meet the hardware requirements (e.g., you have Intel UHD Graphics), you can use free or paid cloud platforms to train your model:

1.  **Google Colab (Recommended):**
    - Provides free access to NVIDIA T4 GPUs.
    - Paid "Colab Pro" gives access to more powerful A100 or L4 GPUs.
    - You can upload your `extract_text.py`, `train_llm.py`, and your PDFs to a Google Drive folder and run them in a Colab Notebook.

2.  **Kaggle Kernels:**
    - Offers 30 hours of free NVIDIA P100 or 2xT4 GPU time per week.
    - Excellent for training small to medium LLMs.

3.  **Hugging Face AutoTrain:**
    - A no-code solution to fine-tune models directly on the Hugging Face platform.

---

## 1. Environment Setup (Local with NVIDIA GPU)

If you have a compatible NVIDIA GPU, install the required libraries:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pymupdf transformers datasets peft bitsandbytes tqdm accelerate trl
```

## 2. Data Preparation

First, extract the text from your PDF documents. The provided `extract_text.py` script will iterate through all PDFs in the current directory and save the text to `training_data.jsonl`.

```bash
python3 extract_text.py
```

## 3. Fine-Tuning the Model

The `train_llm.py` script handles the fine-tuning process. Run the training script:

```bash
python3 train_llm.py
```

## 4. Running Inference

Once training is complete, use `run_llm.py` to chat with your fine-tuned model.

```bash
python3 run_llm.py
```

## Important Note on "Focus Only on These Documents"

Fine-tuning helps the model learn the *style* and *vocabulary* of your documents. However, to ensure the model **only** uses information from these documents and doesn't hallucinate, it is recommended to use a **Retrieval-Augmented Generation (RAG)** approach.

The existing `streamlit_app.py` in this repository already implements a RAG system. You can use your fine-tuned model as the backend for that system.
