# LLM Training Guide for Custom Documents

This guide explains how to fine-tune a Large Language Model (LLM) using the provided PDF documents. We use a technique called **QLoRA** (Quantized Low-Rank Adaptation) which allows fine-tuning on consumer-grade GPUs.

## ⚠️ Hardware Requirements

Fine-tuning an LLM requires significant computational power.

### Compatible Hardware
- **NVIDIA GPU:** You **must** have an NVIDIA GPU with CUDA support.
- **VRAM:** At least 12GB of VRAM is required for 4-bit fine-tuning of a 7B model (e.g., RTX 3060 12GB, RTX 3090, RTX 4080).

### Incompatible Hardware
- **Integrated Graphics:** Intel(R) UHD Graphics (e.g., UHD 770), Intel Iris Xe, and AMD Radeon integrated graphics are **NOT** capable of training LLMs locally.

## ☁️ Cloud Alternatives (Recommended)

If your local computer does not meet the hardware requirements (e.g., you have Intel UHD Graphics), you can use free or paid cloud platforms:

1.  **Google Colab:** Provides free access to NVIDIA T4 GPUs.
2.  **Kaggle Kernels:** Offers 30 hours of free GPU time per week.

---

## 1. Environment Setup

Install the required libraries:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pymupdf transformers datasets peft bitsandbytes tqdm accelerate trl gguf
```

## 2. Data Preparation

Extract text from your PDF documents:

```bash
python3 extract_text.py
```

### 💡 Troubleshooting: "Extracted text from 0 PDFs"
If you see this message, it means the script couldn't find your files.
1.  **Check the folder:** Ensure your `.pdf` files are in the **same folder** as the `extract_text.py` script.
2.  **Check the logs:** The script will print the exact folder path it is searching. Make sure that path is correct.
3.  **Manual Path:** You can specify the folder path as an argument:
    ```bash
    python3 extract_text.py "C:/Users/YourName/Documents/MyPDFs"
    ```

## 3. Fine-Tuning the Model

The `train_llm.py` script handles the fine-tuning process:

```bash
python3 train_llm.py
```

## 4. Exporting to GGUF (for Ollama, LM Studio, etc.)

### Step A: Merge the Weights
```bash
python3 merge_model.py
```

### Step B: Convert to GGUF
Using `llama.cpp`:
```bash
python3 convert-hf-to-gguf.py ../results/merged_model --outtype f16 --outfile ../results/my_finetuned_model.gguf
```

## 5. Running Inference

To test your fine-tuned model (non-GGUF version) locally:

```bash
python3 run_llm.py
```

## Important Note on "Focus Only on These Documents"

Fine-tuning helps the model learn the *style* and *vocabulary* of your documents. However, to ensure the model **only** uses information from these documents and doesn't hallucinate, it is recommended to use a **Retrieval-Augmented Generation (RAG)** approach.

The existing `streamlit_app.py` already implements a RAG system. You can use your fine-tuned model as the backend for that system.
