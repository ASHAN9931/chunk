# LLM Training & Navigation Guide

This guide provides the full "navigation" for training an LLM to focus **only** on your documents. Since you are using an integrated Intel GPU, you will need to use **Google Colab** (a free cloud-based GPU service) for the training step.

## 🚀 Navigation Overview
1.  **Locally:** Prepare your data (`extract_text.py`).
2.  **In the Cloud (Google Colab):** Train your model (`train_llm.py`).
3.  **Locally/Cloud:** Merge and convert to GGUF (`merge_model.py`).

---

## ⚠️ Hardware Check (Local)
As you've seen, local training requires an NVIDIA GPU. Your **Intel UHD Graphics** are great for display but cannot perform the billions of calculations needed for LLM training.

**Solution:** Follow the Google Colab steps below.

---

## 1. Data Preparation (Local)

Run this on your computer where the PDFs are located:
```bash
python extract_text.py
```
This will create a file called **`training_data.jsonl`**. This is the only file you need to upload for training.

---

## 2. Google Colab Training Walkthrough (Cloud)

This is how you train without a local NVIDIA GPU:

1.  **Go to:** [colab.research.google.com](https://colab.research.google.com).
2.  **Create a New Notebook.**
3.  **Change Runtime Type:**
    - Click `Runtime` -> `Change runtime type`.
    - Select **T4 GPU** (or any available GPU).
4.  **Upload Files:**
    - Click the Folder icon 📁 on the left sidebar.
    - Upload `train_llm.py` and your `training_data.jsonl`.
5.  **Install Dependencies:**
    - In a code cell, run:
      ```bash
      !pip install torch transformers datasets peft bitsandbytes tqdm accelerate trl
      ```
6.  **Start Training:**
    - In a new cell, run:
      ```bash
      !python train_llm.py
      ```
7.  **Download Result:**
    - Once finished, download the `results/final_checkpoint` folder.

---

## 3. How to "Only Focus" on Your Documents

To ensure the LLM stays strictly within the context of your legal and environmental documents, you should use a **System Prompt** during inference.

In `run_llm.py` or your RAG application, set the system prompt to:

> "You are a specialized assistant for Sri Lankan Agrarian and Environmental law. You must ONLY answer questions based on the provided documents. If the answer is not in the documents, state that you do not know. Do not use any outside knowledge."

Fine-tuning teaches the model the *language* of your documents, but the **System Prompt** provides the *rules* for focusing.

---

## 4. Exporting to GGUF

Once you have your trained model:

1.  **Merge weights:** Run `python merge_model.py` to combine the adapters.
2.  **Convert:** Use `llama.cpp`'s `convert-hf-to-gguf.py` as described in previous sections.

---

## 💡 Summary of Files
- `extract_text.py`: Run this first to get your data ready.
- `train_llm.py`: Run this on Google Colab to train the AI.
- `merge_model.py`: Run this after training to prepare for GGUF export.
- `run_llm.py`: Run this to test your model.
- `training_data.jsonl`: Your actual data (keep this private).
