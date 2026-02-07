import os
import fitz  # PyMuPDF
import json
from tqdm import tqdm

def extract_text_from_pdfs(pdf_folder, output_file):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    all_text = []

    for pdf_file in tqdm(pdf_files, desc="Extracting text from PDFs"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()

            if text.strip():
                all_text.append({
                    "source": pdf_file,
                    "text": text
                })
            doc.close()
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_text:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Extracted text from {len(all_text)} PDFs and saved to {output_file}")

if __name__ == "__main__":
    extract_text_from_pdfs(".", "training_data.jsonl")
