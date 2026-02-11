import os
import fitz  # PyMuPDF
import json
from tqdm import tqdm
import sys

def extract_text_from_pdfs(pdf_folder, output_file):
    # Normalize path
    pdf_folder = os.path.abspath(pdf_folder)
    print(f"🔍 Searching for PDF files in: {pdf_folder}")

    if not os.path.exists(pdf_folder):
        print(f"❌ Error: Folder '{pdf_folder}' does not exist.")
        return

    all_files = os.listdir(pdf_folder)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDF files found in {pdf_folder}")
        print(f"Total files in folder: {len(all_files)}")
        if all_files:
            print("First few files found:")
            for f in all_files[:5]:
                print(f" - {f}")

        # Fallback: check script directory if different from CWD
        script_dir = os.path.abspath(os.path.dirname(__file__))
        if pdf_folder != script_dir:
            print(f"💡 Trying script directory instead: {script_dir}")
            return extract_text_from_pdfs(script_dir, output_file)
        return

    print(f"📄 Found {len(pdf_files)} PDF files. Starting extraction...")
    all_text_entries = []

    for pdf_file in tqdm(pdf_files, desc="Extracting text"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            doc = fitz.open(pdf_path)
            # Extract each page as a separate training entry to avoid truncation
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                if text:
                    all_text_entries.append({
                        "source": f"{pdf_file}_page_{page_num+1}",
                        "text": text
                    })
            doc.close()
        except Exception as e:
            print(f"\n❌ Error processing {pdf_file}: {e}")

    if all_text_entries:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in all_text_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ Successfully extracted {len(all_text_entries)} pages from {len(pdf_files)} PDFs and saved to {output_file}")
    else:
        print("⚠️ No text could be extracted from the found PDFs.")

if __name__ == "__main__":
    # Default to current working directory
    target_dir = "."
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]

    extract_text_from_pdfs(target_dir, "training_data.jsonl")
