# scripts/extract_pdfs.py
from pathlib import Path
from src.utils.extract_pdf import extract_images_from_pdf

input_dir = Path("data/pdfs")
output_base = Path("data/extracted_images")

for pdf_path in sorted(input_dir.glob("*.pdf")):
    if pdf_path.name.startswith("."):
        continue  # skip hidden files
    out_dir = output_base / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    images = extract_images_from_pdf(pdf_path, out_dir)
    print(f"{pdf_path} -> {len(images)} images")
