#!/usr/bin/env python3
"""
extract_pdf_images.py

Walks a given directory, finds all .pdf files, and
extracts each embedded image into separate PNG files
named <pdf_basename>_1.png, <pdf_basename>_2.png, …
"""

import os
import fitz  # PyMuPDF
import argparse

def extract_images_from_pdf(pdf_path):
    """
    Extracts images from a single PDF, saving them as PNGs.
    Names them <basename>_1.png, <basename>_2.png, ...
    Returns the number of images extracted.
    """
    doc = fitz.open(pdf_path)
    base, _ = os.path.splitext(pdf_path)
    counter = 1

    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            # convert CMYK to RGB
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            out_path = f"{base}_{counter}.png"
            pix.save(out_path)
            pix = None
            counter += 1

    return counter - 1

def main():
    parser = argparse.ArgumentParser(
        description="Extract embedded images from all PDFs in a directory."
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing .pdf files"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"✖ Error: '{args.folder}' is not a directory")
        return 1

    total = 0
    for entry in sorted(os.listdir(args.folder)):
        if not entry.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(args.folder, entry)
        print(f"▶ Processing {entry}...")
        num = extract_images_from_pdf(pdf_path)
        print(f"  → Extracted {num} image(s)\n")
        total += num

    print(f"✅ Done. Total images extracted: {total}")
    return 0

if __name__ == "__main__":
    exit(main())
