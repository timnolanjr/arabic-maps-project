# Arabic Maps Project

A digital humanities toolkit for analyzing circular world maps in Arabic manuscripts.

This project enables automated detection, transcription, and spatial comparison of circular manuscript maps—especially those representing *al-Baḥr al-Muḥīṭ* (The Encircling Ocean). It brings together computer vision, OCR, and metadata integration to support large-scale analysis of cartographic variation, copyist influence, and map reuse across manuscripts.

## Features

| Module | Description |
|--------|-------------|
| Circle Detection | Hough Transform with interactive point-click refinement to identify circular map frames |
| Arabic OCR | Full-page optical character recognition using QARI-OCR and EasyOCR |
| Toponym Projection | Extracts Arabic place names and maps their centroids onto polar coordinates |
| Map Comparison | Enables quantitative analysis of variation across maps in terms of structure, labeling, and reuse |
| Metadata Integration | Syncs with Google Sheets for annotation, progress tracking, and redundancy via local CSV |
| Utility Scripts | Includes tools to find missing files, extract embedded images from PDFs, and fill in blank spreadsheet entries |

## Pipeline Overview

```
               Raw Map Scans (TIFF, JPG)
                        │
                        ▼
             Circle Detection (Hough)
                        │
                        ▼
           Arabic OCR (EasyOCR/Strabo/Qari)
                        │
                        ▼
              Toponym Extraction
                        │
                        ▼
          Polar Coordinate Projection
                        │
                        ▼
       Map Comparison & Quantitative Analysis
```

## Directory Layout

```
.
├── data/
│   ├── Raw_Maps/               # Raw scanned maps
│   └── map_metadata.csv        # Local mirror of Google Sheet metadata
├── src/
│   ├── preprocessing.py        # Circle detection + metadata sync
│   ├── qari_ocr.py             # Full-page Arabic OCR using Qwen2VL
│   ├── easyocr_detect.py       # EasyOCR CRAFT-based region detector
│   ├── strabo_detect.py        # Wrapper for Strabo or Tesseract OCR
│   ├── fill_sheet_metadata.py  # Fill missing metadata in sheet
│   ├── Raw_Maps_find_missing.py# Sanity check for missing files
│   ├── extract_pdf_images.py   # Extract embedded images from PDFs
│   └── test_sheets.py          # Diagnostics for Google Sheets API
├── requirements.txt
├── .env                        # Your credentials and config
└── README.md
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-org/arabic-maps-project.git
cd arabic-maps-project
```

### 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure your environment

Create a `.env` file in the project root with:

```
GOOGLE_SHEETS_CREDENTIALS=path/to/creds.json
SHEET_ID=your-google-sheet-id
OFFLINE_CSV=data/map_metadata.csv
GOOGLE_DRIVE_FOLDER_ID=your-drive-folder-id  # optional
```

### 4. Prepare your Google Sheet

Ensure it has columns including:

- File Name
- File Type
- File Size
- Dimensions (COLxROW)
- Color Space
- Map Layout
- Circle Center X, Circle Center Y, Circle Radius

## Running the Tools

### Find missing files

```bash
python src/Raw_Maps_find_missing.py
```

### Fill blank metadata from disk

```bash
python src/fill_sheet_metadata.py
```

### Extract images from PDFs

```bash
python src/extract_pdf_images.py data/Raw_Maps
```

### Detect map circle boundaries

```bash
python src/preprocessing.py \
  -i data/Raw_Maps \
  -o data/Processed_Maps \
  --interactive
```

### Run full-page OCR with QARI

```bash
python src/qari_ocr.py data/Raw_Maps --device cuda
```

## Background

Circular maps abound in Arabic manuscripts, but few studies focus on the maps themselves rather than the geographers who authored them. This project treats the map as a living manuscript object—copied, varied, and adapted. By automating detection, transcription, and coordinate mapping, we reveal copyist behavior, inter-manuscript relationships, and patterns of transmission invisible to traditional cataloging.

## License

MIT License. See LICENSE file for details.

.
├── data/
│   ├── Raw\_Maps/                  # Raw scans (jpg, png, tiff, pdf…)
│   └── map\_metadata.csv           # Offline CSV mirror of Google Sheet
├── src/
│   ├── Raw\_Maps\_find\_missing.py
│   ├── fill\_sheet\_metadata.py
│   ├── preprocessing.py
│   └── extract\_pdf\_images.py     # Utility for PDF image extraction
├── .env                            # Credentials & config
├── requirements.txt
└── README.md

```

License  
-------  
MIT License. See LICENSE file for full text.  
