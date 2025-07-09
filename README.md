
Arabic Maps Project  
===================

A toolkit for ingesting, processing, and annotating digitized circular maps from Arabic manuscripts. This repository provides:  
- Automated metadata extraction (file size, dimensions, color space)  
- Google Sheets integration for master metadata 
- Circle detection (via Hough Transform) with interactive review  
- Offline CSV for redundancy and batch reporting  
- Utilities to find missing files, extract embedded PDF images, and fill in blank sheet entries  

Features  (As of 8 July 2025)
--------  
**Raw_Maps_find_missing.py**  
Identify which files from the Google Spreadsheet are missing locally.

**fill_sheet_metadata.py**  
Scan your a Maps directory, read file properties, and backfill empty “File Type”, “File Size”, “Dimensions (COLxROW)”, and “Color Space” cells in the Google Spreadsheet.

**preprocessing.py**  
  - Reads “Map Layout” from the Google Sheet
  - Currently filters to full-circle layouts (one_page_full, two_page_full)
  - Runs Hough circle detection (interactive or automatic) to find the map's outer perimeter. As a reminder, we are aiming for the outer edge of [al-Baḥr al-Muḥīṭ, *The Encircling Ocean/Sea*](https://referenceworks.brill.com/display/entries/EIEO/SIM-1064.xml?rskey=arnJ0m&result=1).
  - Annotates and saves results, updates “Circle Center X” (pixels), "Circle Center Y" (pixels), and "Circle Radius" (pixels) in the Google Sheet. 
  - Logs everything back to both Google Sheets and an offline CSV

**extract_pdf_images.py**  
Extracts embedded images from any PDFs in your raw folder.

Getting Started  
---------------  
1. Clone the repo  
```

git clone [https://github.com/your-org/arabic-maps-project.git](https://github.com/your-org/arabic-maps-project.git)
cd arabic-maps-project

```

2. Install dependencies  
```

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

3. Configure your environment  
Create a `.env` in the project root with:  
```

GOOGLE\_SHEETS\_CREDENTIALS=path/to/creds.json
SHEET\_ID=<your-sheet-id>
OFFLINE\_CSV=data/map\_metadata.csv

````

4. Populate your sheet  
Ensure your Google Sheet has columns matching the header row in `data/map_metadata.csv`.

5. Run the tools  
- Find missing raw files:  
  ```
  python src/Raw_Maps_find_missing.py
  ```  
- Backfill blank metadata cells:  
  ```
  python src/fill_sheet_metadata.py
  ```  
- Detect circles and annotate:  
  ```
  python src/preprocessing.py \
    -i data/Raw_Maps \
    -o data/Perfect_Maps_Processed \
    [--interactive] [--show]
  ```

Directory Layout  
----------------  
````

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
