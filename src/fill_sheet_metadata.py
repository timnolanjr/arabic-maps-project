#!/usr/bin/env python3
"""
fill_sheet_metadata.py

Find any blank 'File Size', 'Dimensions (COLxROW)' or 'Color Space' cells in your Google Sheet,
look up the matching image under RAW_DIR, and fill them in.
"""

import os
import unicodedata

from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

RAW_DIR = '/Users/tim/Documents/Projects/arabic-maps-project/data/Raw_Maps'


def norm(s: str) -> str:
    """Unicode‐normalize to NFC, strip, and lowercase."""
    return unicodedata.normalize('NFC', (s or '').strip()).casefold()


def load_sheet():
    load_dotenv()
    creds = Credentials.from_service_account_file(
        os.environ['GOOGLE_SHEETS_CREDENTIALS'],
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    ws = gspread.authorize(creds).open_by_key(os.environ['SHEET_ID']).sheet1
    header = ws.row_values(1)
    return ws, header


def build_disk_index():
    """
    Return a dict mapping normalized 'base+ext' → actual filename,
    so we can do a single lookup per sheet row.
    """
    idx = {}
    for fn in os.listdir(RAW_DIR):
        if fn.startswith('.'):
            continue
        path = os.path.join(RAW_DIR, fn)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(fn)
        key = norm(base) + norm(ext)
        idx[key] = fn
    return idx


def main():
    ws, header = load_sheet()
    # find columns (1-based)
    c_fname  = header.index('File Name') + 1
    c_ftype  = header.index('File Type') + 1
    c_size   = header.index('File Size') + 1
    c_dims   = header.index('Dimensions (COLxROW)') + 1
    c_cspace = header.index('Color Space') + 1

    disk = build_disk_index()
    all_rows = ws.get_all_values()

    updates = []
    for row_idx, row in enumerate(all_rows[1:], start=2):
        raw_base = row[c_fname-1]
        raw_ext  = row[c_ftype-1]
        if not raw_base or not raw_ext:
            continue

        # only fill truly blank sheet cells
        need_size   = not row[c_size-1].strip()
        need_dims   = not row[c_dims-1].strip()
        need_cspace = not row[c_cspace-1].strip()
        if not (need_size or need_dims or need_cspace):
            continue

        key = norm(raw_base) + norm(raw_ext)
        if key not in disk:
            print(f"⚠ couldn't find on disk: {raw_base}{raw_ext}")
            continue

        fn = disk[key]
        path = os.path.join(RAW_DIR, fn)

        # prepare replacements
        vals = {}
        if need_size:
            vals[c_size] = str(os.path.getsize(path))
        if need_dims or need_cspace:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    mode = img.mode
            except Exception as e:
                print(f"✖ cannot open {fn}: {e}")
                continue
            if need_dims:
                vals[c_dims] = f"{w}x{h}"
            if need_cspace:
                # assume grayscale if mode 'L' or 'LA'
                vals[c_cspace] = 'Gray' if mode in ('L','LA') else 'sRGB'

        for col, val in vals.items():
            cell = gspread.utils.rowcol_to_a1(row_idx, col)
            updates.append({"range": cell, "values": [[val]]})
            print(f"  ☐ will set {cell} ({header[col-1]}) → {val}")

    if not updates:
        print("✅ all blanks already filled.")
        return

    batch_body = {"valueInputOption": "RAW", "data": updates}
    print(f"\n🚀 applying batch update of {len(updates)} cells…")
    try:
        ws.batch_update(batch_body)
        print("✅ batch_update succeeded.")
    except Exception as e:
        print(f"❗ batch_update failed: {e}\n🔄 falling back to single‐cell updates…")
        for u in updates:
            rng = u["range"]
            val = u["values"][0][0]
            try:
                ws.update(rng, [[val]], value_input_option="RAW")
                print(f"  ✓ {rng} = {val}")
            except Exception as e2:
                print(f"  ✖ {rng} failed: {e2}")

if __name__ == "__main__":
    main()
