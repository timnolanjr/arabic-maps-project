#!/usr/bin/env python3
"""
find_missing_raw.py

Fetches File Name+File Type from your Google Sheet,
then reports which of those files are NOT present
in your local Raw_Maps directory.
"""

import os
import unicodedata

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# ── CONFIG ────────────────────────────────────────────────────────────────
RAW_DIR = '/Users/tim/Documents/Projects/arabic-maps-project/data/Raw_Maps'
# Make sure GOOGLE_SHEETS_CREDENTIALS & SHEET_ID are set in your .env
# ── end CONFIG ────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    return unicodedata.normalize('NFC', (s or '').strip())

def fetch_expected():
    load_dotenv()
    creds_path = os.environ['GOOGLE_SHEETS_CREDENTIALS']
    sheet_id   = os.environ['SHEET_ID']
    scopes     = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
    creds      = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc         = gspread.authorize(creds)
    ws         = gc.open_by_key(sheet_id).sheet1

    hdr        = ws.row_values(1)
    fn_col     = hdr.index('File Name') + 1
    ft_col     = hdr.index('File Type') + 1

    names = ws.col_values(fn_col)[1:]
    exts  = ws.col_values(ft_col)[1:]
    expected = set()
    for n, e in zip(names, exts):
        n2 = norm(n)
        e2 = norm(e).lower()
        if n2 and e2:
            expected.add(f"{n2}{e2}")
    return expected

def fetch_actual():
    actual = set()
    for fn in os.listdir(RAW_DIR):
        if fn.startswith('.'):
            continue
        path = os.path.join(RAW_DIR, fn)
        if os.path.isfile(path):
            actual.add(norm(fn))
    return actual

def main():
    expected = fetch_expected()
    actual   = fetch_actual()

    missing = sorted(expected - actual, key=lambda s: s.casefold())
    if not missing:
        print("✅ All expected files are present in Raw_Maps.")
    else:
        print("📂 Missing files in Raw_Maps:")
        for fn in missing:
            print("  -", fn)

if __name__ == "__main__":
    main()
