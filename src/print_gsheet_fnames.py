#!/usr/bin/env python3
"""
print_gsheet_fnames.py

Connect to the configured Google Sheet and print out
the contents of the 'File Name' column (one per line).
"""

import os
import unicodedata
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

def norm(s: str) -> str:
    return unicodedata.normalize('NFC', s or '').strip()

def main():
    load_dotenv()  # expects GOOGLE_SHEETS_CREDENTIALS & SHEET_ID in .env
    creds_path = os.environ['GOOGLE_SHEETS_CREDENTIALS']
    sheet_id   = os.environ['SHEET_ID']

    # authorize
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc    = gspread.authorize(creds)
    ws    = gc.open_by_key(sheet_id).sheet1

    # find the index of the "File Name" column
    header = ws.row_values(1)
    try:
        fn_col = header.index('File Name') + 1
    except ValueError:
        print("✖ ERROR: 'File Name' not found in sheet header:", header)
        return

    # pull all file-names (skip the header row)
    raw_names = ws.col_values(fn_col)[1:]

    print("📋 File Names in Google Sheet:")
    print("------------------------------")
    for i, name in enumerate(raw_names, start=2):
        print(f"{i:>3}: {norm(name)}")

if __name__ == "__main__":
    main()
