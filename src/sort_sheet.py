#!/usr/bin/env python3
"""
sort_sheet.py

Fetch all rows from a Google Sheet, sort by the 'File Name' column,
and overwrite the sheet with the sorted data.
"""

import os
import unicodedata
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

def norm(s: str) -> str:
    return unicodedata.normalize('NFC', s or '').strip().lower()

def main():
    load_dotenv()  # expects GOOGLE_SHEETS_CREDENTIALS & SHEET_ID in .env
    creds_path = os.environ['GOOGLE_SHEETS_CREDENTIALS']
    sheet_id   = os.environ['SHEET_ID']

    # authorize
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc    = gspread.authorize(creds)
    ws    = gc.open_by_key(sheet_id).sheet1

    # fetch all data
    all_values = ws.get_all_values()
    if not all_values:
        print("⚠ Sheet is empty.")
        return

    header = all_values[0]
    data   = all_values[1:]

    try:
        idx = header.index('File Name')
    except ValueError:
        print("✖ 'File Name' column not found in header:", header)
        return

    # sort rows by normalized File Name
    data_sorted = sorted(data, key=lambda row: norm(row[idx]))

    # overwrite sheet: header + sorted data
    new_rows = [header] + data_sorted
    ws.clear()
    ws.update('A1', new_rows, value_input_option='RAW')

    print(f"✅ Sheet sorted by 'File Name' ({len(data_sorted)} rows).")

if __name__ == '__main__':
    main()
