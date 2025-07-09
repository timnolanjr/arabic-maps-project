#!/usr/bin/env python3
"""
src/test_sheets.py

Diagnostics for Google Sheets access:
  - Prints service account email
  - Attempts open_by_key and open_by_url
  - Lists accessible spreadsheets via openall()
"""

import os
import json
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# ── Load config ─────────────────────────────────────────────────────────────
load_dotenv()  # if you’re loading from .env
CRED_PATH = os.environ.get('GOOGLE_SHEETS_CREDENTIALS', '/Users/tim/Documents/Projects/arabic-maps-project/creds.json')
SHEET_ID  = os.environ.get('SHEET_ID', '1AvfZHis7JE6OLxS8wnLUjMZeAuFZgOL3Q1n0ePb5Nlo')

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# ── Authorize ────────────────────────────────────────────────────────────────
creds = Credentials.from_service_account_file(CRED_PATH, scopes=SCOPES)
print("Service account email:", creds.service_account_email)
print("Using credential file:", CRED_PATH)
print("Testing SHEET_ID:", repr(SHEET_ID), "\n")

gc = gspread.authorize(creds)
print("✅ gspread client created.\n")

# ── Test open_by_key ─────────────────────────────────────────────────────────
print("→ Trying open_by_key…")
try:
    ss = gc.open_by_key(SHEET_ID)
    print("   🎉 open_by_key succeeded! Title:", ss.title)
except Exception as e:
    print("   ❌ open_by_key error:", repr(e))
print()

# ── Test open_by_url ─────────────────────────────────────────────────────────
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"
print("→ Trying open_by_url…")
try:
    ss2 = gc.open_by_url(url)
    print("   🎉 open_by_url succeeded! Title:", ss2.title)
except Exception as e:
    print("   ❌ open_by_url error:", repr(e))
print()

# ── List via openall ─────────────────────────────────────────────────────────
print("→ Listing with openall():")
sheets = gc.openall()
print(f"   Number of spreadsheets visible: {len(sheets)}")
for idx, sheet in enumerate(sheets[:20], start=1):
    print(f"   {idx}. {sheet.id} → {sheet.title}")

