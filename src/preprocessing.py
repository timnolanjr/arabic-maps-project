#!/usr/bin/env python3
"""
src/preprocessing.py

Preprocess Arabic circular map scans:
  - Reads Map Layout & processed flags from Google Sheet
  - Skips only those already in the sheet
  - Filters to one_page_full or two_page_full layouts
  - Sorts remaining images by file size (ascending)
  - Detects circles (Hough) + small center-dot
  - Interactive candidate selection or largest-radius pick
      * y = keep this circle
      * n = reject, try next candidate
      * s = skip this image entirely
  - Saves annotated images
  - Optionally displays (--show)
  - Updates Google Sheet and logs to offline CSV
  - Verbose prints to track progress
"""

import argparse
import os
import time
import unicodedata

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
ALLOWED_LAYOUTS = {'one_page_full', 'two_page_full'}


def norm(s: str) -> str:
    return unicodedata.normalize('NFC', (s or '').strip())


def key_from_filename(fname: str) -> str:
    return norm(os.path.splitext(fname)[0])


# ── Load environment & offline CSV ─────────────────────────────────────
load_dotenv()
CRED_PATH   = os.environ.get('GOOGLE_SHEETS_CREDENTIALS', '')
SHEET_ID    = os.environ.get('SHEET_ID', '')
OFFLINE_CSV = os.environ.get('OFFLINE_CSV', 'map_metadata.csv')

print(f"📋 Loading local CSV log from {OFFLINE_CSV}...")
try:
    df_local = pd.read_csv(OFFLINE_CSV, dtype=str)
    df_local['File Name'] = df_local['File Name'].fillna('').map(norm)
    print(f"  → {len(df_local)} rows loaded from CSV.")
except FileNotFoundError:
    print("  → No local CSV found, starting fresh.")
    df_local = pd.DataFrame(columns=[
        'File Name','Map Layout','Circle Center X','Circle Center Y','Circle Radius','Processed At'
    ])

processed_local = {
    fn for fn, row in df_local.set_index('File Name').iterrows()
    if all(str(row.get(c,'')).strip() for c in ('Circle Center X','Circle Center Y','Circle Radius'))
}
print(f"  → {len(processed_local)} files marked processed in local CSV.\n")


# ── Connect to Google Sheets ───────────────────────────────────────────
use_google = True
print("🔗 Connecting to Google Sheets...")
try:
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds  = Credentials.from_service_account_file(CRED_PATH, scopes=SCOPES)
    gc     = gspread.authorize(creds)
    ws     = gc.open_by_key(SHEET_ID).sheet1

    hdr            = ws.row_values(1)
    FILE_NAME_COL  = hdr.index('File Name') + 1
    MAP_LAYOUT_COL = hdr.index('Map Layout') + 1
    X_COL          = hdr.index('Circle Center X') + 1
    Y_COL          = hdr.index('Circle Center Y') + 1
    R_COL          = hdr.index('Circle Radius') + 1

    print("  ✅ Connected to Google Sheets.\n")
except Exception as e:
    use_google = False
    print(f"  ⚠ Cannot connect to Google Sheets: {e}\n")


def fetch_sheet_state():
    raw_names = ws.col_values(FILE_NAME_COL)[1:]
    names      = [norm(n) for n in raw_names]
    layouts    = ws.col_values(MAP_LAYOUT_COL)[1:]
    xs         = ws.col_values(X_COL)[1:]
    layout_map    = dict(zip(names, layouts))
    processed_set = { names[i] for i, v in enumerate(xs) if str(v).strip() }
    print(f"  → Sheet has {len(layout_map)} entries, {len(processed_set)} already processed.\n")
    return layout_map, processed_set


def save_metadata(key: str, x: int, y: int, r: int, layout: str):
    ts = time.strftime('%Y-%m-%dT%H:%M:%S')
    print(f"✎ Saving metadata for '{key}': X={x}, Y={y}, R={r}, layout={layout}")
    if use_google:
        try:
            all_names = ws.col_values(FILE_NAME_COL)
            normed    = [norm(n) for n in all_names]
            row_index = normed.index(key) + 1
            ws.update_cell(row_index, X_COL, str(x))
            ws.update_cell(row_index, Y_COL, str(y))
            ws.update_cell(row_index, R_COL, str(r))
            print(f"  🗒 Updated sheet row {row_index} for '{key}'")
        except Exception as e:
            print(f"  ⚠ Sheet update failed for '{key}': {e}")

    meta = {
        'File Name':       key,
        'Map Layout':      layout,
        'Circle Center X': x,
        'Circle Center Y': y,
        'Circle Radius':   r,
        'Processed At':    ts
    }
    df = pd.read_csv(OFFLINE_CSV, dtype=str) if os.path.exists(OFFLINE_CSV) else pd.DataFrame()
    df['File Name'] = df['File Name'].fillna('').map(norm)
    df = df[df['File Name'] != key]
    df = pd.concat([df, pd.DataFrame([meta])], ignore_index=True)
    df.to_csv(OFFLINE_CSV, index=False)
    print(f"  🗒 Wrote to local CSV: {OFFLINE_CSV}\n")


def pick_one(src, cands):
    """
    Show candidates in turn.  y=keep, n=next, s=skip image.
    Returns (x,y,r), or 'skip' to abort this image, or None if no choice.
    """
    for i, (cx, cy, cr) in enumerate(cands, start=1):
        img = src.copy()
        cv.circle(img, (cx, cy), cr, (255, 0, 255), 3)
        cv.circle(img, (cx, cy),   int(cr/100), (255, 0, 255), 5)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title(f"Candidate #{i}  (y=keep, n=next, s=skip this image)")
        plt.axis('off'); plt.show()
        ans = ''
        while ans.lower() not in ('y','n','s'):
            ans = input("Keep this circle? (y/n/s): ")
        plt.close()
        if ans.lower() == 'y':
            return (cx, cy, cr)
        if ans.lower() == 's':
            return 'skip'
    return None


def detect_circle(path: str, interactive: bool):
    src = cv.imread(path, cv.IMREAD_COLOR)
    if src is None:
        print(f"✖ Cannot open {path}")
        return None
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    h, w = gray.shape
    s    = min(h, w)
    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=100, param2=10,
        minRadius=int(s*0.1), maxRadius=int(s*0.5)
    )
    if circles is None:
        return None
    cands = np.uint16(np.around(circles[0])).tolist()
    return pick_one(src, cands) if interactive else max(cands, key=lambda c: c[2])


def main():
    p = argparse.ArgumentParser(description="Detect circles in Arabic maps and update metadata.")
    p.add_argument("-i","--input-dir",  required=True, help="Directory of raw map images")
    p.add_argument("-o","--output-dir", default="Maps_circleGuess", help="Where to save annotated outputs")
    p.add_argument("--interactive",   action="store_true", help="Choose among circle candidates")
    p.add_argument("--show",          action="store_true", help="Display each annotated image")
    args = p.parse_args()

    print(f"🚀 Starting processing:\n  input-dir = {args.input_dir}\n  output-dir = {args.output_dir}\n")
    if not os.path.isdir(args.input_dir):
        print(f"✖ Input directory '{args.input_dir}' not found")
        return 1

    if use_google:
        layout_map, processed_google = fetch_sheet_state()
    else:
        layout_map, processed_google = {}, set()

    # Gather → filter by layout → sort by file-size ascending
    candidates = []
    for fn in os.listdir(args.input_dir):
        if fn.startswith('.') or fn.lower().endswith('.pdf'):
            continue
        if not fn.lower().endswith(VALID_EXTENSIONS):
            continue
        key    = key_from_filename(fn)
        layout = layout_map.get(key, '')
        if layout not in ALLOWED_LAYOUTS:
            print(f"  ↪ Skipping '{fn}' (layout='{layout}')")
            continue
        path = os.path.join(args.input_dir, fn)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = float('inf')
        candidates.append((fn, layout, size))

    candidates.sort(key=lambda x: x[2])
    print(f"🔍 {len(candidates)} images queued (filtered+sorted by size).\n")

    os.makedirs(args.output_dir, exist_ok=True)
    for fn, layout, size in tqdm(candidates, desc="Processing", unit="file"):
        key = key_from_filename(fn)
        if key in processed_google:
            print(f"\n  ↪ Skipping '{fn}' (already in sheet)\n")
            continue

        raw_path = os.path.join(args.input_dir, fn)
        print(f"\n▶ Processing: {fn}   (size={size} bytes)")
        result = detect_circle(raw_path, args.interactive)
        if result == 'skip':
            print(f"⏭ Skipped '{fn}' by user request\n")
            continue
        if not result:
            print(f"⚠ No circle found in '{fn}', skipping\n")
            continue

        x, y, r = result
        img = cv.imread(raw_path, cv.IMREAD_COLOR)
        cv.circle(img, (x, y), r, (255,0,255), 3)
        cv.circle(img, (x, y),   int(r/100), (255, 0, 255), 5)

        title = f"{fn}  (X={x}, Y={y}, R={r})"
        if args.show:
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()

        out_path = os.path.join(args.output_dir, fn)
        cv.imwrite(out_path, img)
        print(f"  ✔ Saved annotated image to {out_path}")

        save_metadata(key, x, y, r, layout)

    print("\n🏁 All done.")
    return 0


if __name__ == "__main__":
    exit(main())
