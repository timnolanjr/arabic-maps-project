#!/usr/bin/env python3
"""
src/preprocessing.py

Preprocess Arabic circular map scans:
  - Reads Map Layout & processed flags from Google Sheet
  - Filters to one_page_full or two_page_full layouts
  - Sorts images by file size (ascending)
  - Interactive candidate selection:
      * click 8 edge points in a Matplotlib window
      * fit a best‐fit circle through those 8 points
      * run HoughCircles in a tight ROI around that circle
      * score each candidate by “mean radial error” to your 8 clicks
      * batch‐export 100 candidate previews for you to inspect and pick by index
  - Non‐interactive mode: full‐image HoughCircles pick largest radius
  - Saves annotated images, uploads to Google Drive
  - Optionally displays (--show)
  - Updates Google Sheet and logs to offline CSV
  - Prints verbose progress and elapsed times
"""

import argparse, os, time, unicodedata, math, tempfile, shutil, subprocess, sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv

VALID_EXTS = ('.jpg','.jpeg','.png','.tif','.tiff')
LAYOUTS_OK = {'one_page_full','two_page_full'}

def norm(s): return unicodedata.normalize('NFC', (s or '').strip())
def key_from(fn): return norm(os.path.splitext(fn)[0])

load_dotenv()
CRED      = os.environ['GOOGLE_SHEETS_CREDENTIALS']
SHEET_ID  = os.environ['SHEET_ID']
CSV_FILE  = os.environ.get('OFFLINE_CSV','map_metadata.csv')
DRIVE_FLD = os.environ.get('GOOGLE_DRIVE_FOLDER_ID','')

print(f"[1/7] Loading local log: {CSV_FILE}")
try:
    df_local = pd.read_csv(CSV_FILE, dtype=str).fillna('')
    df_local['File Name'] = df_local['File Name'].map(norm)
    print(f"    → {len(df_local)} rows loaded from CSV")
except FileNotFoundError:
    df_local = pd.DataFrame(columns=[
        'File Name','Map Layout','Circle Center X',
        'Circle Center Y','Circle Radius','Processed At'
    ])
processed_local = {
    fn for fn,row in df_local.set_index('File Name').iterrows()
    if all(row[c].strip() for c in ('Circle Center X','Circle Center Y','Circle Radius'))
}
print(f"    → {len(processed_local)} fully processed locally\n")

print("[2/7] Connecting to Google Sheets & Drive...")
creds = Credentials.from_service_account_file(CRED, scopes=[
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
])
# Sheets
gc = gspread.authorize(creds)
ws = gc.open_by_key(SHEET_ID).sheet1
hdr = ws.row_values(1)
COL_FN, COL_LAYOUT, COL_X, COL_Y, COL_R = (
    hdr.index('File Name')+1,
    hdr.index('Map Layout')+1,
    hdr.index('Circle Center X')+1,
    hdr.index('Circle Center Y')+1,
    hdr.index('Circle Radius')+1,
)
# Drive
drive = build('drive','v3',credentials=creds)
print("    → Connected to Google APIs\n")

def fetch_sheet():
    names   = [norm(n) for n in ws.col_values(COL_FN)[1:]]
    layouts = ws.col_values(COL_LAYOUT)[1:]
    xs      = ws.col_values(COL_X)[1:]
    layout_map = dict(zip(names,layouts))
    done = {names[i] for i,v in enumerate(xs) if v.strip()}
    print(f"[3/7] Sheet: {len(names)} entries, {len(done)} already have circle data\n")
    return layout_map, done

def save_metadata(key,x,y,r,layout):
    ts = time.strftime('%Y-%m-%dT%H:%M:%S')
    print(f"[6/7] Writing metadata for '{key}' → X={x}, Y={y}, R={r}")
    # sheet
    names = [norm(n) for n in ws.col_values(COL_FN)]
    row   = names.index(key)+1
    ws.update_cell(row,COL_X,str(x))
    ws.update_cell(row,COL_Y,str(y))
    ws.update_cell(row,COL_R,str(r))
    # csv
    rec = {
        'File Name':key,'Map Layout':layout,
        'Circle Center X':x,'Circle Center Y':y,
        'Circle Radius':r,'Processed At':ts
    }
    df = pd.read_csv(CSV_FILE,dtype=str) if os.path.exists(CSV_FILE) else pd.DataFrame()
    df['File Name'] = df['File Name'].fillna('').map(norm)
    df = df[df['File Name']!=key]
    df = pd.concat([df,pd.DataFrame([rec])],ignore_index=True)
    df.to_csv(CSV_FILE,index=False)
    print("    → Metadata saved\n")

def upload_to_drive(local, name):
    if not DRIVE_FLD: return
    print(f"[7/7] Uploading {name} to Google Drive folder {DRIVE_FLD}...")
    media = MediaFileUpload(local, mimetype='image/jpeg', resumable=True, chunksize=10*1024*1024)
    req = drive.files().create(
        body={'name':name,'parents':[DRIVE_FLD]},
        media_body=media,
        fields='id'
    )
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"    → {int(status.progress()*100)}% uploaded")
    print("    → Upload complete\n")

def fit_circle(xs,ys):
    A = np.column_stack([2*xs,2*ys,np.ones_like(xs)])
    b = xs*xs+ys*ys
    c,_,_,_ = np.linalg.lstsq(A,b,rcond=None)
    cx,cy = c[0],c[1]
    r = math.sqrt(c[2]+cx*cx+cy*cy)
    return cx,cy,r

def click_points(img, n=8):
    print("[4/7] Opening point-selection window...")
    fig,ax = plt.subplots()
    ax.imshow(img); ax.set_title(f"Click {n} edge points"); plt.axis('off')
    pts = plt.ginput(n, timeout=-1)
    plt.close(fig)
    # ensure it's fully closed
    plt.pause(0.1)
    print(f"    → {len(pts)} points collected\n")
    return pts

def detect_circle_roi(gray,src):
    pts = click_points(cv.cvtColor(gray,cv.COLOR_GRAY2RGB),8)
    if len(pts)<3:
        print("    • insufficient clicks; skipping ROI detection\n")
        return []
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    cx0,cy0,r0 = fit_circle(xs,ys)
    print(f"    • fitted center=({cx0:.1f},{cy0:.1f}), r≈{r0:.1f}")
    print("[5/7] Computing HoughCircles in ROI…")
    m = int(r0*1.2)
    x0,y0 = max(0,int(cx0-m)), max(0,int(cy0-m))
    x1,y1 = min(gray.shape[1],int(cx0+m)), min(gray.shape[0],int(cy0+m))
    roi = gray[y0:y1,x0:x1]
    cir = cv.HoughCircles(
        roi, cv.HOUGH_GRADIENT, dp=1, minDist=2,
        param1=100, param2=10,
        minRadius=int(r0*0.85), maxRadius=int(r0*1.15)
    )
    if cir is None:
        print("    • no Hough candidates found\n")
        return []
    raw = np.uint16(np.around(cir[0]))
    cands = [(c[0]+x0, c[1]+y0, c[2]) for c in raw]
    def radial_err(c):
        cx,cy,cr = c
        return np.mean([abs(math.hypot(xi-cx,yi-cy)-cr) for xi,yi in pts])
    sorted_cands = sorted(cands, key=radial_err)
    print(f"    • {len(sorted_cands)} candidates ranked by radial error\n")
    return sorted_cands

def pick_batch(src,cands,batch=100):
    tmp = tempfile.mkdtemp(prefix="hough_preview_")
    try:
        tot = len(cands)
        for s in range(0,tot,batch):
            e = min(s+batch,tot)
            for i,(cx,cy,cr) in enumerate(cands[s:e], s+1):
                im = src.copy()
                cv.circle(im,(int(cx),int(cy)),int(cr),(255,0,255),3)
                cv.circle(im,(int(cx),int(cy)),max(3,int(cr*0.01)),(255,0,255),3)
                cv.imwrite(f"{tmp}/cand_{i:04d}.jpg",im)
            print(f"    • Preview {s+1}-{e} saved to {tmp}")
            subprocess.run(['open',tmp] if sys.platform=='darwin' else ['xdg-open',tmp])
            choice = input(f"    Pick 1–{tot}, 'n' next batch, 's' skip: ")
            if choice.lower()=='n': continue
            if choice.lower()=='s': return 'skip'
            if choice.isdigit() and 1<=int(choice)<=tot:
                return cands[int(choice)-1]
    finally:
        shutil.rmtree(tmp)
    return None

def detect_circle(path,interactive):
    src = cv.imread(path)
    if src is None:
        print(f"cannot open {path}\n"); return None
    gray = cv.medianBlur(cv.cvtColor(src,cv.COLOR_BGR2GRAY),5)
    if interactive:
        cands = detect_circle_roi(gray,src)
        if not cands: return None
        return pick_batch(src,cands)
    else:
        cir = cv.HoughCircles(
            gray, cv.HOUGH_GRADIENT, dp=1, minDist=2,
            param1=100, param2=10,
            minRadius=int(min(gray.shape)*0.1),
            maxRadius=int(min(gray.shape)*0.5)
        )
        if cir is None:
            print("    • No Hough detected; skipping\n")
            return None
        raw = np.uint16(np.around(cir[0]))
        best = max(raw.tolist(), key=lambda c: c[2])
        return tuple(best)

def human_size(b):
    for u in ['B','KB','MB','GB']:
        if b < 1024: return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}TB"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-dir', required=True)
    parser.add_argument('-o','--output-dir', default='Maps_circleGuess')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    layout_map, done = fetch_sheet()
    files = [f for f in os.listdir(args.input_dir)
             if f.lower().endswith(VALID_EXTS)]
    files.sort(key=lambda f: os.path.getsize(os.path.join(args.input_dir,f)))
    print(f"[   ] {len(files)} files queued (filtered + sorted by size)\n")

    os.makedirs(args.output_dir, exist_ok=True)
    for fn in tqdm(files, desc="Files"):
        key = key_from(fn)
        if layout_map.get(key,'') not in LAYOUTS_OK:
            print(f" skipping {fn}: layout != allowed")
            continue
        if key in done:
            ans = input(f"{fn} already done—skip? (y/n): ")
            if ans.lower().startswith('y'):
                print(" skipped\n")
                continue

        path = os.path.join(args.input_dir, fn)
        sz = os.path.getsize(path)
        print(f"\n→ Processing {fn} ({human_size(sz)})")
        t0 = time.time()

        res = detect_circle(path, args.interactive)
        if not res or res=='skip':
            print(" skipped\n")
            continue

        x,y,r = res
        src = cv.imread(path)
        cv.circle(src,(int(x),int(y)),int(r),(255,0,255),3)
        cv.circle(src,(int(x),int(y)),max(3,int(r*0.01)),(255,0,255),3)
        out = os.path.join(args.output_dir, fn)
        cv.imwrite(out, src)
        upload_to_drive(out, fn)
        if args.show:
            plt.imshow(cv.cvtColor(src,cv.COLOR_BGR2RGB))
            plt.title(fn); plt.axis('off'); plt.show()

        save_metadata(key, x, y, r, layout_map[key])
        dt = time.time() - t0
        print(f"Elapsed {int(dt//60)}m{dt%60:.1f}s\n")

if __name__=='__main__':
    main()
