import os
import time
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from . import config

creds = Credentials.from_service_account_file(
    config.GOOGLE_SHEETS_CREDENTIALS,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)

gc = gspread.authorize(creds)
ws = gc.open_by_key(config.SHEET_ID).sheet1
hdr = ws.row_values(1)
COL_FN = hdr.index('File Name') + 1
COL_LAYOUT = hdr.index('Map Layout') + 1
COL_X = hdr.index('Circle Center X') + 1
COL_Y = hdr.index('Circle Center Y') + 1
COL_R = hdr.index('Circle Radius') + 1

# Google Drive
_drive = build('drive', 'v3', credentials=creds)


def fetch_sheet():
    names = [n.strip() for n in ws.col_values(COL_FN)[1:]]
    layouts = ws.col_values(COL_LAYOUT)[1:]
    xs = ws.col_values(COL_X)[1:]
    layout_map = dict(zip(names, layouts))
    done = {names[i] for i, v in enumerate(xs) if v.strip()}
    return layout_map, done


def save_metadata(key: str, x: int, y: int, r: int, layout: str) -> None:
    ts = time.strftime('%Y-%m-%dT%H:%M:%S')
    names = [n.strip() for n in ws.col_values(COL_FN)]
    row = names.index(key) + 1
    ws.update_cell(row, COL_X, str(x))
    ws.update_cell(row, COL_Y, str(y))
    ws.update_cell(row, COL_R, str(r))

    rec = {
        'File Name': key,
        'Map Layout': layout,
        'Circle Center X': x,
        'Circle Center Y': y,
        'Circle Radius': r,
        'Processed At': ts,
    }
    df = pd.read_csv(config.OFFLINE_CSV, dtype=str) if os.path.exists(config.OFFLINE_CSV) else pd.DataFrame()
    df['File Name'] = df['File Name'].fillna('').map(str.strip)
    df = df[df['File Name'] != key]
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df.to_csv(config.OFFLINE_CSV, index=False)


def upload_to_drive(local: str, name: str) -> None:
    if not config.GOOGLE_DRIVE_FOLDER_ID:
        return
    media = MediaFileUpload(local, mimetype='image/jpeg', resumable=True, chunksize=10 * 1024 * 1024)
    req = _drive.files().create(
        body={'name': name, 'parents': [config.GOOGLE_DRIVE_FOLDER_ID]},
        media_body=media,
        fields='id',
    )
    while True:
        status, resp = req.next_chunk()
        if resp is not None:
            break
        if status:
            pass
    # No return value; Google Drive ID not used

