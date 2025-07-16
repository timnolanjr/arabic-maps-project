from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_SHEETS_CREDENTIALS = os.environ['GOOGLE_SHEETS_CREDENTIALS']
SHEET_ID = os.environ['SHEET_ID']
OFFLINE_CSV = os.environ.get('OFFLINE_CSV', 'map_metadata.csv')
GOOGLE_DRIVE_FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', '')
