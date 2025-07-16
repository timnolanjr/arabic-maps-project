import os
import unicodedata

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
LAYOUTS_OK = {'one_page_full', 'two_page_full'}


def norm(s: str) -> str:
    return unicodedata.normalize('NFC', (s or '').strip())


def key_from(fn: str) -> str:
    return norm(os.path.splitext(fn)[0])
