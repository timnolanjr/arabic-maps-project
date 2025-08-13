from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np

def boxes_to_mask(boxes: List[Tuple[int,int,int,int]], shape: Tuple[int,int]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for x, y, w, h in boxes:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask

def morph_merge_mask(mask: np.ndarray, close_ksize: int, open_ksize: int, iterations: int) -> np.ndarray:
    out = mask.copy()
    if close_ksize and close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=iterations)
    if open_ksize and open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=iterations)
    return out

def mask_to_boxes(mask: np.ndarray) -> List[Tuple[int,int,int,int]]:
    num, labels = cv2.connectedComponents(mask > 0, connectivity=8)
    boxes: List[Tuple[int,int,int,int]] = []
    for lab in range(1, num):
        ys, xs = np.where(labels == lab)
        if ys.size == 0: continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append((int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)))
    return boxes