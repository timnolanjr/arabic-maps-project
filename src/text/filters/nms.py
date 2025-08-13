from __future__ import annotations

from typing import List, Tuple
import numpy as np

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    a_area = max(0.0, aw * ah); b_area = max(0.0, bw * bh)
    union = max(1e-6, a_area + b_area - inter_area)
    return float(inter_area / union)

def nms_filter(
    boxes: np.ndarray,   # (N,4) [x,y,w,h]
    scores: np.ndarray,  # (N,)
    iou_thresh: float = 0.3,
) -> Tuple[List[int], List[int]]:
    N = int(boxes.shape[0]) if boxes is not None else 0
    if N == 0:
        return [], []
    boxes = boxes.astype(float, copy=False)
    scores = scores.astype(float, copy=False)
    order = np.argsort(-scores)  # descending
    keep: List[int] = []; suppressed: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.empty(rest.shape[0], dtype=np.float32)
        for k, j in enumerate(rest):
            ious[k] = _iou(boxes[i], boxes[int(j)])
        to_suppress = rest[ious > iou_thresh]
        suppressed.extend(map(int, to_suppress))
        order = rest[ious <= iou_thresh]
    return keep, suppressed