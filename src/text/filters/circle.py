from __future__ import annotations

from typing import List, Tuple
import numpy as np

def circle_coverage_filter(
    boxes: List[Tuple[int, int, int, int]],
    *,
    center: Tuple[int, int],
    radius: int,
    samples: int = 5,
    min_cover: float = 0.5,
) -> Tuple[List[int], List[int]]:
    cx, cy = center
    r2 = float(radius) * float(radius)
    kept: List[int] = []; removed: List[int] = []

    for i, (x, y, w, h) in enumerate(boxes):
        if w <= 0 or h <= 0:
            removed.append(i); continue
        xs = np.linspace(x + 0.5, x + w - 0.5, num=max(2, samples), dtype=np.float32)
        ys = np.linspace(y + 0.5, y + h - 0.5, num=max(2, samples), dtype=np.float32)
        total = len(xs) * len(ys); in_count = 0
        for yy in ys:
            dy2 = (yy - cy) * (yy - cy)
            dx2 = (xs - cx) * (xs - cx)
            in_count += int(np.count_nonzero(dx2 + dy2 <= r2))
        cover = in_count / float(total)
        (kept if cover >= min_cover else removed).append(i)
    return kept, removed