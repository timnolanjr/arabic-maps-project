from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence
from src.text.backends.mser import Region

@dataclass
class GeometryConfig:
    area_px: Tuple[int, int] = (30, 20000)
    aspect_ratio: Tuple[float, float] = (0.15, 8.0)

def geometry_filter_regions(
    regions: Sequence[Region],
    cfg: GeometryConfig,
) -> Tuple[List[int], List[int]]:
    kept: List[int] = []
    removed: List[int] = []
    amin, amax = cfg.area_px
    ar_min, ar_max = cfg.aspect_ratio
    for i, r in enumerate(regions):
        x, y, w, h = r.bbox
        if w <= 0 or h <= 0: removed.append(i); continue
        area = w * h
        if area < amin or area > amax: removed.append(i); continue
        ar = w / float(h); ar_inv = h / float(w)
        if not ((ar_min <= ar <= ar_max) or (ar_min <= ar_inv <= ar_max)):
            removed.append(i); continue
        kept.append(i)
    return kept, removed

def geometry_filter_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    cfg: GeometryConfig,
) -> Tuple[List[int], List[int]]:
    kept: List[int] = []
    removed: List[int] = []
    amin, amax = cfg.area_px
    ar_min, ar_max = cfg.aspect_ratio
    for i, (x, y, w, h) in enumerate(boxes):
        if w <= 0 or h <= 0: removed.append(i); continue
        area = w * h
        if area < amin or area > amax: removed.append(i); continue
        ar = w / float(h); ar_inv = h / float(w)
        if not ((ar_min <= ar <= ar_max) or (ar_min <= ar_inv <= ar_max)):
            removed.append(i); continue
        kept.append(i)
    return kept, removed