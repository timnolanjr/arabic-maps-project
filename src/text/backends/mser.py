from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

@dataclass
class Region:
    contour: np.ndarray              # (N,2)
    bbox: Tuple[int, int, int, int]  # x,y,w,h
    area: int                        # convex-hull area proxy
    extent: float                    # area / (w*h)

def _contour_area(points: np.ndarray) -> int:
    if points is None or len(points) < 3:
        return 0
    hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.int32))
    return int(abs(cv2.contourArea(hull)))

def _make_mser(
    *, delta, min_area, max_area, max_variation, min_diversity,
    max_evolution, area_threshold, min_margin, edge_blur_size,
):
    # 1) keyword args (some builds support underscores)
    try:
        return cv2.MSER_create(
            _delta=delta, _min_area=min_area, _max_area=max_area,
            _max_variation=max_variation, _min_diversity=min_diversity,
            _max_evolution=max_evolution, _area_threshold=area_threshold,
            _min_margin=min_margin, _edge_blur_size=edge_blur_size,
        )
    except TypeError:
        pass
    # 2) positional (widely supported)
    try:
        return cv2.MSER_create(
            int(delta), int(min_area), int(max_area),
            float(max_variation), float(min_diversity), int(max_evolution),
            float(area_threshold), float(min_margin), int(edge_blur_size),
        )
    except TypeError:
        pass
    # 3) no-arg + setters
    m = cv2.MSER_create()
    if hasattr(m, "setDelta"): m.setDelta(int(delta))
    if hasattr(m, "setMinArea"): m.setMinArea(int(min_area))
    if hasattr(m, "setMaxArea"): m.setMaxArea(int(max_area))
    if hasattr(m, "setMaxVariation"): m.setMaxVariation(float(max_variation))
    if hasattr(m, "setMinDiversity"): m.setMinDiversity(float(min_diversity))
    if hasattr(m, "setMaxEvolution"): m.setMaxEvolution(int(max_evolution))
    if hasattr(m, "setAreaThreshold"): m.setAreaThreshold(float(area_threshold))
    if hasattr(m, "setMinMargin"): m.setMinMargin(float(min_margin))
    if hasattr(m, "setEdgeBlurSize"): m.setEdgeBlurSize(int(edge_blur_size))
    return m

def detect_mser_regions(
    bgr: np.ndarray,
    *, delta=5, min_area=60, max_area=14400, max_variation=0.3, min_diversity=0.2,
    max_evolution=200, area_threshold=1.01, min_margin=0.003, edge_blur_size=3,
) -> List[Region]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mser = _make_mser(
        delta=delta, min_area=min_area, max_area=max_area, max_variation=max_variation,
        min_diversity=min_diversity, max_evolution=max_evolution,
        area_threshold=area_threshold, min_margin=min_margin, edge_blur_size=edge_blur_size,
    )
    pts_list, boxes = mser.detectRegions(gray)
    regions: List[Region] = []
    for pts, (x, y, w, h) in zip(pts_list, boxes):
        area = _contour_area(pts)
        extent = float(area) / float(max(1, w * h))
        regions.append(Region(pts, (int(x), int(y), int(w), int(h)), area, extent))
    return regions