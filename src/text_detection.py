# src/text_detection.py

#!/usr/bin/env python3
"""
Text detection utilities for the Arabic Maps project.

Each detector now takes an `nms_filter: bool = False` argument.
If True, non-max suppression is run; otherwise raw boxes are returned.
"""

from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

BoundingBox = Tuple[int, int, int, int]


def _prepare_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    return cv2.medianBlur(gray, 3)


def detect_text_morphology(
    img: np.ndarray,
    *,
    nms_filter: bool = False,
) -> List[BoundingBox]:
    gray = _prepare_image(img)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31, C=15,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[BoundingBox] = []
    h, w = gray.shape
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 100:
            continue
        aspect = cw / float(ch)
        if aspect < 0.2 or aspect > 15:
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + cw), min(h, y + ch)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    return _non_max_suppression(boxes) if nms_filter else boxes


def detect_text_mser(
    img: np.ndarray,
    *,
    # MSER parameters
    delta: int = 5,
    min_area: int = 60,
    max_area: int = 14400,
    max_variation: float = 0.3,
    min_diversity: float = 0.2,
    max_evolution: int = 1000,
    area_threshold: float = 1.01,
    min_margin: float = 0.003,
    edge_blur_size: int = 3,
    
    # Post-run filters
    geom_filter: bool = False,
    sw_filter: bool = False,
    sw_threshold: float = 0.4,
    geom_thresholds: dict | None = None,
    nms_filter: bool = False,
) -> List[BoundingBox]:
    gray = _prepare_image(img)

    mser = cv2.MSER_create(
        delta,
        min_area,
        max_area,
        max_variation,
        min_diversity,
        max_evolution,
        area_threshold,
        min_margin,
        edge_blur_size,
    )
    regions, _ = mser.detectRegions(gray)

    geom_thresholds = geom_thresholds or {
        "aspect_max": 3.0,
        "eccentricity_max": 0.995,
        "solidity_min": 0.3,
        "extent_range": (0.2, 0.9),
        "euler_min": -4,
    }

    boxes: List[BoundingBox] = []
    h, w = gray.shape

    for pts in regions:
        x, y, cw, ch = cv2.boundingRect(pts.reshape(-1, 1, 2))
        if cw * ch < min_area:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[pts[:, 1], pts[:, 0]] = 1

        if geom_filter:
            props = regionprops(mask)[0]
            aspect = cw / float(ch)
            if aspect > geom_thresholds["aspect_max"]:
                continue
            if props.eccentricity > geom_thresholds["eccentricity_max"]:
                continue
            if props.solidity < geom_thresholds["solidity_min"]:
                continue
            if not (geom_thresholds["extent_range"][0] <= props.extent <= geom_thresholds["extent_range"][1]):
                continue
            if props.euler_number < geom_thresholds["euler_min"]:
                continue

        if sw_filter:
            padded = np.pad(mask, 1, mode="constant", constant_values=0)
            dist = distance_transform_edt(padded)
            skel = skeletonize(padded > 0)
            sw_vals = dist[skel]
            if sw_vals.size == 0:
                continue
            sw_metric = sw_vals.std() / float(sw_vals.mean())
            if sw_metric > sw_threshold:
                continue

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + cw), min(h, y + ch)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    return _non_max_suppression(boxes) if nms_filter else boxes


def detect_text_canny(
    img: np.ndarray,
    *,
    nms_filter: bool = False,
) -> List[BoundingBox]:
    gray = _prepare_image(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BoundingBox] = []
    h, w = gray.shape
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 100:
            continue
        aspect = cw / float(ch)
        if aspect < 0.1 or aspect > 20:
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + cw), min(h, y + ch)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    return _non_max_suppression(boxes) if nms_filter else boxes


def detect_text_sobel(
    img: np.ndarray,
    *,
    nms_filter: bool = False,
) -> List[BoundingBox]:
    gray = _prepare_image(img)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobel_x, sobel_y)
    mag8 = cv2.convertScaleAbs(mag)
    _, binary = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BoundingBox] = []
    h, w = gray.shape
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 100:
            continue
        aspect = cw / float(ch)
        if aspect < 0.1 or aspect > 20:
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + cw), min(h, y + ch)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    return _non_max_suppression(boxes) if nms_filter else boxes


def detect_text_gradient(
    img: np.ndarray,
    *,
    nms_filter: bool = False,
) -> List[BoundingBox]:
    gray = _prepare_image(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    grad8 = cv2.convertScaleAbs(grad)
    _, binary = cv2.threshold(grad8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, horiz, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BoundingBox] = []
    h, w = gray.shape
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 100:
            continue
        aspect = cw / float(ch)
        if aspect < 0.1 or aspect > 20:
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + cw), min(h, y + ch)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    return _non_max_suppression(boxes) if nms_filter else boxes


def draw_bounding_boxes(
    img: np.ndarray,
    boxes: List[BoundingBox],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def _non_max_suppression(
    boxes: List[BoundingBox],
    iou_threshold: float = 0.3
) -> List[BoundingBox]:
    if not boxes:
        return []
    arr = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=float)
    x1, y1, x2, y2 = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return [
        (int(x1[idx]), int(y1[idx]), int(x2[idx] - x1[idx]), int(y2[idx] - y1[idx]))
        for idx in keep
    ]
