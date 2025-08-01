"""
Text detection wrappers for the Arabic Maps Project.

This module exposes helper functions to run a variety of
off-the-shelf text-detection algorithms on a given image. Each
detector returns its bounding boxes in a common format so that
downstream code can remain agnostic to the specific library in
use. The current implementation supports MMOCR (legacy “MMOCR” API
and new “MMOCRInferencer”), EasyOCR and PaddleOCR.

Polygons are always returned as a list of 4-point polygons,
represented as lists of `(x, y)` integer tuples. When the
underlying detector produces a flat coordinate list, this module
handles the conversion internally.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Callable, Any
from pathlib import Path
import cv2
import numpy as np


def _load_image(path: Path) -> np.ndarray:
    """Load an image from disk using OpenCV."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def _process_polygons(
    raw_polys: List[List[float | int | Any]]
) -> List[List[Tuple[int, int]]]:
    """
    Convert a list of flat coordinate lists into lists of (x,y) tuples.
    MMOCR returns det_polygons as [x1,y1,x2,y2,...]; this helper
    converts into a list of (x,y) pairs.
    """
    processed: List[List[Tuple[int, int]]] = []
    for poly in raw_polys or []:
        if not isinstance(poly, (list, tuple)):
            continue
        # already in [(x,y),...] form?
        if len(poly) and isinstance(poly[0], (list, tuple)) and len(poly[0]) == 2:
            processed.append([(int(pt[0]), int(pt[1])) for pt in poly])
            continue
        # flat list: [x1,y1,x2,y2,...]
        if len(poly) % 2 != 0:
            continue
        pts: List[Tuple[int, int]] = []
        for x, y in zip(poly[::2], poly[1::2]):
            try:
                xi = int(round(float(x)))
                yi = int(round(float(y)))
            except (ValueError, TypeError):
                continue
            pts.append((xi, yi))
        if pts:
            processed.append(pts)
    return processed


def detect_dbnet(
    image: np.ndarray,
    device: str = "cpu",
    binary_thresh: float = 0.3,
    box_thresh: float = 0.5,
    max_candidates: int = 2000,
) -> List[List[Tuple[int, int]]]:
    """
    Run DBNet on `image`, returning a list of 4-point polygons.
    You can tune:
      - binary_thresh   (pixels > this in the binarized map get kept)
      - box_thresh      (boxes with score < this get filtered out)
      - max_candidates  (max number of candidate boxes before NMS)
    """
    # --- Try the NEW MMOCR 1.x API first ---
    try:
        # correct import path for 1.x:
        from mmocr.apis import MMOCRInferencer

        # pass your thresholds in via cfg_options:
        ocr = MMOCRInferencer(
            det="DBNet",
            rec=None,
            device=device,
            cfg_options={
                "model.test_cfg.binary_thresh": binary_thresh,
                "model.test_cfg.box_thresh":    box_thresh,
                "model.test_cfg.max_candidates": max_candidates,
            },
        )
        result = ocr(image)
        # unpack the first batch of predictions:
        pred0 = result.get("predictions", [None])[0] or {}
        raw_polys = pred0.get("polygons") or pred0.get("det_polygons") or []
        return _process_polygons(raw_polys)

    # fall back only if that import truly doesn't exist
    except ImportError:
        from mmocr.apis.inference import MMOCR  # legacy 0.x API

        ocr = MMOCR(
            det="dbnet_resnet50",   # legacy name
            rec=None,
            device=device,
            det_dbnet={
                "binary_thresh": binary_thresh,
                "box_thresh":      box_thresh,
                "max_candidates":  max_candidates,
            },
        )
        result = ocr(image)
        res0 = result[0] if isinstance(result, list) and result else result or {}
        raw_polys = res0.get("det_polygons", [])
        return _process_polygons(raw_polys)



def detect_psenet(image: np.ndarray, device: str = "cpu") -> List[List[Tuple[int, int]]]:
    """Detect text polygons using PSENet via MMOCR."""
    try:
        from mmocr.apis.inferencers import MMOCRInferencer  # type: ignore

        ocr = MMOCRInferencer(det="PSENet", rec=None, device=device)
        result = ocr(image)
        pred0 = result.get("predictions", [])[0] if isinstance(result, dict) else None
        raw_polys = pred0.get("polygons") or pred0.get("det_polygons") or [] if pred0 else []
        return _process_polygons(raw_polys)
    except Exception:
        from mmocr.apis.inference import MMOCR  # type: ignore

        ocr = MMOCR(det="PSENet", rec=None, device=device)
        result = ocr.readtext(image, output=None, print_result=False)
        res0 = result[0] if isinstance(result, list) and result else result
        raw_polys = (res0 or {}).get("det_polygons", [])
        return _process_polygons(raw_polys)


def detect_textsnake(
    image: np.ndarray, device: str = "cpu"
) -> List[List[Tuple[int, int]]]:
    """Detect text polygons using TextSnake via MMOCR."""
    try:
        from mmocr.apis.inferencers import MMOCRInferencer  # type: ignore

        ocr = MMOCRInferencer(det="TextSnake", rec=None, device=device)
        result = ocr(image)
        pred0 = result.get("predictions", [])[0] if isinstance(result, dict) else None
        raw_polys = pred0.get("polygons") or pred0.get("det_polygons") or [] if pred0 else []
        return _process_polygons(raw_polys)
    except Exception:
        from mmocr.apis.inference import MMOCR  # type: ignore

        ocr = MMOCR(det="TextSnake", rec=None, device=device)
        result = ocr.readtext(image, output=None, print_result=False)
        res0 = result[0] if isinstance(result, list) and result else result
        raw_polys = (res0 or {}).get("det_polygons", [])
        return _process_polygons(raw_polys)


def detect_easyocr(image: np.ndarray, lang: str = "ar") -> List[Tuple[List[Tuple[int, int]], str]]:
    """Detect and recognise text using EasyOCR."""
    import easyocr  # type: ignore

    reader = easyocr.Reader([lang], gpu=False)
    results = reader.readtext(image)
    processed = []
    for bbox, text, _ in results:
        pts = [(int(pt[0]), int(pt[1])) for pt in bbox]
        processed.append((pts, text))
    return processed


def detect_paddleocr(
    image: np.ndarray, lang: str = "arabic"
) -> List[Tuple[List[Tuple[int, int]], str]]:
    """Detect and recognise text using PaddleOCR."""
    from paddleocr import PaddleOCR  # type: ignore

    ocr = PaddleOCR(lang=lang)
    results = ocr.ocr(image, cls=False)
    processed = []
    for line in results:
        for bbox, (text, _conf) in line:
            pts = [(int(pt[0]), int(pt[1])) for pt in bbox]
            processed.append((pts, text))
    return processed


DETECTOR_REGISTRY: Dict[str, Callable[..., Any]] = {
    "dbnet": detect_dbnet,
    "psenet": detect_psenet,
    "textsnake": detect_textsnake,
    "easyocr": detect_easyocr,
    "paddleocr": detect_paddleocr,
}


def detect_text(
    image_path: Path, detector_name: str = "dbnet", **kwargs: Any
) -> List[Any]:
    """
    Generic text‐detection entry point. Reads an image from
    ``image_path`` and dispatches to one of the detectors
    registered in ``DETECTOR_REGISTRY``.
    """
    image = _load_image(image_path)
    if detector_name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {detector_name}")
    detector_fn = DETECTOR_REGISTRY[detector_name]
    return detector_fn(image, **kwargs)
