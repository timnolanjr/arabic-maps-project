"""
Text detection utilities for the Arabic Maps project.

This module implements several simple text‑detection algorithms based
on OpenCV primitives.  The goal is to provide working fallback
detectors in pure Python without any heavy third‑party dependencies.
If more sophisticated detectors (e.g. MMOCR, EasyOCR, PaddleOCR) are
available in your environment, you can extend this module by adding
additional methods here.  Each detector function accepts a BGR or
grayscale image as a NumPy array and returns a list of bounding boxes
in ``(x, y, w, h)`` format.

The following detectors are currently provided:

``detect_text_morphology``
    Uses adaptive thresholding and morphological operations to connect
    characters into candidate text blobs.  Suitable for high‑contrast
    images where text is darker than the background.

``detect_text_mser``
    Wraps OpenCV's MSER (Maximally Stable Extremal Regions) detector.
    This method is more robust to varying illumination but may produce
    many small regions; post‑processing filters small or extremely
    elongated boxes.

``detect_text_canny``
    Applies Canny edge detection followed by dilation to highlight
    clusters of edges that form text.  Works best when characters have
    distinct edges but may merge neighbouring words into a single box.

``draw_bounding_boxes``
    Given an image and a list of bounding boxes, returns a copy of
    the image with rectangles drawn around each detected region.

Example usage::

    from pathlib import Path
    from src.text import detect_text_morphology, draw_bounding_boxes
    import cv2

    img = cv2.imread("/path/to/map.jpg")
    boxes = detect_text_morphology(img)
    annotated = draw_bounding_boxes(img, boxes)
    cv2.imwrite("output.jpg", annotated)

"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


BoundingBox = Tuple[int, int, int, int]


def _prepare_image(img: np.ndarray) -> np.ndarray:
    """Ensure the input image is grayscale.

    If ``img`` has 3 channels (BGR), it is converted to grayscale.
    A median blur is applied to reduce noise before thresholding.

    Parameters
    ----------
    img:
        Input image as a NumPy array in BGR or grayscale format.

    Returns
    -------
    np.ndarray
        A single‑channel grayscale image.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    # Median filter to suppress pepper noise without blurring edges too much
    return cv2.medianBlur(gray, 3)


def detect_text_morphology(img: np.ndarray) -> List[BoundingBox]:
    """Detect candidate text regions using morphological operations.

    This method converts the image to grayscale, applies adaptive
    thresholding to binarise it, and then dilates the binary mask with
    a horizontal structuring element to bridge small gaps between
    characters.  Contours of the dilated mask are extracted and
    returned as bounding boxes.  Very small regions and boxes with
    extreme aspect ratios are filtered out.

    Parameters
    ----------
    img:
        Input image in BGR or grayscale format.

    Returns
    -------
    List[BoundingBox]
        A list of detected bounding boxes (x, y, width, height).
    """
    gray = _prepare_image(img)

    # Adaptive thresholding produces a binary image where text appears
    # white (1) on a black (0) background.  The blockSize and C
    # parameters may need tuning for different resolutions.
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15,
    )

    # Dilation with a horizontal kernel helps merge adjacent characters
    # into single connected components.  Adjust the kernel width based
    # on image size; here a 15×3 rectangle is used as a heuristic.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours of dilated regions.  OpenCV returns a hierarchy
    # along with the list of contours; we ignore the hierarchy.
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[BoundingBox] = []
    h, w = gray.shape
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        # Filter out very small regions
        area = cw * ch
        if area < 100:  # skip tiny blobs
            continue
        # Filter out extremely tall or wide boxes (likely non‑text)
        aspect = cw / float(ch)
        if aspect < 0.2 or aspect > 15:
            continue
        # Ensure the box lies within image bounds
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + cw)
        y1 = min(h, y + ch)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    # Remove overlapping boxes using NMS to reduce duplicates
    return _non_max_suppression(boxes)


def detect_text_mser(img: np.ndarray) -> List[BoundingBox]:
    """Detect text regions using Maximally Stable Extremal Regions (MSER).

    MSER is a blob detector that is often used for text detection in
    natural images.  Detected regions are converted to bounding boxes.
    Small regions and boxes with extreme aspect ratios are filtered out
    to reduce false positives.

    Parameters
    ----------
    img:
        Input image in BGR or grayscale format.

    Returns
    -------
    List[BoundingBox]
        A list of bounding boxes enclosing MSER regions.
    """
    gray = _prepare_image(img)

    # Initialise MSER detector.  Parameters such as delta, minArea and
    # maxArea control the stability and size of detected regions.  These
    # values may require tuning depending on the resolution of your
    # images.  See OpenCV documentation for details.
    # Create an MSER detector with default parameters.  Some OpenCV
    # versions do not accept keyword arguments for MSER_create.  If you
    # wish to adjust settings such as delta or min/max area, use the
    # corresponding setter methods on the returned detector.  The
    # defaults are chosen to work reasonably well for high‑resolution
    # scanned images.
    mser = cv2.MSER_create()

    regions, _ = mser.detectRegions(gray)
    boxes: List[BoundingBox] = []
    h, w = gray.shape
    for pts in regions:
        x, y, cw, ch = cv2.boundingRect(pts.reshape(-1, 1, 2))
        area = cw * ch
        if area < 100:
            continue
        aspect = cw / float(ch)
        if aspect < 0.2 or aspect > 15:
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + cw), min(h, y + ch)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    return _non_max_suppression(boxes)


def detect_text_canny(img: np.ndarray) -> List[BoundingBox]:
    """Detect text regions using Canny edges and dilation.

    This heuristic method uses Canny edge detection to find edges in
    the image.  Edges are dilated to connect neighbouring strokes,
    forming blobs corresponding to text lines.  Contours of these
    blobs are extracted and filtered by size and aspect ratio.

    Parameters
    ----------
    img:
        Input image in BGR or grayscale format.

    Returns
    -------
    List[BoundingBox]
        A list of bounding boxes enclosing detected text regions.
    """
    gray = _prepare_image(img)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # Dilate edges to link characters.  A rectangular kernel merges
    # neighbouring strokes; adjust kernel dimensions as needed.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[BoundingBox] = []
    h, w = gray.shape
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < 100:
            continue
        aspect = cw / float(ch)
        if aspect < 0.1 or aspect > 20:
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + cw), min(h, y + ch)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    return _non_max_suppression(boxes)


def draw_bounding_boxes(img: np.ndarray, boxes: List[BoundingBox], color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Return a copy of the image with bounding boxes drawn.

    Parameters
    ----------
    img:
        Input image (BGR) on which to draw.

    boxes:
        List of bounding boxes to draw, each as (x, y, w, h).

    color:
        Tuple specifying the colour of the rectangle in BGR format.

    thickness:
        Thickness of the rectangle outline.

    Returns
    -------
    np.ndarray
        Image copy with rectangles drawn.
    """
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return out


def _non_max_suppression(boxes: List[BoundingBox], iou_threshold: float = 0.3) -> List[BoundingBox]:
    """Suppress overlapping bounding boxes using non‑maximum suppression (NMS).

    The NMS algorithm iteratively selects the box with the largest area
    and removes any remaining boxes whose Intersection over Union (IoU)
    with the selected box exceeds ``iou_threshold``.  This reduces
    duplicate detections that arise from methods like MSER.

    Parameters
    ----------
    boxes:
        List of bounding boxes as (x, y, w, h).

    iou_threshold:
        IoU threshold above which boxes are considered duplicates and
        suppressed.  Lower values are more aggressive.

    Returns
    -------
    List[BoundingBox]
        Filtered list of boxes after suppression.
    """
    if not boxes:
        return []

    # Convert to numpy arrays for vectorised operations
    boxes_np = np.array([
        [x, y, x + w, y + h] for (x, y, w, h) in boxes
    ], dtype=float)
    x1, y1, x2, y2 = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Sort boxes by area (largest first)
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
        # Compute IoU and find indices where IoU is below threshold
        ious = inter / union
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    # Reconstruct list of boxes to return
    return [
        (int(x1[idx]), int(y1[idx]), int(x2[idx] - x1[idx]), int(y2[idx] - y1[idx]))
        for idx in keep
    ]
