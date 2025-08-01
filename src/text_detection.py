"""
text_detect.py
================

This module exposes a set of helper functions for running various off‑the‑shelf
text detection algorithms on a given image.  Each detector returns its
bounding boxes in a common format so that downstream code can remain agnostic
to the specific library in use.

Supported detectors
-------------------

The following detectors are currently implemented.  Each function takes an
OpenCV ``numpy.ndarray`` image and returns a list of polygons (lists of
``(x, y)`` integer tuples) describing the detected text regions.  For
detectors that perform recognition (EasyOCR/PaddleOCR), the text will be
returned alongside the polygon as a tuple ``(polygon, text)``.

* ``detect_dbnet`` – Runs the DBNet text detector from MMOCR.  DBNet is a
  segmentation‑based method that uses differentiable binarization to adaptively
  threshold the probability map【695739460380995†L371-L382】.
* ``detect_psenet`` – Runs the Progressive Scale Expansion Network (PSENet)
  detector.  PSENet generates multiple kernels for each text instance and
  progressively expands them to the full shape【103694822508524†L68-L77】.
* ``detect_textsnake`` – Runs TextSnake, which represents text as a sequence
  of overlapping discs along a symmetric axis【481200937890620†L335-L344】.
* ``detect_easyocr`` – Uses EasyOCR's CRAFT‑based detector and recogniser.  This
  method returns both bounding boxes and the recognised text.  Under the
  hood CRAFT uses text, low‑text and link thresholds to filter regions【437174331985299†L46-L52】.
* ``detect_paddleocr`` – Uses PaddleOCR to detect and recognise text.  Only
  the language can currently be configured; the underlying detector thresholds
  remain at their defaults【240472464281277†L681-L696】.
* ``detect_craft`` – Runs the CRAFT detector directly via MMOCR.  Thresholds
  controlling text confidence, link confidence and the low‑text cutoff can be
  adjusted【437174331985299†L46-L52】.
* ``detect_east`` – Runs the EAST detector using OpenCV’s DNN module.  The
  confidence threshold and non‑maximum suppression (NMS) threshold can be set
  via arguments【685682250874597†L320-L347】.
* ``detect_doctr`` – Uses Mindee’s docTR detection predictor.  Supports
  selecting an architecture and controlling whether pages are assumed to be
  straight or rotated【187098695961302†L302-L319】.

The ``detect_text`` function provides a unified entry point that reads an
image from disk and dispatches to the requested detector.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import inspect


def _load_image(path: Path) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        The loaded image in BGR format.

    Raises
    ------
    FileNotFoundError
        If the image cannot be read.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def _process_polygons(raw_polys: List[List[Any]]) -> List[List[Tuple[int, int]]]:
    """Convert a list of raw polygon definitions into a list of (x, y) tuples.

    Some detectors return flat coordinate lists (e.g. ``[x1, y1, x2, y2, …]``)
    while others return nested lists of coordinate pairs.  This helper
    normalises both formats into a list of integer tuples.

    Parameters
    ----------
    raw_polys : list
        The raw polygons as returned by the detector.

    Returns
    -------
    list
        A list of polygons, each represented as a list of ``(x, y)`` tuples.
    """
    processed: List[List[Tuple[int, int]]] = []
    for poly in raw_polys or []:
        if not isinstance(poly, (list, tuple)):
            continue
        # Already a list of point tuples?
        if poly and isinstance(poly[0], (list, tuple)) and len(poly[0]) == 2:
            processed.append([(int(pt[0]), int(pt[1])) for pt in poly])
            continue
        # Flat list of coordinates [x1, y1, x2, y2, …]
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
    """Run the DBNet detector on an image, with fallback to no-cfg_options if needed."""
    try:
        # import the modern API
        from mmocr.apis import MMOCRInferencer  # type: ignore
    except ImportError as e:
        raise ImportError("mmocr>=1.0 is required for DBNet detection") from e

    # first try with cfg_options
    try:
        ocr = MMOCRInferencer(
            det="DBNet",
            rec=None,
            device=device,
            cfg_options={
                "model.test_cfg.binary_thresh":    binary_thresh,
                "model.test_cfg.box_thresh":       box_thresh,
                "model.test_cfg.max_candidates":   max_candidates,
            },
        )
    except TypeError:
        # older MMOCRInferencer versions don’t accept cfg_options → retry without it
        ocr = MMOCRInferencer(det="DBNet", rec=None, device=device)

    # run inference
    result = ocr(image)
    pred0 = (result.get("predictions", [None])[0] or {})
    raw_polys = pred0.get("polygons") or pred0.get("det_polygons") or []
    return _process_polygons(raw_polys)



def detect_psenet(image: np.ndarray, device: str = "cpu") -> List[List[Tuple[int, int]]]:
    """Detect text regions using PSENet.

    PSENet progressively expands shrunk kernels to recover full text shapes【103694822508524†L68-L77】.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    device : str, optional
        Device on which to run the model.

    Returns
    -------
    list
        A list of polygons describing detected text regions.
    """
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


def detect_textsnake(image: np.ndarray, device: str = "cpu") -> List[List[Tuple[int, int]]]:
    """Detect text regions using TextSnake.

    TextSnake represents text as overlapping discs along a symmetric axis【481200937890620†L335-L344】.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    device : str, optional
        Device on which to run the model.

    Returns
    -------
    list
        A list of polygons describing detected text regions.
    """
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
    """Detect and recognise text using EasyOCR.

    EasyOCR wraps the CRAFT detector and returns bounding boxes along with
    recognised text.  Only the recognition language can be specified in this
    wrapper; detection thresholds remain at their defaults【437174331985299†L46-L52】.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    lang : str, optional
        Language code to use for recognition (default is Arabic).

    Returns
    -------
    list
        A list of tuples ``(polygon, text)`` where ``polygon`` is a list of
        ``(x, y)`` tuples and ``text`` is the recognised string.
    """
    import easyocr  # type: ignore

    reader = easyocr.Reader([lang], gpu=False)
    results = reader.readtext(image)
    processed: List[Tuple[List[Tuple[int, int]], str]] = []
    for bbox, text, _ in results:
        pts = [(int(pt[0]), int(pt[1])) for pt in bbox]
        processed.append((pts, text))
    return processed


def detect_paddleocr(
    image: np.ndarray,
    lang: str = "arabic",
) -> List[Tuple[List[Tuple[int, int]], str]]:
    """Detect and recognise text using PaddleOCR.

    PaddleOCR supports multiple languages and provides a complete OCR pipeline.
    Only the language parameter is exposed in this wrapper; the underlying
    detection thresholds remain at their defaults【240472464281277†L681-L696】.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    lang : str, optional
        Language code (e.g. ``"arabic"``) used for the detection and
        recognition models.

    Returns
    -------
    list
        A list of tuples ``(polygon, text)`` where ``polygon`` is a list of
        ``(x, y)`` tuples and ``text`` is the recognised string.
    """
    from paddleocr import PaddleOCR  # type: ignore

    ocr = PaddleOCR(lang=lang)
    results = ocr.ocr(image, cls=False)
    processed: List[Tuple[List[Tuple[int, int]], str]] = []
    for line in results:
        for bbox, (text, _conf) in line:
            pts = [(int(pt[0]), int(pt[1])) for pt in bbox]
            processed.append((pts, text))
    return processed


def detect_craft(
    image: np.ndarray,
    cuda: bool = False,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    long_size: int = 1280,
) -> List[List[Tuple[int, int]]]:
    """Detect text boxes using the CRAFT algorithm via MMOCR.

    CRAFT predicts character regions and affinities between characters.  The
    thresholds control the text confidence, link confidence and lower bound
    respectively【437174331985299†L46-L52】.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in BGR format.
    cuda : bool, optional
        Whether to run on GPU (``cuda:0``) or CPU (default).
    text_threshold : float, optional
        Threshold for text confidence scores.
    link_threshold : float, optional
        Threshold for link/affinity scores.
    low_text : float, optional
        Lower bound for text scores.
    long_size : int, optional
        Resize the longer side of the image to this size before inference.

    Returns
    -------
    list
        A list of polygons describing detected text regions.
    """
    from mmocr.apis import MMOCRInferencer  # type: ignore

    device_str = "cuda:0" if cuda else "cpu"
    cfg_options = {
        "model.test_cfg.text_threshold": text_threshold,
        "model.test_cfg.link_threshold": link_threshold,
        "model.test_cfg.low_text": low_text,
        "model.test_cfg.long_size": long_size,
    }
    ocr = MMOCRInferencer(det="CRAFT", rec=None, device=device_str, cfg_options=cfg_options)
    result = ocr(image)
    pred0 = (result.get("predictions", []) or [{}])[0]
    raw_polys = pred0.get("polygons") or pred0.get("det_polygons") or []
    return _process_polygons(raw_polys)


def _decode_east_predictions(
    scores: np.ndarray,
    geometry: np.ndarray,
    conf_threshold: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """Decode EAST output into bounding boxes.

    This helper is used internally by ``detect_east`` and replicates the
    post‑processing described in the EAST paper.  It decodes geometry and score
    maps into bounding boxes and corresponding confidence scores.

    Parameters
    ----------
    scores : numpy.ndarray
        Score volume (1×1×H×W) from the EAST model.
    geometry : numpy.ndarray
        Geometry volume (1×5×H×W) from the EAST model.
    conf_threshold : float
        Confidence threshold to retain candidate boxes.

    Returns
    -------
    list of tuples
        Bounding boxes in the form ``(startX, startY, endX, endY)``.
    list of floats
        Confidence scores corresponding to each box.
    """
    (num_rows, num_cols) = scores.shape[2:4]
    boxes: List[Tuple[int, int, int, int]] = []
    confidences: List[float] = []
    for y in range(num_rows):
        for x in range(num_cols):
            score = scores[0, 0, y, x]
            if score < conf_threshold:
                continue
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            end_x = offset_x + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x]
            end_y = offset_y - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x]
            start_x = end_x - w
            start_y = end_y - h
            boxes.append((int(start_x), int(start_y), int(end_x), int(end_y)))
            confidences.append(float(score))
    return boxes, confidences


def detect_east(
    image: np.ndarray,
    model_path: str = "frozen_east_text_detection.pb",
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> List[List[Tuple[int, int]]]:
    """Detect text boxes using the EAST detector.

    The EAST model produces a score map and a geometry map.  Boxes are
    filtered by the confidence threshold and overlapping boxes are removed
    using non‑maximum suppression.  The OpenCV API exposes methods to set
    these thresholds【685682250874597†L320-L347】.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in BGR format.
    model_path : str, optional
        Path to the frozen EAST model (.pb).  You must download a compatible
        model before using this detector.
    conf_threshold : float, optional
        Minimum confidence for a candidate region to be considered.
    nms_threshold : float, optional
        Non‑maxima suppression threshold used to suppress overlapping boxes.

    Returns
    -------
    list
        A list of polygons (each a list of four points) representing detected
        text regions.

    Raises
    ------
    FileNotFoundError
        If the EAST model file cannot be found at ``model_path``.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"EAST model file '{model_path}' could not be found. "
            "Download the frozen graph and provide its path via model_path."
        )
    net = cv2.dnn.readNet(model_path)
    (H, W) = image.shape[:2]
    new_w, new_h = 320, 320
    rW = W / float(new_w)
    rH = H / float(new_h)
    resized = cv2.resize(image, (new_w, new_h))
    blob = cv2.dnn.blobFromImage(
        resized,
        1.0,
        (new_w, new_h),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    (scores, geometry) = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ])
    boxes, confidences = _decode_east_predictions(scores, geometry, conf_threshold)
    if not boxes:
        return []
    rects = []
    for (start_x, start_y, end_x, end_y) in boxes:
        rects.append([
            int(start_x),
            int(start_y),
            int(end_x - start_x),
            int(end_y - start_y),
        ])
    indices = cv2.dnn.NMSBoxes(rects, confidences, conf_threshold, nms_threshold)
    polygons: List[List[Tuple[int, int]]] = []
    if len(indices) > 0:
        for i in indices.flatten():
            (start_x, start_y, end_x, end_y) = boxes[i]
            sx = int(round(start_x * rW))
            sy = int(round(start_y * rH))
            ex = int(round(end_x * rW))
            ey = int(round(end_y * rH))
            polygons.append([
                (sx, sy),
                (ex, sy),
                (ex, ey),
                (sx, ey),
            ])
    return polygons


def detect_doctr(
    image: np.ndarray,
    arch: str = "db_resnet50",
    pretrained: bool = True,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    batch_size: int = 1,
) -> List[List[Tuple[int, int]]]:
    """Detect text using a docTR detection predictor.

    docTR supports several architectures (e.g. ``db_resnet50``, ``linknet_resnet18``)
    and offers options to control page geometry assumptions【187098695961302†L302-L319】.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in BGR format.
    arch : str, optional
        Detection architecture to load.
    pretrained : bool, optional
        Whether to load pretrained weights (recommended).  If ``False``, the
        model will be randomly initialised.
    assume_straight_pages : bool, optional
        If ``True``, fit straight bounding boxes; set to ``False`` to allow
        rotated boxes.
    preserve_aspect_ratio : bool, optional
        Whether to keep the aspect ratio when resizing.
    batch_size : int, optional
        Number of images to process at once (unused in this implementation but
        exposed for API completeness).

    Returns
    -------
    list
        A list of polygons describing detected text regions.
    """
    try:
        from PIL import Image  # type: ignore
        from doctr.models import detection_predictor  # type: ignore
    except ImportError as e:
        raise ImportError(
            "doctr is required for docTR detection; install via pip"
        ) from e
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    model = detection_predictor(
        arch=arch,
        pretrained=pretrained,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )
    pages = model([pil_img])
    if not pages:
        return []
    page = pages[0]
    boxes = page.get("boxes") if isinstance(page, dict) else getattr(page, "boxes", None)
    if boxes is None:
        return []
    width, height = pil_img.size
    polygons: List[List[Tuple[int, int]]] = []
    for box in boxes:
        try:
            xmin, ymin, xmax, ymax = [float(v) for v in box]
        except Exception:
            continue
        x0 = int(round(xmin * width))
        y0 = int(round(ymin * height))
        x1 = int(round(xmax * width))
        y1 = int(round(ymax * height))
        polygons.append([
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (x0, y1),
        ])
    return polygons


DETECTOR_REGISTRY: Dict[str, Callable[..., Any]] = {
    "dbnet": detect_dbnet,
    "psenet": detect_psenet,
    "textsnake": detect_textsnake,
    "easyocr": detect_easyocr,
    "paddleocr": detect_paddleocr,
    "craft": detect_craft,
    "east": detect_east,
    "doctr": detect_doctr,
}

class TextDetInferencer:
    """
    Wraps the registered detectors and exposes a simple .detect() method
    that takes a file path and returns only the polygons.
    """
    def __init__(self, detector_name: str = "dbnet", **kwargs: Any):
        if detector_name not in DETECTOR_REGISTRY:
            raise ValueError(f"Unknown detector: {detector_name}")
        self.detector_name = detector_name
        self.kwargs = kwargs

    def detect(self, image_path: Path) -> List[List[Tuple[int, int]]]:
        """
        Load the image from disk and run the chosen detector,
        returning only the list of polygons.
        """
        image = _load_image(Path(image_path))
        detector_fn = DETECTOR_REGISTRY[self.detector_name]

        # introspect valid parameters for this detector
        sig = inspect.signature(detector_fn)
        valid_kwargs = {
            k: v for k, v in self.kwargs.items()
            if k in sig.parameters
        }

        return detector_fn(image, **valid_kwargs)



def detect_text(
    image_path: Path,
    detector_name: str = "dbnet",
    **kwargs: Any,
) -> List[Any]:
    """Read an image from disk and detect text using the specified detector.

    Parameters
    ----------
    image_path : Path
        Path to the input image.
    detector_name : str, optional
        Name of the detector to use (must be a key in ``DETECTOR_REGISTRY``).
    **kwargs : dict, optional
        Additional keyword arguments forwarded to the underlying detector.

    Returns
    -------
    list
        The detector’s raw output.  Polygons are returned for most detectors;
        EasyOCR and PaddleOCR return tuples of ``(polygon, text)``.

    Raises
    ------
    ValueError
        If ``detector_name`` is not recognised.
    FileNotFoundError
        If the image cannot be loaded.
    """
    image = _load_image(image_path)
    if detector_name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {detector_name}")
    detector_fn = DETECTOR_REGISTRY[detector_name]
    return detector_fn(image, **kwargs)