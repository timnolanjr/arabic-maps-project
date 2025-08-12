from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .vis import compose_geometry_overlay
from .palette import DEFAULT_PALETTE

# ----------------------------
# Basic image helpers
# ----------------------------
def median_blur_and_gray(bgr_img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Convert BGR image to grayscale and apply median blur."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, ksize)

def show_image(img: np.ndarray, title: Optional[str] = None, block: bool = True):
    """Display a BGR OpenCV image in Matplotlib."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.show(block=block)

# ----------------------------
# Overlay convenience
# ----------------------------
def build_overlay(
    bgr_img: np.ndarray,
    params: dict,
    text_boxes=None,
    *,
    use_default_palette: bool = True
) -> np.ndarray:
    """
    Build a geometry overlay using the unified palette. Returns a new image.
    """
    palette = DEFAULT_PALETTE if use_default_palette else DEFAULT_PALETTE
    return compose_geometry_overlay(bgr_img, params, palette=palette, text_boxes=text_boxes)

# ----------------------------
# Back-compat shim (preferred API lives in src.utils.overlay)
# ----------------------------
from pathlib import Path
import json
import warnings

def render_overlay_from_paths(
    img_path: Path,
    params_json_path: Path,
    out_path: Path,
    *,
    unicode_glyphs: bool = True,  # accepted for back-compat; prefer lang='ar' in new API
    text_boxes=None,
):
    """
    Deprecated: use src.utils.overlay.render_overlay_from_paths instead.
    """
    warnings.warn(
        "render_overlay_from_paths is deprecated; use src.utils.overlay.render_overlay_from_paths",
        DeprecationWarning,
        stacklevel=2,
    )
    from .overlay import render_overlay_from_paths as _impl
    return _impl(img_path, params_json_path, out_path, unicode_glyphs=unicode_glyphs)
