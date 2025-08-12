# src/utils/image.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Basic image helpers (yours)
# ----------------------------

def median_blur_and_gray(bgr_img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Convert BGR image to grayscale and apply median blur.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, ksize)


def show_image(img: np.ndarray, title: str = None):
    """
    Display a BGR OpenCV image in Matplotlib with optional blocking window.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.show(block=True)


# ----------------------------
# Overlay rendering utilities
# ----------------------------

def _scale_for_image(w: int, h: int) -> float:
    """Rough UI scale factor (≈1.0 at 1000px on the short side)."""
    return max(0.1, min(w, h) / 1000.0)


def _line_endpoints_from_rho_theta(
    rho: float, theta: float, w: int, h: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Convert (rho, theta) line parameters to two endpoints that span the image.
    theta in radians; rho in pixels (normal form).
    """
    a, b = math.cos(theta), math.sin(theta)
    if abs(b) < 1e-12:  # vertical line
        x = int(round(rho / (a if abs(a) > 1e-12 else 1.0)))
        return (x, 0), (x, h)
    y0 = (rho - 0 * a) / b
    y1 = (rho - w * a) / b
    return (0, int(round(y0))), (w, int(round(y1)))


def _put_text_centered(
    img: np.ndarray,
    text: str,
    center_xy: Tuple[int, int],
    font: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """
    Draw text so its bounding box is centered on (x, y).
    """
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = center_xy
    org = (int(x - tw / 2), int(y + th / 2 - bl / 2))
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def render_overlay(
    img_bgr: np.ndarray,
    params: Dict,
    *,
    draw_annotations: bool = True,
    unicode_glyphs: bool = False,
    circle_color: Tuple[int, int, int] = (0, 255, 0),
    edge_color: Tuple[int, int, int] = (255, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    label_bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Draw circle, edge, tangent, and cardinal labels on a copy of img_bgr.

    Required params keys:
      center_x, center_y, radius, rho, theta, tangent_x, tangent_y

    Returns:
      New BGR image with overlay.
    """
    required = ["center_x", "center_y", "radius", "rho", "theta", "tangent_x", "tangent_y"]
    for k in required:
        if k not in params:
            raise KeyError(f"Missing '{k}' in params")

    cx, cy = int(params["center_x"]), int(params["center_y"])
    r = int(params["radius"])
    rho, theta = float(params["rho"]), float(params["theta"])
    tx, ty = int(params["tangent_x"]), int(params["tangent_y"])

    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    scale = _scale_for_image(w, h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale
    text_thick = max(1, int(2 * scale))
    circle_thick = max(1, int(2 * scale))
    line_thick = max(1, int(2 * scale))
    marker_rad = max(2, int(max(2, r * 0.02)))
    margin = int(20 * scale)

    # Circle + center
    cv2.circle(overlay, (cx, cy), r, circle_color, thickness=circle_thick)
    cv2.circle(overlay, (cx, cy), marker_rad, circle_color, -1)

    # Edge line
    p1, p2 = _line_endpoints_from_rho_theta(rho, theta, w, h)
    cv2.line(overlay, p1, p2, edge_color, thickness=line_thick)

    # Cardinal markers (S at tangent)
    cardinals = {
        "S": (tx, ty),
        "E": (cx - r, cy),
        "N": (cx, cy + r),
        "W": (cx + r, cy),
    }
    for label, (x_, y_) in cardinals.items():
        x, y = int(x_), int(y_)
        if label == "S":
            cv2.circle(overlay, (x, y), int(1.5 * marker_rad), (0, 255, 255), -1)
            _put_text_centered(overlay, label, (x, y), font, font_scale, (0,0,0), text_thick)
        else:
            cv2.circle(overlay, (x, y), int(1.5 * marker_rad), label_bg_color, -1)
            _put_text_centered(overlay, label, (x, y), font, font_scale, text_color, text_thick)

    # Metadata annotation (optional)
    if draw_annotations:
        if unicode_glyphs:
            rho_txt = f"\u03C1={rho:.1f}"
            theta_txt = f"\u03B8={math.degrees(theta):.1f}\N{DEGREE SIGN}"
        else:
            rho_txt = f"rho={rho:.1f}"
            theta_txt = f"theta={math.degrees(theta):.1f} deg"

        lines = [
            f" ",
            f"Map Coords: cx={cx}, cy={cy}, r={r}",
            f"Top Edge: {rho_txt}, {theta_txt}",
            f"Tangent point (South) tx={tx}, ty={ty}",
        ]
        y0 = margin
        colors=[(0,0,0),(0,255,0),(255,255,0),(0,255,255)]
        for i, s in enumerate(lines):
            color = colors[i % len(colors)]
            # outline (thicker, black)
            cv2.putText(overlay, s, (margin, y0), font, font_scale, (0,0,0), text_thick + 2, cv2.LINE_AA)
            # main colored text
            cv2.putText(overlay, s, (margin, y0), font, font_scale, color,text_thick,cv2.LINE_AA)
            y0 += int(25 * scale)

    return overlay


def render_overlay_from_paths(
    image_path: Path,
    params_path: Path,
    out_path: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Convenience wrapper: load image + params.json, render, and write to disk.

    Args:
      image_path: path to the input image (JPG/PNG/etc.)
      params_path: path to params.json
      out_path: optional path for the output overlay (defaults to params_overlay.[jpg|png])
      **kwargs: forwarded to render_overlay (e.g., unicode_glyphs=True)

    Returns:
      Path to the written overlay.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    out = render_overlay(img, params, **kwargs)

    if out_path is None:
        out_ext = ".jpg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else ".png"
        out_path = image_path.with_name("params_overlay" + out_ext)

    cv2.imwrite(str(out_path), out)
    return out_path
