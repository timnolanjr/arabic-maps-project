from __future__ import annotations
from typing import Dict, Tuple, Optional
import cv2
import numpy as np

from .palette import DEFAULT_PALETTE, Palette

Point = Tuple[int, int]

def draw_circle(img: np.ndarray, center: Point, radius: int, *,
                palette: Palette = DEFAULT_PALETTE) -> None:
    cv2.circle(img, center, radius, palette.circle.bgr, palette.thick.thickness, lineType=cv2.LINE_AA)
    cv2.circle(img, center, 6, palette.center.bgr, -1, lineType=cv2.LINE_AA)

def draw_edge_line(img: np.ndarray, rho: float, theta: float, *,
                   palette: Palette = DEFAULT_PALETTE, length: Optional[int] = None) -> None:
    h, w = img.shape[:2]
    if length is None:
        length = int(1.5 * max(h, w))
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + length * (-b)), int(y0 + length * (a))
    x2, y2 = int(x0 - length * (-b)), int(y0 - length * (a))
    cv2.line(img, (x1, y1), (x2, y2), palette.rho_theta.bgr, palette.thick.thickness, lineType=cv2.LINE_AA)

def draw_tangent_point(img: np.ndarray, pt: Point, *,
                       palette: Palette = DEFAULT_PALETTE) -> None:
    size = 10
    cv2.line(img, (pt[0] - size, pt[1]), (pt[0] + size, pt[1]), palette.tangent_point.bgr, 2, cv2.LINE_AA)
    cv2.line(img, (pt[0], pt[1] - size), (pt[0], pt[1] + size), palette.tangent_point.bgr, 2, cv2.LINE_AA)
    cv2.circle(img, pt, 5, palette.tangent_point.bgr, -1, cv2.LINE_AA)

def draw_legend(img: np.ndarray, *,
                palette: Palette = DEFAULT_PALETTE,
                origin: Point = (20, 20), line_height: int = 26) -> None:
    items = [
        ("Circle", palette.circle.bgr),
        ("Edge (ρ, θ)", palette.rho_theta.bgr),
        ("Tangent (S)", palette.tangent_point.bgr),
        ("Center", palette.center.bgr),
        ("Text boxes", palette.text_box.bgr),
    ]
    x, y = origin
    pad = 10
    width = 230
    height = pad * 2 + len(items) * line_height
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - pad), (x - pad + width, y - pad + height),
                  palette.legend_bg.bgr, thickness=-1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for i, (label, color) in enumerate(items):
        yy = y + i * line_height
        cv2.circle(img, (x + 8, yy + 8), 7, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (x + 28, yy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    palette.text_label.bgr, 1, cv2.LINE_AA)

def draw_text_boxes(img: np.ndarray, boxes, *,
                    palette: Palette = DEFAULT_PALETTE) -> None:
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), palette.text_box.bgr, 2, cv2.LINE_AA)

def compose_geometry_overlay(bgr_img: np.ndarray, params: Dict, *,
                             palette: Palette = DEFAULT_PALETTE,
                             text_boxes=None) -> np.ndarray:
    """
    Returns a copy of bgr_img with circle, edge, tangent point, (optional) text boxes and legend.
    Expects params to contain: center_x, center_y, radius, rho, theta, tangent_x, tangent_y.
    """
    out = bgr_img.copy()
    cx, cy = int(params["center_x"]), int(params["center_y"])
    r = int(params["radius"])
    draw_circle(out, (cx, cy), r, palette=palette)
    if "rho" in params and "theta" in params:
        draw_edge_line(out, float(params["rho"]), float(params["theta"]), palette=palette)
    if "tangent_x" in params and "tangent_y" in params:
        draw_tangent_point(out, (int(params["tangent_x"]), int(params["tangent_y"])), palette=palette)
    if text_boxes:
        draw_text_boxes(out, text_boxes, palette=palette)
    draw_legend(out, palette=palette)
    return out
