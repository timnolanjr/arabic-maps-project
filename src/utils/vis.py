# src/utils/vis.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Union
import cv2
import numpy as np

from .palette import DEFAULT_PALETTE, Palette

Point = Tuple[int, int]

# -------- Scaling helpers --------
def _auto_scale(shape: Union[Tuple[int, int, int], Tuple[int, int]]) -> float:
    """
    Heuristic: scale thickness/markers with image size.
    Base at ~1200px on long edge -> scale=1.0. Clamp to [0.75, 6.0].
    """
    if len(shape) == 3:
        h, w = shape[:2]
    else:
        h, w = shape
    s = max(h, w) / 1200.0
    return float(max(0.75, min(6.0, s)))

def _t(px: int, scale: float) -> int:
    return max(1, int(round(px * scale)))

def _fs(base: float, scale: float) -> float:
    return max(0.5, base * scale)

def _ul_origin(shape: Tuple[int, int, int], scale: float, pad_ratio: float = 0.02) -> Point:
    """Upper-left with small padding relative to image size."""
    h, w = shape[:2]
    pad = max(6, int(round(pad_ratio * max(h, w))))
    return (pad, pad)


# -------- Primitives --------
def draw_circle(img: np.ndarray, center: Point, radius: int, *,
                palette: Palette = DEFAULT_PALETTE,
                scale: Optional[float] = None) -> None:
    s = _auto_scale(img.shape) if scale is None else scale
    cv2.circle(img, center, radius, palette.circle.bgr, _t(palette.thick.thickness, s), lineType=cv2.LINE_AA)
    cv2.circle(img, center, _t(6, s), palette.center.bgr, -1, lineType=cv2.LINE_AA)

def draw_edge_line(img: np.ndarray, rho: float, theta: float, *,
                   palette: Palette = DEFAULT_PALETTE,
                   length: Optional[int] = None,
                   scale: Optional[float] = None) -> None:
    h, w = img.shape[:2]
    s = _auto_scale(img.shape) if scale is None else scale
    if length is None:
        length = int(1.5 * max(h, w))
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + length * (-b)), int(y0 + length * (a))
    x2, y2 = int(x0 - length * (-b)), int(y0 - length * (a))
    cv2.line(img, (x1, y1), (x2, y2), palette.rho_theta.bgr, _t(palette.thick.thickness, s), lineType=cv2.LINE_AA)

def draw_tangent_point(img: np.ndarray, pt: Point, *,
                       palette: Palette = DEFAULT_PALETTE,
                       scale: Optional[float] = None) -> None:
    s = _auto_scale(img.shape) if scale is None else scale
    sz = _t(10, s)
    th = _t(2, s)
    cv2.line(img, (pt[0] - sz, pt[1]), (pt[0] + sz, pt[1]), palette.tangent_point.bgr, th, cv2.LINE_AA)
    cv2.line(img, (pt[0], pt[1] - sz), (pt[0], pt[1] + sz), palette.tangent_point.bgr, th, cv2.LINE_AA)
    cv2.circle(img, pt, _t(5, s), palette.tangent_point.bgr, -1, cv2.LINE_AA)


# -------- Legend (metadata-only with colored dots) --------
def _meta_items(meta: Dict[str, Dict[str, float]], palette: Palette) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    Build legend rows as (text, colorBGR) in the requested order:
      Center (yellow), Top Edge (blue), Tangent (red).
    """
    items: List[Tuple[str, Tuple[int, int, int]]] = []

    # Center first (yellow)
    if "center" in meta:
        cx = int(meta["center"]["x"])
        cy = int(meta["center"]["y"])
        if "radius" in meta["center"]:
            r = int(meta["center"]["radius"])
            items.append((f"Center: x={cx}, y={cy}, r={r}", palette.center.bgr))
        else:
            items.append((f"Center: x={cx}, y={cy}", palette.center.bgr))

    # Top Edge next (blue)
    if "edge" in meta:
        rho = int(round(meta["edge"]["rho"]))
        theta = float(meta["edge"]["theta"])
        items.append((f"Edge: rho={rho}, theta={theta:.3f} rad", palette.rho_theta.bgr))

    # Tangent last (red)
    if "tangent" in meta:
        tx = int(meta["tangent"]["x"])
        ty = int(meta["tangent"]["y"])
        items.append((f"Tangent (S): x={tx}, y={ty}", palette.tangent_point.bgr))

    return items

def draw_legend(img: np.ndarray, *,
                palette: Palette = DEFAULT_PALETTE,
                origin: Optional[Point] = None,
                line_height: Optional[int] = None,
                lang: str = "en",  # kept for signature compat; not used here
                labels: Optional[List[Tuple[str, Tuple[int, int, int]]]] = None,  # ignored
                scale: Optional[float] = None,
                meta: Optional[Dict[str, Dict[str, float]]] = None) -> Point:
    """
    Draw a legend in the upper-left with small padding, showing ONLY:
      • (yellow dot) Center: x=…, y=…, r=…
      • (blue dot)   Top Edge: rho=…, theta=….… rad
      • (red dot)    Tangent (S): x=…, y=…

    Returns the (x, y) origin actually used.
    """
    s = _auto_scale(img.shape) if scale is None else scale
    rows = _meta_items(meta or {}, palette)
    if not rows:
        rows = [("(no metadata)", palette.text_label.bgr)]

    # typography & spacing
    pad = _t(10, s)
    lh = line_height if line_height is not None else _t(26, s)
    dot_r = _t(7, s)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = _fs(0.6, s)
    font_thick = _t(1, s)

    def text_w(t: str) -> int:
        (w, _h), _ = cv2.getTextSize(t, font, font_scale, font_thick)
        return w

    left_glyph_w = _t(28, s)  # dot + gap
    max_w = max(text_w(txt) for txt, _ in rows)
    width = pad * 2 + left_glyph_w + max_w
    height = pad * 2 + len(rows) * lh

    # origin (UL with padding) if not provided
    if origin is None:
        x, y = _ul_origin(img.shape, s)
    else:
        x, y = origin

    # background box
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - pad), (x - pad + width, y - pad + height),
                  palette.legend_bg.bgr, thickness=-1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # draw rows
    for i, (text, color) in enumerate(rows):
        yy = y + i * lh
        cv2.circle(img, (x + 8, yy + 8), dot_r, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, text, (x + left_glyph_w, yy + _t(14, s)),
                    font, font_scale, palette.text_label.bgr, font_thick, cv2.LINE_AA)

    return int(x), int(y)


# -------- Text boxes --------
def draw_text_boxes(img: np.ndarray, boxes, *,
                    palette: Palette = DEFAULT_PALETTE,
                    scale: Optional[float] = None) -> None:
    s = _auto_scale(img.shape) if scale is None else scale
    th = _t(2, s)
    for (x, y, w, h) in boxes or []:
        cv2.rectangle(img, (x, y), (x + w, y + h), palette.text_box.bgr, th, cv2.LINE_AA)


# -------- High-level composer --------
def compose_geometry_overlay(bgr_img: np.ndarray, params: Dict, *,
                             palette: Palette = DEFAULT_PALETTE,
                             text_boxes=None,
                             add_legend: bool = True,
                             lang: str = "en",
                             scale: Optional[float] = None,
                             return_metadata: bool = False):
    """
    Draw circle/edge/tangent/text boxes and a metadata-only legend (upper-left, padded).
    Auto-scales stroke widths and text to image size.
    If return_metadata=True, returns (overlay_image, metadata_dict). Otherwise returns image.

    Metadata schema (ints for coords):
      {
        "center": {"x": int, "y": int, "radius": int?}
        "tangent": {"x": int, "y": int}
        "edge": {"rho": float, "theta": float}
        "legend": {"x": int, "y": int}
      }
    """
    out = bgr_img.copy()
    s = _auto_scale(out.shape) if scale is None else scale
    metadata: Dict[str, Dict[str, float]] = {}

    # Collect metadata from params
    if {"center_x", "center_y"} <= params.keys():
        cx, cy = int(round(params["center_x"])), int(round(params["center_y"]))
        metadata["center"] = {"x": cx, "y": cy}
        if "radius" in params:
            metadata["center"]["radius"] = int(round(params["radius"]))
    if {"tangent_x", "tangent_y"} <= params.keys():
        tx, ty = int(round(params["tangent_x"])), int(round(params["tangent_y"]))
        metadata["tangent"] = {"x": tx, "y": ty}
    if {"rho", "theta"} <= params.keys():
        metadata["edge"] = {"rho": float(params["rho"]), "theta": float(params["theta"])}

    # Draw layers
    if "center" in metadata and ("radius" in metadata["center"] or "radius" in params):
        radius = int(metadata["center"].get("radius", int(round(params["radius"]))))
        draw_circle(out, (metadata["center"]["x"], metadata["center"]["y"]), radius,
                    palette=palette, scale=s)

    if "edge" in metadata:
        draw_edge_line(out, metadata["edge"]["rho"], metadata["edge"]["theta"],
                       palette=palette, scale=s)

    if "tangent" in metadata:
        draw_tangent_point(out, (metadata["tangent"]["x"], metadata["tangent"]["y"]),
                           palette=palette, scale=s)

    if text_boxes:
        draw_text_boxes(out, text_boxes, palette=palette, scale=s)

    if add_legend:
        ox, oy = draw_legend(out, palette=palette, lang=lang, scale=s, meta=metadata)
        metadata["legend"] = {"x": int(ox), "y": int(oy)}

    if return_metadata:
        return out, metadata
    return out