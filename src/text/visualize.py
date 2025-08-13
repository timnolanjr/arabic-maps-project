from __future__ import annotations
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from src.utils.palette import Palette, DEFAULT_PALETTE

def _auto_scale(shape) -> float:
    h, w = shape[:2]
    s = max(h, w) / 1200.0
    return float(max(0.75, min(6.0, s)))

def _t(px: int, s: float) -> int:
    return max(1, int(round(px * s)))

def _draw_title(
    img: np.ndarray,
    text: str,
    *,
    s: float,
    loc: str = "tr",
    pad_px: int = 10,
    bg_color=(0, 0, 0),
    fg_color=(240, 240, 240),
    alpha: float = 0.5,
) -> None:
    if not text:
        return
    pad = _t(pad_px, s)
    fs = _t(7, s) / 10.0
    ft = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    # allow simple multi-line with " | " split
    lines = [t.strip() for t in text.split(" | ") if t.strip()]
    widths = []
    heights = []
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, fs, ft)
        widths.append(w); heights.append(h)
    block_w = pad * 2 + max(widths or [0])
    block_h = pad * 2 + sum(heights) + _t(6, s) * max(0, len(lines) - 1)

    H, W = img.shape[:2]
    if loc.lower() == "tr":
        x0 = W - block_w - pad
        y0 = pad
    elif loc.lower() == "tl":
        x0 = pad
        y0 = pad
    elif loc.lower() == "br":
        x0 = W - block_w - pad
        y0 = H - block_h - pad
    else:  # "bl"
        x0 = pad
        y0 = H - block_h - pad

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + block_w, y0 + block_h), bg_color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    y = y0 + pad + heights[0] if heights else y0 + pad
    for line in lines:
        cv2.putText(img, line, (x0 + pad, y), font, fs, fg_color, 1, cv2.LINE_AA)
        y += heights[0] + _t(6, s)

def render_stage_overlay(
    bgr: np.ndarray,
    records: List[Dict[str, Any]],
    *,
    include_removed: bool = True,
    scale: Optional[float] = None,
    show_labels: bool = False,
    palette: Palette = DEFAULT_PALETTE,
    box_thickness: Optional[int] = None,
    title: Optional[str] = None,
    title_loc: str = "tr",
) -> np.ndarray:
    out = bgr.copy()
    s = _auto_scale(out.shape) if scale is None else float(scale)
    th = int(box_thickness) if box_thickness is not None else _t(2, s)

    def color_for(status: str):
        if status == "kept": return palette.text_kept.bgr
        if status == "removed_nms": return palette.text_removed_nms.bgr
        if status == "removed_circle": return palette.text_removed_circle.bgr
        if status == "removed_geom": return palette.text_removed_geom.bgr
        return palette.text_candidate.bgr

    counts = {"kept": 0, "removed_nms": 0, "removed_circle": 0, "removed_geom": 0}
    for r in records:
        st = r.get("status", "kept")
        if st != "kept" and not include_removed:
            continue
        x, y, w, h = map(int, r["bbox"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color_for(st), th, cv2.LINE_AA)
        counts[st] = counts.get(st, 0) + 1
        if show_labels:
            cv2.putText(out, st.replace("_", " "), (x, max(0, y - _t(4, s))),
                        cv2.FONT_HERSHEY_SIMPLEX, _t(6, s)/10.0, color_for(st), 1, cv2.LINE_AA)

    # Legend (UL)
    pad = _t(10, s); lh = _t(22, s); fs = _t(6, s)/10.0; ft = 1; font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"kept: {counts.get('kept',0)}", palette.text_kept.bgr),
        (f"removed_nms: {counts.get('removed_nms',0)}", palette.text_removed_nms.bgr),
        (f"removed_circle: {counts.get('removed_circle',0)}", palette.text_removed_circle.bgr),
        (f"removed_geom: {counts.get('removed_geom',0)}", palette.text_removed_geom.bgr),
    ]
    maxw = 0
    for txt, _ in lines:
        (w, _), _ = cv2.getTextSize(txt, font, fs, ft); maxw = max(maxw, w)
    width = pad * 2 + _t(18, s) + _t(10, s) + maxw
    height = pad * 2 + lh * len(lines)
    x0, y0 = pad, pad
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), palette.legend_bg.bgr, thickness=-1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
    for i, (txt, color) in enumerate(lines):
        yy = y0 + pad + i * lh
        cv2.circle(out, (x0 + pad, yy + _t(8, s)), _t(6, s), color, -1, cv2.LINE_AA)
        cv2.putText(out, txt, (x0 + pad + _t(18, s) + _t(6, s), yy + _t(10, s)),
                    font, fs, (240, 240, 240), 1, cv2.LINE_AA)

    # Optional title (e.g., param summary) — top-right by default
    if title:
        _draw_title(out, title, s=s, loc=title_loc, bg_color=palette.legend_bg.bgr)

    return out