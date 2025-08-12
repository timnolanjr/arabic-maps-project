from __future__ import annotations

from pathlib import Path
import json
import cv2
from typing import Tuple, Dict, Any

from .palette import DEFAULT_PALETTE, Palette
from .vis import compose_geometry_overlay

def render_overlay_from_paths(
    img_path: Path | str,
    params_json_path: Path | str,
    out_path: Path | str,
    *,
    palette: Palette = DEFAULT_PALETTE,
    include_legend: bool = True,
    include_text_boxes: bool = True,
    lang: str | None = None,
    unicode_glyphs: bool | None = None,  # back-compat: if True and lang is None, use 'ar'
    scale: float | None = None,
    return_metadata: bool = False,
):
    """
    Load image + params.json and write an annotated overlay.
    If return_metadata=True, returns (out_path, metadata_dict); else returns out_path.
    """
    img_path = Path(img_path)
    params_json_path = Path(params_json_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    with open(params_json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    text_boxes = params.get("text_boxes") if include_text_boxes else None
    if lang is None and unicode_glyphs:
        lang = "ar"
    lang = lang or "en"

    overlay, meta = compose_geometry_overlay(
        bgr,
        params,
        palette=palette,
        text_boxes=text_boxes,
        add_legend=include_legend,
        lang=lang,
        scale=scale,
        return_metadata=True,   # always get it here; we decide what to return below
    )

    cv2.imwrite(str(out_path), overlay)

    if return_metadata:
        return out_path, meta
    return out_path
