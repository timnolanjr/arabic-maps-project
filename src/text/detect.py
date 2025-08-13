from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import hashlib
import cv2
import numpy as np

from src.text.backends.mser import detect_mser_regions
from src.text.config import TextDetectConfig
from src.text.visualize import render_stage_overlay
from src.utils.palette import DEFAULT_PALETTE, Palette

def _sha256_file(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def _to_boxes(records: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
    return [tuple(map(int, r["bbox"])) for r in records if r.get("status") == "kept"]

def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def detect_text_regions(
    bgr: np.ndarray,
    cfg: TextDetectConfig,
    *,
    circle_center: Optional[Tuple[int, int]] = None,   # unused when filters off
    circle_radius: Optional[int] = None,               # unused when filters off
    img_path: Optional[Path] = None,
    params_path: Optional[Path] = None,
    out_jsonl: Optional[Path] = None,
    run_id: Optional[str] = None,
    return_debug: bool = False,
    return_overlay: bool = False,
    palette: Palette = DEFAULT_PALETTE,
):
    """
    Minimal MSER orchestrator. With filters disabled this:
      1) runs MSER,
      2) marks all boxes as 'kept',
      3) (optionally) writes JSONL,
      4) (optionally) returns an overlay image.
    """
    # --- 1) backend ---
    mp = cfg.mser
    regions = detect_mser_regions(
        bgr,
        delta=mp.delta, min_area=mp.min_area, max_area=mp.max_area,
        max_variation=mp.max_variation, min_diversity=mp.min_diversity,
        max_evolution=mp.max_evolution, area_threshold=mp.area_threshold,
        min_margin=mp.min_margin, edge_blur_size=mp.edge_blur_size,
    )

    # --- 2) records (no filters -> everything kept) ---
    records: List[Dict[str, Any]] = []
    for i, r in enumerate(regions):
        x, y, w, h = r.bbox
        bbox_area = max(1, w * h)
        score = min(1.0, r.area / float(bbox_area))   # extent proxy
        records.append({
            "id": f"mser-{i:06d}",
            "bbox": [int(x), int(y), int(w), int(h)],
            "score": float(score),
            "status": "kept",
            "stage": "final",
            "reason": "ok",
            "method": "mser",
            "method_version": cv2.__version__,
            "coord_space": "image_native",
        })

    # --- 3) optional JSONL ---
    if out_jsonl is not None and img_path is not None and params_path is not None:
        img_sha = _sha256_file(img_path)
        params_sha = _sha256_file(params_path)
        for r in records:
            r.setdefault("image_sha256", img_sha)
            r.setdefault("params_sha256", params_sha)
            if run_id: r.setdefault("run_id", run_id)
        _write_jsonl(out_jsonl, records)

    # --- 4) optional overlay ---
    overlay_img = None
    if return_debug or return_overlay or cfg.visual.include_removed or cfg.visual.show_labels or cfg.visual.scale is not None:
        overlay_img = render_stage_overlay(
            bgr, records,
            include_removed=cfg.visual.include_removed,
            scale=cfg.visual.scale,
            show_labels=cfg.visual.show_labels,
            palette=palette,
        )

    kept_boxes = _to_boxes(records)
    if return_debug:
        return kept_boxes, overlay_img, records
    if return_overlay:
        return kept_boxes, overlay_img
    return kept_boxes
