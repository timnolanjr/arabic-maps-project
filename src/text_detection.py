from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import asdict
from collections import Counter
import json
import hashlib
import datetime as _dt

import cv2
import numpy as np

from src.text.backends.mser import detect_mser_regions
from src.text.config import TextDetectConfig
from src.text.visualize import render_stage_overlay
from src.utils.palette import DEFAULT_PALETTE, Palette

# Filters
from src.text.filters.geometry import GeometryConfig, geometry_filter_boxes
from src.text.filters.circle import circle_coverage_filter
from src.text.filters.nms import nms_filter
from src.text.filters.morphology import boxes_to_mask, morph_merge_mask, mask_to_boxes


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _sha256_file(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _to_boxes(records: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
    return [tuple(map(int, r["bbox"])) for r in records if r.get("status") == "kept"]


def _resize_boxes(boxes: List[Tuple[int, int, int, int]], scale: float) -> List[Tuple[int, int, int, int]]:
    if abs(scale - 1.0) < 1e-6:
        return boxes
    inv = 1.0 / float(scale)
    out: List[Tuple[int, int, int, int]] = []
    for x, y, w, h in boxes:
        out.append((
            int(round(x * inv)),
            int(round(y * inv)),
            int(round(w * inv)),
            int(round(h * inv)),
        ))
    return out


def _run_mser_variants(bgr: np.ndarray, cfg: TextDetectConfig) -> List[Tuple[int, int, int, int]]:
    """
    Run MSER over requested polarities and scales; return UNION of boxes
    projected back to full-res coordinates.
    """
    polar = cfg.mser.polarity.lower()
    polarities = [polar] if polar in ("dark", "bright") else ("dark", "bright")
    scales = tuple(cfg.multiscale.scales) if getattr(cfg, "multiscale", None) and cfg.multiscale.enabled else (1.0,)

    mp = cfg.mser
    boxes_all: List[Tuple[int, int, int, int]] = []

    for s in scales:
        if s != 1.0:
            img_s = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            img_s = bgr

        for pol in polarities:
            gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
            if pol == "bright":
                gray = cv2.bitwise_not(gray)
            img_for_backend = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            regions = detect_mser_regions(
                img_for_backend,
                delta=mp.delta,
                min_area=mp.min_area,
                max_area=mp.max_area,
                max_variation=mp.max_variation,
                min_diversity=mp.min_diversity,
                max_evolution=mp.max_evolution,
                area_threshold=mp.area_threshold,
                min_margin=mp.min_margin,
                edge_blur_size=mp.edge_blur_size,
            )
            boxes = [r.bbox for r in regions]
            boxes = _resize_boxes(boxes, s)
            boxes_all.extend(boxes)

    return boxes_all


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def detect_text_regions(
    bgr: np.ndarray,
    cfg: TextDetectConfig,
    *,
    circle_center: Optional[Tuple[int, int]] = None,
    circle_radius: Optional[int] = None,
    img_path: Optional[Path] = None,       # for JSONL metadata (optional)
    params_path: Optional[Path] = None,    # for JSONL metadata (optional)
    out_jsonl: Optional[Path] = None,      # write JSONL if provided (optional)
    run_id: Optional[str] = None,
    return_debug: bool = False,            # if True → (boxes, overlay, records, summary)
    return_overlay: bool = False,          # if True → (boxes, overlay)
    overlay_title: Optional[str] = None,   # draw this title on the overlay (TR)
    palette: Palette = DEFAULT_PALETTE,
):
    """
    MSER-based text region detection with optional:
      - polarity ('dark'/'bright'/'both')
      - multi-scale pyramid
      - morphology merge (close/open on a union mask)
      - geometry filter (area/aspect)
      - circle coverage filter
      - NMS

    Returns:
      - if return_debug:  (kept_boxes, overlay_bgr, records, summary)
      - elif return_overlay: (kept_boxes, overlay_bgr)
      - else: kept_boxes
    """
    H, W = bgr.shape[:2]
    img_area = float(H * W)

    # --- 1) Candidates via MSER variants (polarity & scales) ---
    boxes0 = _run_mser_variants(bgr, cfg)

    # --- 1b) Optional morphology merge on union mask ---
    if getattr(cfg, "morph", None) and cfg.morph.enabled:
        mask = boxes_to_mask(boxes0, (H, W))
        mask = morph_merge_mask(mask, cfg.morph.close_ksize, cfg.morph.open_ksize, cfg.morph.iterations)
        boxes0 = mask_to_boxes(mask)

    # --- Seed records (status=candidate) ---
    records: List[Dict[str, Any]] = []
    for i, (x, y, w, h) in enumerate(boxes0):
        records.append({
            "id": f"mser-{i:06d}",
            "bbox": [int(x), int(y), int(w), int(h)],
            "score": 1.0,
            "status": "candidate",
            "stage": "mser",
            "method": "mser",
            "method_version": cv2.__version__,
            "coord_space": "image_native",
        })

    # --- 2) Geometry filter (area/aspect) ---
    if cfg.geometry.enabled:
        g_min_px = int(cfg.geometry.area_pct[0] * img_area)
        g_max_px = int(cfg.geometry.area_pct[1] * img_area)
        kept_idx, removed_idx = geometry_filter_boxes(
            [tuple(map(int, r["bbox"])) for r in records],
            GeometryConfig(area_px=(g_min_px, g_max_px), aspect_ratio=cfg.geometry.aspect_ratio),
        )
        for idx in removed_idx:
            records[idx]["status"] = "removed_geom"
            records[idx]["stage"] = "geom"
            records[idx]["reason"] = "geometry"
        for idx in kept_idx:
            if records[idx]["status"] == "candidate":
                records[idx]["status"] = "post_geom"
                records[idx]["stage"] = "geom"
    else:
        for r in records:
            if r["status"] == "candidate":
                r["status"] = "post_geom"
                r["stage"] = "geom"

    # --- 3) Circle coverage filter ---
    if cfg.circle.enabled and circle_center is not None and circle_radius is not None:
        survivors = [i for i, r in enumerate(records) if r["status"].startswith("post_")]
        if survivors:
            boxes = [tuple(map(int, records[i]["bbox"])) for i in survivors]
            kept_local, removed_local = circle_coverage_filter(
                boxes,
                center=circle_center,
                radius=int(circle_radius),
                min_cover=cfg.circle.min_cover,
            )
            kept_idx = [survivors[k] for k in kept_local]
            removed_idx = [survivors[k] for k in removed_local]
            for idx in removed_idx:
                if records[idx]["status"].startswith("post_"):
                    records[idx]["status"] = "removed_circle"
                    records[idx]["stage"] = "circle"
                    records[idx]["reason"] = f"circle_min_cover<{cfg.circle.min_cover}"
            for idx in kept_idx:
                if records[idx]["status"].startswith("post_"):
                    records[idx]["status"] = "post_circle"
                    records[idx]["stage"] = "circle"
    else:
        for r in records:
            if r["status"].startswith("post_"):
                r["status"] = "post_circle"
                r["stage"] = "circle"

    # --- 4) NMS ---
    survivors = [i for i, r in enumerate(records) if r["status"].startswith("post_")]
    kept_after_nms_idx: List[int] = []
    if cfg.nms.enabled and survivors:
        boxes = np.array([records[i]["bbox"] for i in survivors], dtype=np.float32)
        scores = np.array([records[i]["score"] for i in survivors], dtype=np.float32)
        keep_local, suppr_local = nms_filter(boxes, scores, iou_thresh=cfg.nms.iou)
        kept_after_nms_idx = [survivors[k] for k in keep_local]
        for s in suppr_local:
            idx = survivors[s]
            records[idx]["status"] = "removed_nms"
            records[idx]["stage"] = "nms"
            records[idx]["reason"] = f"iou>{cfg.nms.iou}"
    else:
        kept_after_nms_idx = survivors

    for idx in kept_after_nms_idx:
        records[idx]["status"] = "kept"
        records[idx]["stage"] = "final"
        records[idx]["reason"] = "ok"

    # --- 5) Optional JSONL export ---
    if out_jsonl is not None and img_path is not None and params_path is not None:
        img_sha = _sha256_file(img_path)
        params_sha = _sha256_file(params_path)
        for r in records:
            r.setdefault("image_sha256", img_sha)
            r.setdefault("params_sha256", params_sha)
            if run_id:
                r.setdefault("run_id", run_id)
        _write_jsonl(out_jsonl, records)

    # --- 6) Optional overlay ---
    overlay_img = None
    if return_debug or return_overlay or cfg.visual.include_removed or cfg.visual.show_labels or cfg.visual.scale is not None:
        overlay_img = render_stage_overlay(
            bgr,
            records,
            include_removed=cfg.visual.include_removed,
            scale=cfg.visual.scale,
            show_labels=cfg.visual.show_labels,
            palette=palette,
            box_thickness=cfg.visual.box_thickness,
            title=overlay_title,
            title_loc="tr",
        )

    kept_boxes = _to_boxes(records)

    if return_debug:
        # Build a concise summary for run-metadata
        by_status = dict(Counter(r["status"] for r in records))
        by_stage  = dict(Counter(r["stage"] for r in records))
        summary = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "total": len(records),
            "kept": len(kept_boxes),
            "by_status": by_status,
            "by_stage": by_stage,
        }
        return kept_boxes, overlay_img, records, summary

    if return_overlay:
        return kept_boxes, overlay_img

    return kept_boxes