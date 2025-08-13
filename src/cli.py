from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys
import cv2
from collections import Counter

from src.text.config import TextDetectConfig, MserParams, GeometryFilter, CircleFilter, NmsConfig, VisualConfig
from src.text_detection import detect_text_regions
from src.utils.palette import DEFAULT_PALETTE
from src.utils.naming import next_runpair_paths

def _path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def _parse_scales(s: str | None) -> tuple[float, ...]:
    if not s: return ()
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return tuple(out)

def _fmt_scales(enabled: bool, scales: tuple[float, ...]) -> str:
    if not enabled or not scales:
        return "1.0"
    return ",".join(f"{s:.1f}" for s in scales)

def _params_title(cfg: TextDetectConfig, run_id: str | None) -> str:
    lines = [
        f"MSER delta={cfg.mser.delta} area=[{cfg.mser.min_area},{cfg.mser.max_area}]",
        f"polarity={cfg.mser.polarity}",
        f"scales={_fmt_scales(cfg.multiscale.enabled, tuple(cfg.multiscale.scales))}",
        f"geom={'on' if cfg.geometry.enabled else 'off'}",
        "nms=" + ("on IoU={:.2f}".format(cfg.nms.iou) if cfg.nms.enabled else "off"),
        "circle=" + ("on minCover={:.2f}".format(cfg.circle.min_cover) if cfg.circle.enabled else "off"),
    ]
    if getattr(cfg, "morph", None) and cfg.morph.enabled:
        lines.append(f"morph=close{cfg.morph.close_ksize}/open{cfg.morph.open_ksize} x{cfg.morph.iterations}")
    if run_id:
        lines.append(f"run={run_id}")
    # visualize.py splits on " | " → each item becomes its own line
    return " | ".join(lines)


def cmd_text(args: argparse.Namespace) -> None:
    img_path = _path(args.image)
    base_out = _path(args.out)
    out_dir = base_out / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {img_path}")

    # Build config from CLI
    cfg = TextDetectConfig(
        method=args.method,
        mser=MserParams(
            delta=args.mser_delta,
            min_area=args.mser_min_area,
            max_area=args.mser_max_area,
            max_variation=args.mser_max_variation,
            min_diversity=args.mser_min_diversity,
            max_evolution=args.mser_max_evolution,
            area_threshold=args.mser_area_threshold,
            min_margin=args.mser_min_margin,
            edge_blur_size=args.mser_edge_blur_size,
            polarity=args.polarity,
        ),
        geometry=GeometryFilter(
            enabled=not args.no_geom_filter,
            area_pct=(args.geom_area_min_pct, args.geom_area_max_pct),
            aspect_ratio=(args.geom_ar_min, args.geom_ar_max),
        ),
        circle=CircleFilter(
            enabled=args.circle_filter,
            min_cover=args.circle_min_cover,
        ),
        nms=NmsConfig(
            enabled=not args.no_nms,
            iou=args.nms_iou,
        ),
        visual=VisualConfig(
            include_removed=not args.no_removed,
            scale=args.overlay_scale,
            show_labels=args.overlay_labels,
            box_thickness=args.overlay_box_thickness,
        ),
    )

    # multiscale
    scales = _parse_scales(args.scales)
    if args.multiscale or scales:
        cfg.multiscale.enabled = True
        cfg.multiscale.scales = scales if scales else (1.0, 0.75, 0.5)

    # morphology
    if args.morph:
        cfg.morph.enabled = True
        cfg.morph.close_ksize = args.morph_close
        cfg.morph.open_ksize = args.morph_open
        cfg.morph.iterations = args.morph_iter

    circle_center = circle_radius = None
    if args.circle_filter and args.params:
        params = json.loads(Path(args.params).read_text(encoding="utf-8"))
        circle_center = (int(params["center_x"]), int(params["center_y"]))
        circle_radius = int(params["radius"])

    # Title to stamp on overlay (ASCII-safe)
    title = _params_title(cfg, args.run_id) if args.stamp_params else None

    # Run detection
    kept_boxes, overlay, records, summary = detect_text_regions(
        bgr, cfg,
        circle_center=circle_center,
        circle_radius=circle_radius,
        img_path=img_path,
        params_path=Path(args.params) if args.params else None,
        out_jsonl=None,                 # we’re not writing JSONL in this paired mode
        run_id=args.run_id,
        return_debug=True,              # need summary for runmeta JSON
        overlay_title=title,
        palette=DEFAULT_PALETTE,
    )

    # Choose paired filenames (jpg + json) with the SAME index
    img_out, runmeta_out, idx = next_runpair_paths(
        out_dir=out_dir,
        stem=img_path.stem,
        method=args.method,
        img_ext=".jpg",
        json_ext=".json",
        width=4,
        start=1,
    )

    # Save overlay JPG
    ok = cv2.imwrite(str(img_out), overlay if overlay is not None else bgr)
    if not ok:
        # fallback encode
        ok2, buf = cv2.imencode(".jpg", overlay if overlay is not None else bgr)
        if not ok2:
            raise SystemExit(f"Failed to encode overlay for: {img_out}")
        with open(img_out, "wb") as f:
            f.write(buf.tobytes())
    print(f" → overlay: {img_out}")

    # Save run metadata JSON (always, so it matches the JPG)
    runmeta = {
        "image": str(img_path),
        "output_overlay": str(img_out),
        "run_id": args.run_id,
        "config": asdict(cfg),
        "summary": summary,
    }
    runmeta_out.write_text(json.dumps(runmeta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" → runmeta: {runmeta_out}")

    print(f"Kept {len(kept_boxes)} boxes")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arabmaps", description="Arabic circular maps toolkit CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- text subparser (updated) ---
    st = sub.add_parser("text", help="Run a text detector and save an overlay with boxes")
    st.add_argument("image", help="Image file")
    st.add_argument("-o", "--out", default="data/processed_maps", help="Output directory")
    st.add_argument("--method", default="mser", choices=["mser"])

    # Polarity / multiscale / morphology
    st.add_argument("--polarity", default="both", choices=["dark", "bright", "both"], help="MSER polarity")
    st.add_argument("--multiscale", action="store_true", help="Enable multiscale pyramid (use --scales or defaults)")
    st.add_argument("--scales", help="Comma-separated scales, e.g. '1.0,0.75,0.5'")
    st.add_argument("--morph", action="store_true", help="Enable morphology merge (close→open)")
    st.add_argument("--morph-close", type=int, default=3)
    st.add_argument("--morph-open", type=int, default=0)
    st.add_argument("--morph-iter", type=int, default=1)

    # MSER params
    st.add_argument("--mser-delta", type=int, default=5)
    st.add_argument("--mser-min-area", type=int, default=60)
    st.add_argument("--mser-max-area", type=int, default=14400)
    st.add_argument("--mser-max-variation", type=float, default=0.3)
    st.add_argument("--mser-min-diversity", type=float, default=0.2)
    st.add_argument("--mser-max-evolution", type=int, default=200)
    st.add_argument("--mser-area-threshold", type=float, default=1.01)
    st.add_argument("--mser-min-margin", type=float, default=0.003)
    st.add_argument("--mser-edge-blur-size", type=int, default=3)

    # Geometry filter
    st.add_argument("--no-geom-filter", action="store_true", help="Disable geometry filter")
    st.add_argument("--geom-area-min-pct", type=float, default=0.00005)
    st.add_argument("--geom-area-max-pct", type=float, default=0.02)
    st.add_argument("--geom-ar-min", type=float, default=0.15)
    st.add_argument("--geom-ar-max", type=float, default=8.0)

    # Circle filter
    st.add_argument("--circle-filter", action="store_true", help="Enable circle coverage filter")
    st.add_argument("--circle-min-cover", type=float, default=0.5)
    st.add_argument("--params", help="params.json to read circle (center_x,center_y,radius)")

    # NMS
    st.add_argument("--no-nms", action="store_true", help="Disable NMS")
    st.add_argument("--nms-iou", type=float, default=0.3)

    # Visual
    st.add_argument("--no-removed", action="store_true", help="Hide removed boxes in overlay")
    st.add_argument("--overlay-scale", type=float, help="Force overlay scale (e.g., 1.5)")
    st.add_argument("--overlay-labels", action="store_true", help="Draw status labels on boxes")
    st.add_argument("--overlay-box-thickness", type=int, default=3, help="Box outline thickness in pixels")

    # Output / run tracking
    st.add_argument("--save-jsonl", action="store_true", help="Write per-box JSONL")
    st.add_argument("--save-runmeta", action="store_true", help="Write a compact JSON with config + counts")
    st.add_argument("--stamp-params", action="store_true", help="Stamp a one-line param summary onto the overlay (TR)")
    st.add_argument("--run-id", default=None)

    st.set_defaults(func=cmd_text)
    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()