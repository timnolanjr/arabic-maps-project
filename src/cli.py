from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from src.pipeline_core import run_pipeline
from src.circle import interactive_detect_and_save as detect_circle
from src.edges import interactive_detect_and_save as detect_edge
from src.text_detection import detect_text_regions  # must exist in your module
from src.utils.vis import draw_text_boxes, draw_legend
from src.utils.palette import DEFAULT_PALETTE

def _path(p: str) -> Path:
    return Path(p).expanduser().resolve()

def cmd_pipeline(args: argparse.Namespace) -> None:
    run_pipeline(_path(args.input), _path(args.out), interactive=args.interactive)

def cmd_circle(args: argparse.Namespace) -> None:
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detect_circle(_path(args.image), out_dir=out_dir, interactive=args.interactive)

def cmd_edges(args: argparse.Namespace) -> None:
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detect_edge(_path(args.image), out_dir=out_dir, interactive=args.interactive)

def cmd_text(args: argparse.Namespace) -> None:
    img_path = _path(args.image)
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    boxes = detect_text_regions(
        bgr,
        method=args.method,
        nms=args.nms_filter,
        geom_filter=args.geom_filter,
        sw_filter=args.sw_filter,
        sw_threshold=args.sw_threshold,
        mser_delta=args.mser_delta,
        mser_min_area=args.mser_min_area,
        mser_max_area=args.mser_max_area,
        mser_max_variation=args.mser_max_variation,
        mser_min_diversity=args.mser_min_diversity,
    )

    overlay = bgr.copy()
    draw_text_boxes(overlay, boxes, palette=DEFAULT_PALETTE)
    draw_legend(overlay, palette=DEFAULT_PALETTE)
    out_path = out_dir / f"{img_path.stem}_text_{args.method}.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f" → wrote {out_path}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arabmaps", description="Arabic circular maps toolkit CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pipeline", help="Run geometry pipeline on a file or directory")
    sp.add_argument("input", help="Image file or directory")
    sp.add_argument("-o", "--out", default="data/processed_maps", help="Output base directory")
    sp.add_argument("--interactive", action="store_true", help="Interactive circle/edge picking")
    sp.set_defaults(func=cmd_pipeline)

    sc = sub.add_parser("circle", help="Circle detection (interactive)")
    sc.add_argument("image", help="Image file")
    sc.add_argument("-o", "--out", default="data/processed_maps", help="Output directory")
    sc.add_argument("--interactive", action="store_true")
    sc.set_defaults(func=cmd_circle)

    se = sub.add_parser("edges", help="Edge detection (interactive)")
    se.add_argument("image", help="Image file")
    se.add_argument("-o", "--out", default="data/processed_maps", help="Output directory")
    se.add_argument("--interactive", action="store_true")
    se.set_defaults(func=cmd_edges)

    st = sub.add_parser("text", help="Run a text detector and save an overlay with boxes")
    st.add_argument("image", help="Image file")
    st.add_argument("-o", "--out", default="data/processed_maps", help="Output directory")
    st.add_argument("--method", default="mser", choices=["morph", "mser", "canny", "sobel", "gradient"])
    st.add_argument("--nms-filter", dest="nms_filter", action="store_true")
    st.add_argument("--geom-filter", action="store_true")
    st.add_argument("--sw-filter", action="store_true")
    st.add_argument("--sw-threshold", type=float, default=0.4)
    st.add_argument("--mser-delta", type=int, default=5)
    st.add_argument("--mser-min-area", type=int, default=60)
    st.add_argument("--mser-max-area", type=int, default=14400)
    st.add_argument("--mser-max-variation", type=float, default=0.3)
    st.add_argument("--mser-min-diversity", type=float, default=0.2)
    st.set_defaults(func=cmd_text)

    return p

def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
