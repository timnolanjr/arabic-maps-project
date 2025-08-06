# scripts/test_text_detection.py

#!/usr/bin/env python3
"""
Command-line tool for text detection.  --apply-nms is OFF by default.
"""

from __future__ import annotations
import argparse, sys, glob
from pathlib import Path

import cv2
from tqdm import tqdm

# ensure src on path
_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from src.text_detection import (
    detect_text_morphology,
    detect_text_mser,
    detect_text_canny,
    detect_text_sobel,
    detect_text_gradient,
    draw_bounding_boxes,
)

METHODS = {
    "morph": detect_text_morphology,
    "mser": detect_text_mser,
    "canny": detect_text_canny,
    "sobel": detect_text_sobel,
    "gradient": detect_text_gradient,
}


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect text in images")
    p.add_argument("input", help="Path or glob to images")
    p.add_argument("--method", choices=METHODS, default="morph")
    p.add_argument("-o", "--output-dir", default="data/processed_maps")
    p.add_argument("--interactive", action="store_true")
    # MSER params
    p.add_argument("--mser-delta",          type=int,   default=5)
    p.add_argument("--mser-min-area",       type=int,   default=60)
    p.add_argument("--mser-max-area",       type=int,   default=14400)
    p.add_argument("--mser-max-variation",  type=float, default=0.3)
    p.add_argument("--mser-min-diversity",  type=float, default=0.2)
    p.add_argument("--mser-max-evolution",  type=int,   default=1000)
    p.add_argument("--mser-area-threshold", type=float, default=1.01)
    p.add_argument("--mser-min-margin",     type=float, default=0.003)
    p.add_argument("--mser-edge-blur-size", type=int,   default=3)
    # Optional filters
    p.add_argument("--mser-geom-filter", action="store_true")
    p.add_argument("--mser-sw-filter",   action="store_true")
    p.add_argument("--mser-sw-threshold", type=float, default=0.4)
    # NEW: apply NMS only if passed (default off)
    p.add_argument("--nms-filter", action="store_true",
                   help="Perform non-max suppression on final boxes")
    return p.parse_args()


def process_image(path: Path, args: argparse.Namespace) -> cv2.Mat:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")

    nms_filter = args.nms_filter

    if args.method == "mser":
        boxes = detect_text_mser(
            img,
            delta=args.mser_delta,
            min_area=args.mser_min_area,
            max_area=args.mser_max_area,
            max_variation=args.mser_max_variation,
            min_diversity=args.mser_min_diversity,
            max_evolution=args.mser_max_evolution,
            area_threshold=args.mser_area_threshold,
            min_margin=args.mser_min_margin,
            edge_blur_size=args.mser_edge_blur_size,
            geom_filter=args.mser_geom_filter,
            sw_filter=args.mser_sw_filter,
            sw_threshold=args.mser_sw_threshold,
            nms_filter=nms_filter,
        )
    else:
        detector = METHODS[args.method]
        boxes = detector(img, nms_filter=nms_filter)

    return draw_bounding_boxes(img, boxes)


def main() -> None:
    args = parse_arguments()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = glob.glob(args.input)
    if not imgs:
        print(f"No files matched: {args.input}", file=sys.stderr)
        sys.exit(1)

    for fn in tqdm(imgs, desc=f"Running {args.method}"):
        p = Path(fn)
        try:
            out_img = process_image(p, args)
        except Exception as e:
            print(f"Error on {p.name}: {e}", file=sys.stderr)
            continue

        if args.interactive:
            cv2.imshow(p.name, out_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            d = out_root / p.stem
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"{p.stem}_{args.method}.jpg"), out_img)


if __name__ == "__main__":
    main()
