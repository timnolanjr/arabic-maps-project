#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.edges import (
    batch_detect_and_save,
    interactive_detect_and_save,
)

def main():
    p = argparse.ArgumentParser(
        description="Detect top horizontal edge (batch or interactive)"
    )
    p.add_argument("input", help="Path to input image")
    p.add_argument(
        "-o", "--output-dir",
        default="data/processed_maps",
        help="Base output directory"
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Use ROI‐based interactive detection"
    )
    p.add_argument(
        "--top-k", type=int, default=10,
        help="How many Hough peaks to fetch"
    )
    p.add_argument(
        "--delta", type=float, default=0.05,
        help="±fraction of image height around clicks (interactive only)"
    )
    p.add_argument(
        "--min-angle", type=float, default=80.0,
        help="Minimum normal angle (deg) for horizontal"
    )
    p.add_argument(
        "--max-angle", type=float, default=110.0,
        help="Maximum normal angle (deg) for horizontal"
    )
    args = p.parse_args()

    img_path = Path(args.input)
    out_base = Path(args.output_dir)

    if args.interactive:
        interactive_detect_and_save(
            input_path=img_path,
            base_output_dir=out_base,
            n_clicks=3,
            delta=args.delta,
            top_k=args.top_k,
            min_angle_deg=args.min_angle,
            max_angle_deg=args.max_angle
        )
    else:
        batch_detect_and_save(
            input_path=img_path,
            base_output_dir=out_base,
            top_k=args.top_k,
            min_angle_deg=args.min_angle,
            max_angle_deg=args.max_angle
        )

if __name__ == "__main__":
    main()
