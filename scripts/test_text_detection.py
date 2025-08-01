#!/usr/bin/env python
"""
Run text detection on input images using one of the supported detectors.

Example usage:

    python scripts/test_text_detection.py data/raw_maps/image.jpg --detector dbnet
    python scripts/test_text_detection.py data/raw_maps/*.jpg --detector easyocr --save
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

from src.text_detection import detect_text
from src.utils.image import show_image
from src.utils.io import make_output_dir, update_json


def draw_polygons(img: np.ndarray, polys: List[List[Tuple[int,int]]]) -> np.ndarray:
    out = img.copy()
    for poly in polys:
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(out, [pts], isClosed=True, color=(0,255,0), thickness=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser("Run text detection on images.")
    p.add_argument("inputs", nargs="+", help="Image files or glob patterns.")
    p.add_argument("--detector", default="dbnet",
                   choices=["dbnet","psenet","textsnake","easyocr","paddleocr"])
    p.add_argument("--device",   default="cpu",
                   help="MMOCR device ('cpu' or 'cuda:0').")

    # DBNet‐only tunables:
    p.add_argument("--binary-thresh", type=float, default=0.3,
                   help="DBNet binarization threshold")
    p.add_argument("--box-thresh",    type=float, default=0.5,
                   help="DBNet box score threshold")
    p.add_argument("--max-candidates",type=int,   default=2000,
                   help="DBNet max candidates before NMS")

    p.add_argument("--save",   action="store_true", help="Save results.")
    p.add_argument("--output", default="data/text_output",
                   help="Base dir for saving results.")
    args = p.parse_args()

    for pat in args.inputs:
        for fp in sorted(Path().glob(pat)):
            print(f"Processing {fp} with {args.detector} …")
            try:
                if args.detector == "dbnet":
                    res = detect_text(fp,
                                      detector_name="dbnet",
                                      device=args.device,
                                      binary_thresh=args.binary_thresh,
                                      box_thresh=args.box_thresh,
                                      max_candidates=args.max_candidates)
                else:
                    res = detect_text(fp,
                                      detector_name=args.detector,
                                      device=args.device)
            except Exception as e:
                print(f"Error running detector on {fp}: {e}")
                continue

            # flatten into pure polys:
            polys: List[List[Tuple[int,int]]] = []
            for item in res:
                if isinstance(item, tuple) and len(item)==2:
                    polys.append(item[0])
                else:
                    polys.append(item)

            img = cv2.imread(str(fp))
            if img is None:
                print(f"Could not load {fp}; skipping.")
                continue

            ann = draw_polygons(img, polys)
            show_image(ann, title=f"{args.detector} detection")

            if args.save:
                outdir = make_output_dir(fp, Path(args.output))
                outimg = outdir / f"text_{args.detector}.jpg"
                cv2.imwrite(str(outimg), ann)
                json_polys = [[[x,y] for x,y in poly] for poly in polys]
                update_json(outdir/"params.json",
                            {f"text_boxes_{args.detector}": json_polys})
                print(f"Saved results to {outdir}")


if __name__=="__main__":
    main()
