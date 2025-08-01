"""
test_text_detecy.py
====================

This script provides a simple command-line interface for running the
text detection functions defined in ``text_detect.py`` via TextDetInferencer.
You can use it to experiment with different detectors on one or more images
and optionally save the annotated results.

Example usage::

    python test_text_detecy.py image.jpg --detector dbnet --save
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

from src.text_detection import TextDetInferencer


def draw_polygons(img: np.ndarray, polys: List[List[Tuple[int, int]]]) -> np.ndarray:
    """Return a copy of ``img`` with polygons drawn in green."""
    out = img.copy()
    for poly in polys:
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return out


def make_output_dir(input_path: Path, base_output: Path) -> Path:
    """Create an output directory based on the input file name."""
    outdir = base_output / input_path.stem
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def update_json(path: Path, data: dict[str, Any]) -> None:
    """Update a JSON file with new data (merging into existing file if present)."""
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            existing = {}
    else:
        existing = {}
    existing.update(data)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run text detection on images using TextDetInferencer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files or glob patterns (e.g. 'image.jpg' or '*.png').",
    )
    parser.add_argument(
        "--detector",
        default="dbnet",
        choices=list(TextDetInferencer.__init__.__defaults__ or []),  # ignored, help only
        help="Which detector to use.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device string (e.g. 'cpu' or 'cuda:0')."
    )
    # DBNet tunables
    parser.add_argument("--binary-thresh", type=float, default=0.3, help="DBNet binarization threshold")
    parser.add_argument("--box-thresh",    type=float, default=0.5, help="DBNet box score threshold")
    parser.add_argument("--max-candidates", type=int, default=2000, help="DBNet max candidates before NMS")
    # EAST tunables
    parser.add_argument("--east-model",   default="frozen_east_text_detection.pb", help="Path to EAST .pb model")
    parser.add_argument("--east-conf",    type=float, default=0.5, help="EAST confidence threshold")
    parser.add_argument("--east-nms",     type=float, default=0.4, help="EAST NMS threshold")
    # docTR tunable
    parser.add_argument("--doctr-arch", default="db_resnet50", help="docTR detection architecture")
    # Optional outputs
    parser.add_argument("--save", action="store_true", help="Save annotated images and JSON")
    parser.add_argument("--output", default="text_output", help="Directory for saving results")
    args = parser.parse_args(argv)

    # Expand inputs
    all_files: List[Path] = []
    for pattern in args.inputs:
        if any(c in pattern for c in "*?[]"):
            all_files.extend(sorted(Path().glob(pattern)))
        else:
            all_files.append(Path(pattern))
    if not all_files:
        print("No input files found.")
        return

    # Build inferencer with all possible kwargs (unused ones are ignored)
    detector_kwargs: dict[str, Any] = {
        "device": args.device,
        "binary_thresh":    args.binary_thresh,
        "box_thresh":       args.box_thresh,
        "max_candidates":   args.max_candidates,
        "model_path":       args.east_model,
        "conf_threshold":   args.east_conf,
        "nms_threshold":    args.east_nms,
        "arch":             args.doctr_arch,
    }
    infer = TextDetInferencer(args.detector, **detector_kwargs)

    for fp in all_files:
        if not fp.is_file():
            print(f"Skipping {fp}: not a file")
            continue
        print(f"Processing {fp} with {args.detector} …")
        try:
            polys = infer.detect(fp)
        except Exception as e:
            print(f"Error running detector on {fp}: {e}")
            continue

        # Load and annotate
        img = cv2.imread(str(fp))
        if img is None:
            print(f"Could not load {fp}; skipping.")
            continue
        annotated = draw_polygons(img, polys)

        # Display (if possible)
        try:
            cv2.imshow(f"{args.detector} – {fp.name}", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass  # headless environments

        # Save if requested
        if args.save:
            base_out = Path(args.output)
            outdir = make_output_dir(fp, base_out)
            outimg = outdir / f"text_{args.detector}.jpg"
            cv2.imwrite(str(outimg), annotated)

            # Convert to nested lists for JSON
            json_polys = [[[x, y] for x, y in poly] for poly in polys]
            update_json(outdir / "params.json", {f"text_boxes_{args.detector}": json_polys})
            print(f"Saved results to {outdir}")

if __name__ == "__main__":
    main()
