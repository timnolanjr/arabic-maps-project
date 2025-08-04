#!/usr/bin/env python3
"""Command-line tool for text detection.

This script exposes the simple text detection algorithms defined in
``src/text_detection.py`` via a convenient command-line interface.  Given an
image (or a glob of images), it will detect text regions using the
selected method and optionally save annotated outputs.  Users can run
the script in interactive mode to display results on screen before
saving.

Usage examples::

    # Detect text in a single image using morphological operations
    python scripts/test_text_detection.py data/raw_maps/map1.jpg --method morph --interactive

    # Batch process all JPG files and save results in data/text_out
    python scripts/test_text_detection.py "data/raw_maps/*.jpg" --method mser -o data/text_out

The available detection methods are ``morph``, ``mser``, ``canny``, ``sobel`` and ``gradient``.

If you install additional OCR libraries (e.g. EasyOCR or MMOCR), you
can extend ``src/text_detection.py`` with new detectors and expose them here by
adding to the ``METHODS`` dictionary below.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path_for_sys

# Ensure the project root is on sys.path so that `src` can be imported
_project_root = _Path_for_sys(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import glob
from pathlib import Path
from typing import Callable, Dict, List

import cv2
from tqdm import tqdm

from src.text_detection import (
    detect_text_morphology,
    detect_text_mser,
    detect_text_canny,
    detect_text_sobel,
    detect_text_gradient,
    draw_bounding_boxes,
)


# Mapping of method names to detector functions
METHODS: Dict[str, Callable[[cv2.Mat], List[tuple[int, int, int, int]]]] = {
    "morph": detect_text_morphology,
    "mser": detect_text_mser,
    "canny": detect_text_canny,
    "sobel": detect_text_sobel,
    "gradient": detect_text_gradient,
}


def process_image(img_path: Path, method: str) -> cv2.Mat:
    """Detect text in a single image and return an annotated copy."""
    detector = METHODS[method]
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {img_path}")
    boxes = detector(img)
    annotated = draw_bounding_boxes(img, boxes)
    return annotated


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect text in images using simple detectors"
    )
    parser.add_argument(
        "input",
        help="Path to an image or a glob pattern (e.g. 'data/*.jpg')",
    )
    parser.add_argument(
        "--method",
        default="morph",
        choices=list(METHODS.keys()),
        help="Which detection method to use",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data/processed_maps",
        help=(
            "Base output directory. Results will be written under "
            "<output-dir>/<image-stem>/<image-stem>_<method>.jpg"
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Display annotated images interactively instead of saving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    method = args.method

    # Set up base output directory
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Expand glob pattern
    img_paths = [Path(p) for p in glob.glob(args.input)]
    if not img_paths:
        raise FileNotFoundError(f"No files matched pattern: {args.input}")

    for img_path in tqdm(img_paths, desc=f"Running {method} detection"):
        try:
            annotated = process_image(img_path, method)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

        if args.interactive:
            # Display using OpenCV's imshow; wait for key press before closing
            cv2.imshow(f"{method} detection: {img_path.name}", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save annotated image into its own folder
        out_dir = base_out / img_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_path.stem}_{method}.jpg"
        cv2.imwrite(str(out_path), annotated)


if __name__ == "__main__":
    main()