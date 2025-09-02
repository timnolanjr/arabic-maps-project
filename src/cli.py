#!/usr/bin/env python3
"""
Unified CLI for the Arabic Maps project.

Subcommands:
  - pipeline : run full geometry pipeline (circle -> edge -> tangent) on file or directory
  - circle   : detect circle only
  - edge     : detect edge line only
  - text     : run text detection (MSER/morph/etc.)

Examples:
  python -m src.cli pipeline data/raw_maps/Ibn.jpg -o data/processed_maps
  python -m src.cli circle   data/raw_maps            -o data/processed_maps --interactive
  python -m src.cli edge    data/raw_maps/Ibn.jpg   -o data/processed_maps
  python -m src.cli text     data/raw_maps           -o data/processed_maps --method mser
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, List
import sys

# ----------------------------
# Helpers
# ----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _iter_images(path: Path) -> Iterable[Path]:
    """
    Yield image files from a file or a directory, skipping dotfiles.
    """
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMG_EXTS:
                yield p
    elif path.is_file():
        if not path.name.startswith(".") and path.suffix.lower() in IMG_EXTS:
            yield path
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")


def _ensure_out_base(out_base: Optional[Path]) -> Optional[Path]:
    if out_base is not None:
        out_base.mkdir(parents=True, exist_ok=True)
    return out_base


# ----------------------------
# Subcommand implementations
# ----------------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    try:
        from src.pipeline_core import run_pipeline
    except Exception as e:
        print("Error: could not import pipeline_core.run_pipeline.", file=sys.stderr)
        raise

    input_path = Path(args.input)
    out_base = Path(args.out) if args.out else None
    _ensure_out_base(out_base)
    run_pipeline(input_path, out_base=out_base, interactive=args.interactive)


def _run_geometry_single(
    which: str,
    img_path: Path,
    out_base: Optional[Path],
    interactive: bool,
) -> Path:
    """
    Handle single-image circle/edges runs.
    """
    out_dir = (out_base or Path("data/processed_maps")) / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if which == "circle":
        from src.circle import interactive_detect_and_save as detect
        detect(img_path, out_dir=out_dir, interactive=interactive)
    elif which == "edge":
        from src.edges import interactive_detect_and_save as detect
        detect(img_path, out_dir=out_dir, interactive=interactive)
    else:
        raise ValueError(which)
    return out_dir


def _run_geometry_batch(which: str, input_dir: Path, out_base: Optional[Path], interactive: bool) -> None:
    for idx, img_path in enumerate(_iter_images(input_dir), start=1):
        print(f"[{idx}] {img_path.name}")
        out_dir = _run_geometry_single(which, img_path, out_base, interactive)
        print(f"  → wrote {out_dir}")


def cmd_circle(args: argparse.Namespace) -> None:
    p = Path(args.input)
    out_base = Path(args.out) if args.out else None
    _ensure_out_base(out_base)
    if p.is_dir():
        _run_geometry_batch("circle", p, out_base, args.interactive)
    else:
        out_dir = _run_geometry_single("circle", p, out_base, args.interactive)
        print(f"→ wrote {out_dir}")


def cmd_edges(args: argparse.Namespace) -> None:
    p = Path(args.input)
    out_base = Path(args.out) if args.out else None
    _ensure_out_base(out_base)
    if p.is_dir():
        _run_geometry_batch("edge", p, out_base, args.interactive)
    else:
        out_dir = _run_geometry_single("edge", p, out_base, args.interactive)
        print(f"→ wrote {out_dir}")


def cmd_text(args: argparse.Namespace) -> None:
    """
    Delegates to src/text_detection.py if present.
    Expected API patterns (we'll try in this order):
      1) detect_and_save(img: Path, out_dir: Path, method: str, interactive: bool=False) -> List[Path]
      2) detect_text_on_image(img: np.ndarray, method: str, ...) + our own save
      3) a function named like detect_text_<method>(...)
    If none are present, we raise a clean error with guidance.
    """
    try:
        import cv2
    except Exception:
        cv2 = None

    try:
        td = __import__("src.text_detection", fromlist=["*"])
    except Exception as e:
        print("Error: src/text_detection.py not found or failed to import.\n"
              "If you have the standalone script scripts/test_text_detection.py, you can still run that.",
              file=sys.stderr)
        raise

    detect_and_save = getattr(td, "detect_and_save", None)
    detect_text_on_image = getattr(td, "detect_text_on_image", None)

    input_path = Path(args.input)
    out_base = Path(args.out) if args.out else Path("data/processed_maps")
    out_base.mkdir(parents=True, exist_ok=True)

    method = args.method
    interactive = args.interactive

    def _save_overlay(img_bgr, boxes, out_file: Path) -> None:
        if cv2 is None:
            return
        draw = img_bgr.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), draw)

    def _run_text_on_file(img_file: Path) -> Path:
        out_dir = out_base / img_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        if callable(detect_and_save):
            paths = detect_and_save(img_file, out_dir, method=method, interactive=interactive)
            print(f"{img_file.name}: {len(paths) if paths else 0} outputs")
            return out_dir

        if cv2 is None or not callable(detect_text_on_image):
            func = getattr(td, f"detect_text_{method}", None)
            if func is None:
                raise RuntimeError(
                    "text_detection API not recognized. Provide either "
                    "`detect_and_save(img, out_dir, method, interactive=False)`, or "
                    "`detect_text_on_image(img_bgr, method, ...)`, or "
                    f"`detect_text_{method}(...)`."
                )
            img_bgr = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Could not read image: {img_file}")
            boxes = func(img_bgr)
            _save_overlay(img_bgr, boxes, out_dir / "overlay.jpg")
            print(f"{img_file.name}: {len(boxes)} boxes")
            return out_dir

        img_bgr = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {img_file}")
        boxes = detect_text_on_image(img_bgr, method=method, interactive=interactive)
        _save_overlay(img_bgr, boxes, out_dir / "overlay.jpg")
        print(f"{img_file.name}: {len(boxes)} boxes")
        return out_dir

    if input_path.is_dir():
        for idx, img in enumerate(_iter_images(input_path), start=1):
            print(f"[{idx}] {img.name}")
            out_dir = _run_text_on_file(img)
            print(f"  → wrote {out_dir}")
    else:
        out_dir = _run_text_on_file(input_path)
        print(f"→ wrote {out_dir}")


# ----------------------------
# Argparse wiring
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arabmaps",
        description="Arabic Maps unified CLI",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # pipeline
    pp = sub.add_parser("pipeline", help="Run full geometry pipeline (circle → edge → tangent)")
    pp.add_argument("input", help="Image file or directory")
    pp.add_argument("-o", "--out", help="Output base directory (default: processed_maps)")
    pp.add_argument("--interactive", action="store_true", help="Interactive mode")
    pp.set_defaults(func=cmd_pipeline)

    # circle
    pc = sub.add_parser("circle", help="Detect circle only")
    pc.add_argument("input", help="Image file or directory")
    pc.add_argument("-o", "--out", help="Output base directory (default: processed_maps)")
    pc.add_argument("--interactive", action="store_true", help="Interactive mode")
    pc.set_defaults(func=cmd_circle)

    # edges
    pe = sub.add_parser("edge", help="Detect edge line only")
    pe.add_argument("input", help="Image file or directory")
    pe.add_argument("-o", "--out", help="Output base directory (default: processed_maps)")
    pe.add_argument("--interactive", action="store_true", help="Interactive mode")
    pe.set_defaults(func=cmd_edges)

    # text
    pt = sub.add_parser("text", help="Run text detection (MSER/morph/etc.)")
    pt.add_argument("input", help="Image file or directory")
    pt.add_argument("-o", "--out", help="Output base directory (default: data/processed_maps)")
    pt.add_argument("--method", default="mser",
                    help="Text detection method (e.g., mser, morph, canny, sobel, gradient)")
    pt.add_argument("--interactive", action="store_true", help="Interactive mode (if supported)")
    pt.set_defaults(func=cmd_text)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    # Default output fallback so users can omit -o
    if getattr(args, "out", None) is None:
        args.out = "data/processed_maps"
    args.func(args)


if __name__ == "__main__":
    main()