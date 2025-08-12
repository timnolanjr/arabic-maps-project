#!/usr/bin/env python3
"""
End-to-end pipeline for raw map images, with cardinal labels.
Supports single files, directories, and glob patterns.

Modes:
- default: run detection steps (if needed) + render overlay
- --render-only: just render overlay from existing params.json
- --no-overlay: run detection steps but don't render
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable, List

from src.utils.metadata import init_params_from_image
from src.circle import interactive_detect_and_save as detect_circle
from src.edges import interactive_detect_and_save as detect_edge
from src.utils.tangent import compute_tangent_point
from src.utils.image import render_overlay_from_paths

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
REQUIRED_KEYS = ["center_x", "center_y", "radius", "rho", "theta", "tangent_x", "tangent_y"]


def gather_images(inp: str) -> List[Path]:
    """Accept a file path, directory, or glob pattern; return sorted image paths."""
    p = Path(inp)

    # Glob pattern (wildcards)
    if any(ch in inp for ch in "*?[]"):
        paths = [Path(x) for x in glob.glob(inp, recursive=True)]
        return sorted([x for x in paths if x.is_file() and x.suffix.lower() in ALLOWED_EXT])

    # Existing path
    if p.exists():
        if p.is_file():
            return [p] if p.suffix.lower() in ALLOWED_EXT else []
        if p.is_dir():
            return sorted(
                q for q in p.iterdir()
                if q.is_file() and not q.name.startswith(".")
                and q.suffix.lower() in ALLOWED_EXT
            )

    return []


def params_complete(params_path: Path) -> bool:
    if not params_path.exists():
        return False
    try:
        data = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(data.get(k) is not None for k in REQUIRED_KEYS)


def ensure_params(img_path: Path, out_dir: Path, *, ask: bool) -> Path:
    """
    Run detection steps as needed to produce params.json with all required fields.
    Returns path to params.json.
    """
    params_path = out_dir / "params.json"
    if params_complete(params_path):
        return params_path

    # Optional prompt (per image)
    if ask:
        ans = input(f"  Process {img_path.name}? [Y/n] ").strip().lower()
        if ans == "n":
            raise RuntimeError("Skipped by user")

    # Run steps
    init_params_from_image(img_path, out_dir.parent)      # writes/initializes params.json
    detect_circle(img_path, out_dir.parent)               # updates params.json
    detect_edge(img_path, out_dir.parent)                 # updates params.json
    compute_tangent_point(params_path)                    # finalizes tangent + rho/theta

    if not params_complete(params_path):
        raise RuntimeError("params.json incomplete after detection steps")

    return params_path


def process_one_map(
    img_path: Path,
    out_base: Path,
    *,
    ask: bool = True,
    force: bool = False,
    render_only: bool = False,
    no_overlay: bool = False,
) -> None:
    """Process a single image."""
    print(f"→ {img_path.name}")

    out_dir = out_base / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.json"
    overlay_path = out_dir / "final_overlay.jpg"

    # Skip guard: existing overlay
    if overlay_path.exists() and not force and not no_overlay:
        print("  ✔ Skipping (overlay exists). Use --force to overwrite.")
        return

    # If render-only, don't run detection—require params.json
    if render_only:
        if not params_complete(params_path):
            print("  ✖ Missing or incomplete params.json; cannot render-only. Run without --render-only first.")
            return
        out = render_overlay_from_paths(img_path, params_path, overlay_path, unicode_glyphs=False)
        print(f"  → Saved overlay to {out}")
        return

    # Otherwise, ensure params.json exists/complete
    try:
        params_path = ensure_params(img_path, out_dir, ask=ask)
        print("  ✔ Parameters ready")
    except RuntimeError as e:
        print(f"  → {e}")
        return

    # Render overlay unless suppressed
    if not no_overlay:
        out = render_overlay_from_paths(img_path, params_path, overlay_path, unicode_glyphs=False)
        print(f"  → Saved overlay to {out}")
    else:
        print("  → Skipped overlay rendering (per --no-overlay)")


def process_all_maps(
    images: Iterable[Path],
    out_base: Path,
    *,
    ask: bool,
    force: bool,
    render_only: bool,
    no_overlay: bool,
) -> None:
    images = list(images)
    total = len(images)
    if total == 0:
        print("No images found to process.")
        return

    for idx, img_path in enumerate(images, start=1):
        print(f"\n[{idx}/{total}] ", end="")
        process_one_map(
            img_path,
            out_base,
            ask=ask,
            force=force,
            render_only=render_only,
            no_overlay=no_overlay,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process raw map images (file, directory, or glob)."
    )
    parser.add_argument("input", help="Image file, directory, or glob (e.g., 'data/raw_maps/*.jpg').")
    parser.add_argument("-o", "--out", default="data/processed_maps", help="Output base directory.")
    parser.add_argument("-y", "--yes", action="store_true", help="Process without interactive prompts.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing overlays.")
    parser.add_argument("--render-only", action="store_true", help="Only render overlay from existing params.json.")
    parser.add_argument("--no-overlay", action="store_true", help="Run detection steps only; skip overlay.")
    args = parser.parse_args()

    imgs = gather_images(args.input)
    process_all_maps(
        imgs,
        Path(args.out),
        ask=not args.yes,
        force=args.force,
        render_only=args.render_only,
        no_overlay=args.no_overlay,
    )
