#!/usr/bin/env python3
"""
End-to-end pipeline for raw map images, with cardinal labels.
Supports single files, directories, and glob patterns.
"""

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Iterable, List

import cv2

from src.utils.metadata import init_params_from_image
from src.circle import interactive_detect_and_save as detect_circle
from src.edges  import interactive_detect_and_save as detect_edge
from src.utils.tangent import compute_tangent_point


ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


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
                [q for q in p.iterdir()
                 if q.is_file()
                 and not q.name.startswith(".")
                 and q.suffix.lower() in ALLOWED_EXT]
            )

    return []


def process_one_map(img_path: Path, out_base: Path, *, ask: bool = True, force: bool = False) -> None:
    """Process a single image."""
    print(f"→ {img_path.name}")

    out_dir      = out_base / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path  = out_dir / "params.json"
    overlay_path = out_dir / "params_overlay.jpg"

    # Skip guards unless forcing
    if not force:
        if overlay_path.exists():
            print(f"  ✔ Skipping (overlay exists)")
            return
        if params_path.exists():
            data = json.loads(params_path.read_text(encoding="utf-8"))
            required = ["center_x","center_y","radius","rho","theta","tangent_x","tangent_y"]
            if all(data.get(k) is not None for k in required):
                print(f"  ✔ Skipping (all parameters present)")
                return

    # Optional prompt
    if ask:
        ans = input(f"  Process {img_path.name}? [Y/n] ").strip().lower()
        if ans == "n":
            print("  → skipped")
            return

    print("  → Processing")

    # 1) Metadata init
    init_params_from_image(img_path, out_base)

    # 2) Circle detection
    detect_circle(img_path, out_base)

    # 3) Edge detection
    detect_edge(img_path, out_base)

    # 4) Tangent computation
    compute_tangent_point(params_path)
    data = json.loads(params_path.read_text(encoding="utf-8"))

    # Extract parameters
    cx, cy, r  = int(data["center_x"]), int(data["center_y"]), int(data["radius"])
    rho, theta = data["rho"], data["theta"]
    tx, ty     = int(data["tangent_x"]), int(data["tangent_y"])

    # 5) Overlay generation
    img = cv2.imread(str(img_path))
    if img is None:
        print("  ✖ Failed to read image; skipping")
        return

    overlay = img.copy()
    h, w = overlay.shape[:2]

    # scale ~ 1.0 at 1000px
    scale        = max(0.1, min(w, h) / 1000.0)
    font_scale   = 0.6 * scale
    text_thick   = max(1, int(2 * scale))
    circle_thick = max(1, int(2 * scale))
    line_thick   = max(1, int(2 * scale))
    marker_rad   = max(2, int(max(2, r * 0.02)))  # keep readable
    offset_px    = int(20 * scale)

    # Draw circle + center
    cv2.circle(overlay, (cx, cy), r, (0, 255, 0), thickness=int(max(2, circle_thick // 2)))
    cv2.circle(overlay, (cx, cy), marker_rad, (0, 255, 0), -1)

    # Draw edge line from (rho, theta)
    a, b = math.cos(theta), math.sin(theta)
    if abs(b) < 1e-6:
        x = int(rho / a) if abs(a) > 1e-12 else 0
        p1, p2 = (x, 0), (x, h)
    else:
        y0 = (rho - 0 * a) / b
        y1 = (rho - w * a) / b
        p1, p2 = (0, int(round(y0))), (w, int(round(y1)))
    cv2.line(overlay, p1, p2, (255, 0, 0), thickness=line_thick)

    # Cardinal-direction labels (S at tangent)
    cardinals = {
            "S": (tx, ty),
            "W": (cx - r, cy),
            "N": (cx, cy + r),
            "E": (cx + r, cy),
        }
    for label, (x, y) in cardinals.items():
        cx, cy = int(x), int(y)

        # draw the black circle
        cv2.circle(overlay, (cx, cy), int(1.5 * marker_rad), (0, 0, 0), -1)

        # measure text and compute centered origin
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
        org = (cx - tw // 2, cy + th // 2 - bl // 2)  # center the text box on (cx, cy)

        # draw the label, anti-aliased
        cv2.putText(
            overlay, label, org,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), text_thick, cv2.LINE_AA
        )


    # Annotate metadata
    text_lines = [
        f"cx={cx}, cy={cy}, r={r}",
        f"rho={rho:.1f}, theta={math.degrees(theta):.1f} degrees",
        f"South / الجَنوب: tx={tx}, ty={ty}",
    ]
    y0 = offset_px
    for line in text_lines:
        cv2.putText(
            overlay, line, (offset_px, y0),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thick
        )
        y0 += int(25 * scale)

    # Save final overlay
    cv2.imwrite(str(overlay_path), overlay)
    print(f"  → Saved overlay to {overlay_path}")


def process_all_maps(images: Iterable[Path], out_base: Path, *, ask: bool, force: bool) -> None:
    images = list(images)
    total = len(images)
    if total == 0:
        print("No images found to process.")
        return

    for idx, img_path in enumerate(images, start=1):
        print(f"\n[{idx}/{total}] ", end="")
        process_one_map(img_path, out_base, ask=ask, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw map images (file, directory, or glob).")
    parser.add_argument("input", help="Image file, directory, or glob (e.g., 'data/raw_maps/*.jpg').")
    parser.add_argument("-o", "--out", default="data/processed_maps", help="Output base directory.")
    parser.add_argument("-y", "--yes", action="store_true", help="Process without interactive prompts.")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist/are complete.")
    args = parser.parse_args()

    imgs = gather_images(args.input)
    process_all_maps(imgs, Path(args.out), ask=not args.yes, force=args.force)
