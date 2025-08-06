#!/usr/bin/env python3
"""
End-to-end pipeline for raw map images, with cardinal labels.
"""

import argparse
import json
import math
from pathlib import Path

import cv2

from src.utils.metadata import init_params_from_image
from src.circle import interactive_detect_and_save as detect_circle
from src.edges  import interactive_detect_and_save as detect_edge
from src.utils.tangent import compute_tangent_point

def process_all_maps(raw_dir: Path, out_base: Path):
    # Collect and count all image files up front
    all_files = [p for p in sorted(raw_dir.iterdir()) if p.is_file() and not p.name.startswith('.')]
    total = len(all_files)

    for idx, img_path in enumerate(all_files, start=1):
        print(f"\n[{idx}/{total}]", end=" ")

        out_dir      = out_base / img_path.stem
        params_path  = out_dir / "params.json"
        overlay_path = out_dir / "final_overlay.jpg"

        # Skip if already fully processed
        if overlay_path.exists():
            print(f"✔ Skipping {img_path.name} (overlay exists)")
            continue
        if params_path.exists():
            data = json.loads(params_path.read_text(encoding="utf-8"))
            required = ["center_x","center_y","radius","rho","theta","tangent_x","tangent_y"]
            if all(data.get(k) is not None for k in required):
                print(f"✔ Skipping {img_path.name} (all parameters present)")
                continue

        # Prompt whether to process
        ans = input(f"Process {img_path.name}? [Y/n] ").strip().lower()
        if ans == "n":
            print("→ skipped")
            continue

        print(f"→ Processing: {img_path.name}")

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
        img     = cv2.imread(str(img_path))
        overlay = img.copy()
        h, w    = overlay.shape[:2]

        # compute scale factor relative to image size (1000px -> scale 1.0)
        scale        = min(w, h) / 1000.0
        font_scale   = 0.6 * scale
        text_thick   = max(1, int(2 * scale))
        circle_thick = max(1, int(2 * scale))
        line_thick   = max(1, int(2 * scale))
        marker_rad   = max(3, int(r * 0.02 * scale))
        offset_px    = int(20 * scale)

        # Draw circle
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), thickness=int(circle_thick/2))
        cv2.circle(overlay, (cx, cy), marker_rad, (0, 255, 0), -1)

        # Draw edge line
        a, b = math.cos(theta), math.sin(theta)
        if abs(b) < 1e-3:
            p1 = (int(rho / a), 0)
            p2 = (int(rho / a), h)
        else:
            y0 = (rho - 0*a)/b
            y1 = (rho - w*a)/b
            p1 = (0, int(y0))
            p2 = (w, int(y1))
        cv2.line(overlay, p1, p2, (255, 0, 0), thickness=line_thick)

        # Cardinal-direction labels
        cardinals = {
            "S": (tx, ty),
            "W": (cx - r, cy),
            "N": (cx, cy + r),
            "E": (cx + r, cy),
        }
        for label, (x, y) in cardinals.items():
            cv2.circle(overlay, (int(x), int(y)), marker_rad, (0, 0, 0), -1)
            cv2.putText(
                overlay,
                label,
                (int(x - 10*scale), int(y + 10*scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255,255,255),
                text_thick,
            )

        # Annotate metadata
        text_lines = [
            f"cx={cx}, cy={cy}, r={r}",
            f"rho={rho:.1f}, theta={math.degrees(theta):.1f}°",
            f"tx={tx}, ty={ty}"
        ]
        y0 = offset_px
        for line in text_lines:
            cv2.putText(
                overlay,
                line,
                (offset_px, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0,255,0),
                text_thick
            )
            y0 += int(25 * scale)

        # Save final overlay
        cv2.imwrite(str(overlay_path), overlay)
        print(f"→ Saved overlay to {overlay_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all raw maps")
    parser.add_argument("raw_dir", help="Path to data/raw_maps/")
    parser.add_argument("-o","--out",      default="data/processed_maps")
    args = parser.parse_args()

    process_all_maps(Path(args.raw_dir), Path(args.out))
