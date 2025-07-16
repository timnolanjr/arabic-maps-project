#!/usr/bin/env python3
"""
easyocr_detect.py

Run EasyOCR’s CRAFT detector on an image and visualize:
  - optional heatmaps (text score & link score)
  - raw bounding-polygon detections
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

def run_easyocr_detect(path, langs=['ar'], gpu=False):
    """
    Detect text regions using EasyOCR's low-level CRAFT detector.

    Returns:
      boxes       : list of Nx4×2 arrays (or flat 4‐tuples) of corner coords
      score_text  : 2D numpy array or None
      score_link  : 2D numpy array or None
    """
    reader = easyocr.Reader(langs, gpu=gpu)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not open image at {path!r}")

    # grab raw output (could be 2‐tuple or 4‐tuple depending on version)
    res = reader.detect(
        img            = img,
        min_size         = 10,
        text_threshold   = 0.4,
        low_text         = 0.2,
        link_threshold   = 0.2,
        mag_ratio        = 1.5,
        slope_ths        = 0.3,
        ycenter_ths      = 0.8,
        height_ths       = 0.8,
        width_ths        = 1.0,
        add_margin       = 0.2,
        optimal_num_chars= None
    )

    # unpack into boxes + optional heatmaps
    boxes = []
    score_text = None
    score_link = None

    if isinstance(res, tuple):
        if len(res) == 4:
            # version that returns (horz, free, score_text, score_link)
            horz, free, score_text, score_link = res
            boxes = horz + free
        elif len(res) == 2:
            # version that returns (horz, free)
            horz, free = res
            boxes = horz + free
        else:
            # fallback: first element is a list of boxes
            boxes = res[0]
    else:
        # if somehow detect returns just one list
        boxes = res

    # ensure numpy arrays
    boxes = [np.array(b, dtype=int) for b in boxes]

    print(f"→ Detected {len(boxes)} total boxes (rect + free‐form).")
    return boxes, score_text, score_link


def plot_heatmaps(score_text, score_link):
    """Plot the CRAFT text & link score heatmaps, if available."""
    if score_text is None or score_link is None:
        print("→ No heatmaps returned by this EasyOCR version.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(score_text, cmap='hot')
    axes[0].set_title('CRAFT Text Heatmap')
    axes[0].axis('off')
    axes[1].imshow(score_link, cmap='hot')
    axes[1].set_title('CRAFT Link Heatmap')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def draw_boxes(path, boxes, color=(0,255,0), thickness=2):
    """
    Draw each polygonal box on the RGB image.
    Handles both flat [x1,y1,x2,y2] and Nx2 quads.
    """
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    for b in boxes:
        pts = np.array(b, dtype=int)

        if pts.ndim == 1 and pts.size == 4:
            # [x1,y1,x2,y2]
            x1,y1,x2,y2 = pts
            cv2.rectangle(img_rgb, (x1,y1), (x2,y2), color, thickness)

        elif pts.ndim == 2 and pts.shape[1] == 2:
            # Nx2 polygon
            poly = pts.reshape(-1,1,2)
            cv2.polylines(img_rgb, [poly], True, color, thickness)

        else:
            # fallback: convex hull of arbitrary points
            hull = cv2.convexHull(pts)
            cv2.polylines(img_rgb, [hull], True, color, thickness)

    plt.figure(figsize=(8,8))
    plt.imshow(img_rgb)
    plt.title("Detected Text Regions")
    plt.axis('off')
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python easyocr_detect.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]

    print(f"\n→ Running EasyOCR CRAFT detector on:\n   {path}\n")
    boxes, score_text, score_link = run_easyocr_detect(path, langs=['ar'], gpu=False)

    print(f"\n→ Visualizing {len(boxes)} boxes …\n")
    plot_heatmaps(score_text, score_link)
    draw_boxes(path, boxes)


if __name__ == "__main__":
    main()
