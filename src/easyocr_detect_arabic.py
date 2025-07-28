#!/usr/bin/env python3
# easyocr_detect_arabic.py
# ------------------------------------------------------------
# Detect Arabic text regions using EasyOCR’s CRAFT detector,
# unpack nested return variants, then plot boxes + heatmaps.
#
# pip install easyocr opencv-python matplotlib
# ------------------------------------------------------------

import sys
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr


def run_easyocr_detect(path, langs=['ar'], gpu=False):
    """
    Detect text regions using EasyOCR's CRAFT detector.
    Returns:
      boxes       : list of integer arrays ([x1,y1,x2,y2] or Nx2 polys)
      score_text  : 2D numpy array or None
      score_link  : 2D numpy array or None
    """
    reader = easyocr.Reader(langs, gpu=gpu)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {path!r}")

    # raw detection output
    res = reader.detect(
        img,
        canvas_size      = 2048,
        text_threshold   = 0.5,
        # low_text         = 0.25,
        # link_threshold   = 0.25,
        # mag_ratio        = 1.,
        # slope_ths        = 0.99,
        # ycenter_ths      = 0.25,
        height_ths       = 0.25,
        width_ths        = 0.25,
        # add_margin       = 0.2,
        optimal_num_chars= None
    )

    # Unpack based on tuple length
    if isinstance(res, tuple):
        if len(res) == 4:
            horz, free, score_text, score_link = res
        elif len(res) == 2:
            horz, free = res
            score_text = score_link = None
        else:
            horz = res[0]
            free = []
            score_text = score_link = None
    else:
        horz, free = res, []
        score_text = score_link = None

    # Flatten if they came as a single nested list
    if isinstance(horz, list) and len(horz) == 1 and isinstance(horz[0], list):
        horz = horz[0]
    if isinstance(free, list) and len(free) == 1 and isinstance(free[0], list):
        free = free[0]

    # Build unified boxes list
    boxes = []

    # Rectangles: each `b` should flatten to [x1,y1,x2,y2]
    for b in horz:
        arr = np.array(b, dtype=int).reshape(-1)
        if arr.size >= 4:
            boxes.append(arr[:4])

    # Polygons: each `p` should be an Nx2 array
    for p in free:
        arr = np.array(p, dtype=int)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
            boxes.append(arr)

    return boxes, score_text, score_link


def plot_heatmaps(score_text, score_link):
    """Plot the CRAFT text & link score heatmaps, if available."""
    if score_text is None or score_link is None:
        print("→ No heatmaps returned by this EasyOCR version.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(score_text, cmap='hot')
    ax1.set_title('Text Score Heatmap'); ax1.axis('off')
    ax2.imshow(score_link, cmap='hot')
    ax2.set_title('Link Score Heatmap'); ax2.axis('off')
    plt.tight_layout(); plt.show()


def draw_boxes(path, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw detected boxes (rectangles or polygons) on the image,
    correctly mapping [x_min, x_max, y_min, y_max].
    """
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for pts in boxes:
        pts = np.array(pts, dtype=np.int32)

        # case 1: 4-element rectangle [x_min, x_max, y_min, y_max]
        if pts.ndim == 1 and pts.size == 4:
            x_min, x_max, y_min, y_max = pts
            cv2.rectangle(img_rgb,
                          (x_min, y_min),
                          (x_max, y_max),
                          color, thickness)

        # case 2: Nx2 polygon
        elif pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] >= 3:
            poly = pts.reshape(-1, 1, 2)
            cv2.polylines(img_rgb, [poly], True, color, thickness)

        # all other shapes skipped

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title("Detected Text Regions")
    plt.axis('off')
    plt.show()



def main():
    parser = argparse.ArgumentParser(
        description="EasyOCR Arabic Text Detection (boxes + heatmaps)")
    parser.add_argument('image', help="Path to input image")
    parser.add_argument('--gpu', action='store_true', help="Use GPU")
    args = parser.parse_args()

    print(f"\n→ Running EasyOCR CRAFT detector on:\n   {args.image}\n")
    boxes, score_text, score_link = run_easyocr_detect(
        args.image, langs=['ar'], gpu=args.gpu
    )
    print(f"→ Detected {len(boxes)} regions.\n")

    plot_heatmaps(score_text, score_link)
    draw_boxes(args.image, boxes)


if __name__ == '__main__':
    main()
