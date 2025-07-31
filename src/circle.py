# src/circle.py

import json
import math
import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils.image import median_blur_and_gray, show_image
from src.utils.io import save_json, update_json, make_output_dir


# -----------------------------------------------------------------------------
# Core detection & fitting
# -----------------------------------------------------------------------------
def detect_circle_hough(
    gray_img: np.ndarray,
    param1: int = 100,
    param2: int = 10,
    min_radius: int = 0,
    max_radius: int = 0,
    top_k: int = 100,
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Run OpenCV HoughCircles on a grayscale image.
    Returns up to top_k circles (cx, cy, r).
    """
    circles = cv2.HoughCircles(
        gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=2,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return None
    return np.around(circles[0]).astype(int).tolist()[:top_k]


def fit_circle(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a circle to points (xs, ys) via least squares. Returns (cx, cy, r).
    """
    A = np.column_stack([2*xs, 2*ys, np.ones_like(xs)])
    b = xs*xs + ys*ys
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(c[2] + cx*cx + cy*cy)
    return cx, cy, r


def generate_radius_candidates(
    gray_img: np.ndarray,
    short_side: int,
    min_factor: float = 0.15,
    max_factor: float = 0.5,
    num_candidates: int = 10,
) -> List[Tuple[int, int, int]]:
    """
    Generate fixed-radius candidates centered on the image center.
    """
    h, w = gray_img.shape
    cx, cy = w//2, h//2
    factors = np.linspace(min_factor, max_factor, num_candidates)
    return [(cx, cy, int(f*short_side)) for f in factors]


def refine_circle_in_roi(
    gray_img: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    delta: float = 0.1,
    param1: int = 100,
    param2: int = 10,
) -> List[Tuple[int, int, int]]:
    """
    Run HoughCircles within ±delta*r around (cx,cy,r). Returns adjusted circles.
    """
    h, w = gray_img.shape
    m = int(r * (1 + delta))
    x0, y0 = max(0, int(cx-m)), max(0, int(cy-m))
    x1, y1 = min(w, int(cx+m)), min(h, int(cy+m))
    roi = gray_img[y0:y1, x0:x1]
    min_r = max(0, int(r*(1-delta)))
    max_r = int(r*(1+delta))
    raw = detect_circle_hough(roi, param1, param2, min_r, max_r, top_k=5) or []
    return [(cx_off+x0, cy_off+y0, rad) for (cx_off,cy_off,rad) in raw]


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
def draw_circle(
    img: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw a single circle and its center point on the image; return a copy.
    """
    out = img.copy()
    cv2.circle(out, (int(cx), int(cy)), int(r), color, thickness)
    cv2.circle(out, (int(cx), int(cy)), max(3, int(r * 0.01)), color, -1)
    return out


def draw_multiple_circles(
    img: np.ndarray,
    circles: List[Tuple[int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw each circle with a unique color and label them 1..N.
    """
    colormap = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (255, 165, 0),
    ]
    color_cycle = itertools.cycle(colormap)
    out = img.copy()
    for idx, (cx, cy, r) in enumerate(circles, start=1):
        color = next(color_cycle)
        cv2.circle(out, (int(cx), int(cy)), int(r), color, thickness)
        cv2.circle(out, (int(cx), int(cy)), max(2, int(r*0.01)), color, -1)
        cv2.putText(out, str(idx), (int(cx)+5, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out


# -----------------------------------------------------------------------------
# Interactive pipeline
# -----------------------------------------------------------------------------
def interactive_detect_and_save(
    input_path: Path,
    base_output_dir: Path,
    n_clicks: int = 8,
):
    """
    Loop:
      1) Let user click n points → show fitted circle
      2) Generate ROI candidates
      3) Review candidates one‐by‐one with prompt "Accept candidate {idx}? (y/n/r): "
         - 'y' → accept and break
         - 'n' → show next
         - 'r' → restart at step 1
      4) Save chosen circle + params.json
    """
    out_dir = make_output_dir(input_path, base_output_dir)
    img = cv2.imread(str(input_path))
    gray = median_blur_and_gray(img)
    h, w = gray.shape

    while True:
        # Step 1: click & show fit
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(disp)
        ax.set_title(f"Click {n_clicks} edge points around the circle's perimeter.\n(close to continue))")
        plt.axis("off")
        pts = plt.ginput(n_clicks, timeout=-1)

        xs = np.array([x for x,_ in pts])
        ys = np.array([y for _,y in pts])
        cx0, cy0, r0 = fit_circle(xs, ys)

        circ = plt.Circle((cx0, cy0), r0,
                          edgecolor="red", facecolor="none", linewidth=2)
        ax.add_patch(circ)
        plt.draw()
        plt.show(block=True)
        plt.close(fig)

        # Step 2: refine
        candidates = refine_circle_in_roi(gray, cx0, cy0, r0)
        if not candidates:
            raise RuntimeError("No ROI candidates found")

        # Step 3: review
        chosen = None
        for idx, (cx, cy, r) in enumerate(candidates, start=1):
            frame = draw_circle(img, cx, cy, r, color=(0,255,0))
            show_image(frame, title=f"Candidate {idx}/{len(candidates)}")
            ans = input(f"Accept candidate {idx}? (y/n/r): ").strip().lower()
            if ans == 'r':
                # restart from step 1
                break
            if ans == 'y':
                chosen = (cx, cy, r)
                break
            # if 'n', continue to next

        # if user pressed 'r', go back to clicking
        if ans == 'r':
            continue

        # if none accepted, default to first
        if chosen is None:
            print("None accepted; defaulting to first candidate.")
            chosen = candidates[0]

        cx, cy, r = chosen
        break  # exit while True

    # Step 4: save final
    final = draw_circle(img, cx, cy, r)
    cv2.imwrite(str(out_dir/"circle_final.jpg"), final)
    update_json(
        out_dir/"params.json",
        {
            "center_x":   cx,
            "center_y":   cy,
            "radius":     r,
            "image_width":  w,
            "image_height": h,
        }
    )
    print(f"Saved circle_final.jpg & updated params.json in {out_dir}")




# -----------------------------------------------------------------------------
# Batch mode: candidate generation
# -----------------------------------------------------------------------------
def batch_detect_and_save(
    input_path: Path,
    base_output_dir: Path,
    num_candidates: int = 100,
):
    """
    Non-interactive: detect up to num_candidates and save params_candidates.json.
    """
    out_dir = make_output_dir(input_path, base_output_dir)
    img = cv2.imread(str(input_path))
    gray = median_blur_and_gray(img)
    h, w = gray.shape

    candidates = detect_circle_hough(gray, top_k=num_candidates) \
                 or generate_radius_candidates(gray, min(h,w), num_candidates=num_candidates)

    data = {"candidates": candidates, "image_width": w, "image_height": h}
    save_json(out_dir/"params_candidates.json", data)
    print(f"Saved {len(candidates)} candidates → {out_dir/'params_candidates.json'}")


# -----------------------------------------------------------------------------
# Review after batch mode
# -----------------------------------------------------------------------------
def review_candidates_and_save(
    input_path: Path,
    base_output_dir: Path,
):
    """
    Load params_candidates.json, show each candidate one-by-one, accept or skip,
    then save circle_final.jpg + params.json for the accepted circle.
    """
    out_dir = make_output_dir(input_path, base_output_dir)
    data = json.loads((out_dir/"params_candidates.json").read_text())
    candidates = data["candidates"]
    w, h = data["image_width"], data["image_height"]

    img = cv2.imread(str(input_path))
    chosen = None
    for idx, (cx, cy, r) in enumerate(candidates, start=1):
        frame = draw_circle(img, cx, cy, r, color=(255,0,255))
        show_image(frame, title=f"Candidate {idx}/{len(candidates)}")
        if input(f"Accept {idx}? (y/N): ").lower().startswith("y"):
            chosen = (cx, cy, r)
            break

    if chosen is None:
        print("No candidate accepted; defaulting to first.")
        chosen = candidates[0]
    cx, cy, r = chosen

    final = draw_circle(img, cx, cy, r)
    cv2.imwrite(str(out_dir/"circle_final.jpg"), final)
    update_json(
        out_dir/"params.json",
        {
            "center_x": cx,
            "center_y": cy,
            "radius":    r,
            "image_width":  w,
            "image_height": h,
        }
    )
    print(f"Saved circle_final.jpg & updated params.json in {out_dir}")
