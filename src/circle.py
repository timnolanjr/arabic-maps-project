import cv2
import numpy as np
import math
import json
from pathlib import Path
from typing import List, Tuple, Optional


def fit_circle(xs, ys):
    """
    Fit a circle using least squares on provided (x, y) points.
    Returns: (cx, cy, r)
    """
    A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b = xs * xs + ys * ys
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(c[2] + cx * cx + cy * cy)
    return cx, cy, r


def detect_circle_hough(
    gray_img,
    param1=100,
    param2=10,
    min_radius=0,
    max_radius=0,
    top_k: int = 5
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Run OpenCV HoughCircles on a grayscale image.
    Returns a list of top_k circles sorted by accumulator score (if available).
    Each circle is (cx, cy, r).
    """
    circles = cv2.HoughCircles(
        gray_img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=2,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return None

    # Convert to list of (cx, cy, r) and preserve order
    circles = np.around(circles[0]).astype(int).tolist()
    return circles[:top_k]


def draw_circle(img, cx, cy, r, color=(255, 0, 255), thickness=3):
    """
    Draw a circle and center dot on the image.
    Returns the modified image.
    """
    out = img.copy()
    cv2.circle(out, (int(cx), int(cy)), int(r), color, thickness)
    cv2.circle(out, (int(cx), int(cy)), max(3, int(r * 0.01)), color, -1)
    return out


def draw_multiple_circles(img, circles: List[Tuple[int, int, int]], color=(0, 255, 0), thickness=2):
    """
    Draw multiple circles on a copy of the input image.
    """
    out = img.copy()
    for i, (cx, cy, r) in enumerate(circles):
        cv2.circle(out, (cx, cy), r, color, thickness)
        cv2.circle(out, (cx, cy), max(2, int(r * 0.01)), color, -1)
        cv2.putText(out, str(i+1), (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def save_circle_params(path: str, cx: int, cy: int, r: int):
    """
    Save selected circle parameters as JSON.
    """
    data = {"center_x": cx, "center_y": cy, "radius": r}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def median_blur_and_gray(bgr_img, ksize=5):
    """
    Convert a BGR image to grayscale and apply median blur.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, ksize)
    return gray


def get_output_paths(input_path: Path, base_output_dir: Path) -> dict:
    """
    Given an input image path and a base output directory, return useful paths
    for that file's per-map output folder.

    Returns a dict with keys:
        'folder', 'circle_guesses_img', 'circle_final_img', 'circle_params_json'
    """
    base_name = input_path.stem
    out_dir = base_output_dir / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "folder": out_dir,
        "circle_guesses_img": out_dir / "circle_guesses.jpg",
        "circle_final_img": out_dir / "circle_final.jpg",
        "circle_params_json": out_dir / "circle_params.json",
    }
