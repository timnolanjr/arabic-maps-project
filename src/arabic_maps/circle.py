import math
import os
import shutil
import subprocess
import sys
from tempfile import mkdtemp
from typing import Iterable, List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def fit_circle(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b = xs * xs + ys * ys
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(c[2] + cx * cx + cy * cy)
    return cx, cy, r


def click_points(img: np.ndarray, n: int = 8) -> List[Tuple[float, float]]:
    disp = cv.cvtColor(img, cv.COLOR_BGR2RGB) if img.ndim == 3 else img
    fig, ax = plt.subplots()
    ax.imshow(disp)
    ax.set_title(f"Click {n} edge points")
    plt.axis('off')

    pts = plt.ginput(n, timeout=-1)
    plt.close(fig)
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
    return pts


def detect_circle_roi(gray: np.ndarray, src: np.ndarray):
    pts = click_points(src, 8)
    if len(pts) < 3:
        return [], []

    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    cx0, cy0, r0 = fit_circle(xs, ys)

    m = int(r0 * 1.2)
    x0 = max(0, int(cx0 - m))
    y0 = max(0, int(cy0 - m))
    x1 = min(gray.shape[1], int(cx0 + m))
    y1 = min(gray.shape[0], int(cy0 + m))
    roi = gray[y0:y1, x0:x1]

    circles = cv.HoughCircles(
        roi, cv.HOUGH_GRADIENT, dp=1, minDist=2,
        param1=100, param2=10,
        minRadius=int(r0 * 0.85), maxRadius=int(r0 * 1.15)
    )
    if circles is None:
        return [], []
    raw = np.uint16(np.around(circles[0]))
    cands = [(c[0] + x0, c[1] + y0, c[2]) for c in raw]

    raw_acc = cands[:50]

    def radial_rms(c):
        cx, cy, cr = c
        dists = np.hypot(xs - cx, ys - cy)
        errs = dists - cr
        return math.sqrt(np.mean(errs * errs))

    err_sorted = sorted(cands, key=radial_rms)[:50]
    return raw_acc, err_sorted


def pick_batch(src: np.ndarray, cands: Iterable[Tuple[int, int, int]], batch_size: int = 100):
    tmp = mkdtemp(prefix="hough_preview_")
    try:
        total = len(cands)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            for i, (cx, cy, cr) in enumerate(cands[start:end], start + 1):
                im = src.copy()
                cv.circle(im, (int(cx), int(cy)), int(cr), (255, 0, 255), 3)
                cv.circle(im, (int(cx), int(cy)), max(3, int(cr * 0.01)), (255, 0, 255), 3)
                cv.imwrite(os.path.join(tmp, f"cand_{i:04d}.jpg"), im)

            subprocess.run(['open', tmp] if sys.platform == 'darwin' else ['xdg-open', tmp])

            choice = input(f"Pick 1–{total}, 'n' next, 's' skip, 'r' redo clicks: ")
            if choice.lower() == 'n':
                continue
            if choice.lower() == 's':
                return 'skip'
            if choice.lower() == 'r':
                return 'redo'
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= total:
                    return cands[idx - 1]
        return None
    finally:
        shutil.rmtree(tmp)


def detect_circle(path: str, interactive: bool):
    src = cv.imread(path)
    if src is None:
        return None
    gray = cv.medianBlur(cv.cvtColor(src, cv.COLOR_BGR2GRAY), 5)

    if interactive:
        raw_acc, err_sorted = detect_circle_roi(gray, src)
        if not raw_acc and not err_sorted:
            return None
        choice = pick_batch(src, err_sorted)
        if choice == 'redo':
            return detect_circle(path, True)
        if choice and choice != 'skip':
            return choice

        choice = pick_batch(src, raw_acc)
        if choice == 'redo':
            return detect_circle(path, True)
        if choice and choice != 'skip':
            return choice
        return None
    else:
        cir = cv.HoughCircles(
            gray, cv.HOUGH_GRADIENT, dp=1, minDist=2,
            param1=100, param2=10,
            minRadius=int(min(gray.shape) * 0.1),
            maxRadius=int(min(gray.shape) * 0.5)
        )
        if cir is None:
            return None
        raw = np.uint16(np.around(cir[0]))
        best = max(raw.tolist(), key=lambda c: c[2])
        return tuple(best)


def human_size(b: int) -> str:
    for u in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}TB"
