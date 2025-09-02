# src/circle.py

import json
import math
import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import shutil
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils.image import median_blur_and_gray, show_image
from src.utils.io import save_json, update_json, make_output_dir

# Unified palette + drawing helpers
from src.utils.palette import DEFAULT_PALETTE
from src.utils.vis import draw_circle as vis_draw_circle, draw_legend


import tempfile
from pathlib import Path

import sys, subprocess, os


def _maybe_open_finder(dir_path: Path) -> None:
    """Open the folder in Finder (macOS); no-op elsewhere."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(["open", str(dir_path)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def _maybe_close_finder(dir_path: Path) -> None:
    """Close Finder windows showing this folder (macOS); no-op elsewhere."""
    if sys.platform != "darwin":
        return
    p = str(dir_path).replace('"', '\\"')
    # Robust AppleScript: compare POSIX path; guard coercions with try blocks
    script = f'''
    on safe_posix_path_of(theTarget)
        try
            set a to (theTarget as alias)
            set tposix to POSIX path of a
            return tposix
        on error
            return ""
        end try
    end safe_posix_path_of

    tell application "Finder"
        set targetPath to "{p}/"
        set winList to every Finder window
        repeat with w in winList
            try
                set t to target of w
                set pposix to my safe_posix_path_of(t)
                if pposix is not "" and pposix is targetPath then
                    close w
                end if
            end try
        end repeat
    end tell
    '''
    try:
        subprocess.run(["osascript", "-e", script], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass



def _save_circle_preview(bgr, cx: float, cy: float, r: float, out_path: Path) -> None:
    """Save a quick visualization of a candidate circle."""
    vis = bgr.copy()
    cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 2)
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)
    cv2.imwrite(str(out_path), vis)

def _paginate_and_select_circle(
    bgr,
    candidates,          # List[Tuple[cx, cy, r]] or list-like
    *,
    page_size: int = 100,
    tmpdir: Path | None = None,
    open_finder: bool = False,
):
    """
    Page through circle candidates, writing previews in chunks of `page_size`.
    Controls:
      - number in shown range -> accept that candidate
      - 'n' -> next page
      - 'p' -> previous page
      - 'r' -> restart paging at first page
      - 'q' -> abort (returns None)
    Returns: (selected_tuple, tmpdir) or (None, tmpdir)
    """
    # normalize
    norm = [(float(cx), float(cy), float(r)) for (cx, cy, r) in candidates]

    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="circle_preview_"))
    else:
        tmpdir = Path(tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)

    n = len(norm)
    page = 0

    while True:
        start = page * page_size
        end = min(start + page_size, n)

        if start >= n:
            print("[circle] No more candidates. Press 'p' for previous page or 'r' to restart.", flush=True)
            page = max(0, page - 1)
            continue

        # write this page’s previews if not already present
        for idx, (cx, cy, r) in enumerate(norm[start:end], start=start + 1):
            out_path = tmpdir / f"cand_{idx:03d}.jpg"
            if not out_path.exists():
                _save_circle_preview(bgr, cx, cy, r, out_path)

        print(f"Preview images ({start+1}–{end}) written to: {tmpdir}")
        if open_finder:
            _maybe_open_finder(tmpdir)

        choice = input(
            f"Enter index [{start+1}–{end}] to accept, "
            f"'n' next, 'p' previous, 'r' restart, or 'q' quit: "
        ).strip().lower()

        if choice == "n":
            page += 1; continue
        if choice == "p":
            page = max(0, page - 1); continue
        if choice == "r":
            page = 0; continue
        if choice == "q":
            if open_finder:
                _maybe_close_finder(tmpdir)
            return None, tmpdir

        if choice.isdigit():
            idx = int(choice)
            if start + 1 <= idx <= end:
                sel = norm[idx - 1]
                print(f"[circle] accepted idx={idx} → cx={sel[0]:.2f}, cy={sel[1]:.2f}, r={sel[2]:.2f}", flush=True)
                if open_finder:
                    _maybe_close_finder(tmpdir)
                return sel, tmpdir

        print("Invalid input.", flush=True)


def _edge_support_fraction(
    dist_map: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    *,
    band: float = 1.5,
    num_samples: int = 720,
) -> float:
    """
    Fraction of sampled perimeter points whose nearest edge is within `band` pixels.
    `dist_map` is a distance transform of the ROI where each value is distance to nearest edge.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
    xs = (cx + r * np.cos(thetas)).astype(int)
    ys = (cy + r * np.sin(thetas)).astype(int)
    H, W = dist_map.shape
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not np.any(m):
        return 0.0
    d = dist_map[ys[m], xs[m]]
    return float(np.mean(d <= band))


def _interleave_by_chunks(
    a: list, b: list, *, chunk: int = 50, key=lambda x: x
) -> list:
    """
    Interleave lists `a` and `b` by chunks: take `chunk` from a, then `chunk` from b, etc.
    Deduplicates by the provided `key` (e.g., tuple of (cx,cy,r)).
    """
    out, seen = [], set()
    i = j = 0
    na, nb = len(a), len(b)
    while i < na or j < nb:
        # from a
        take = 0
        while i < na and take < chunk:
            item = a[i]; i += 1
            k = key(item)
            if k in seen:
                continue
            seen.add(k)
            out.append(item)
            take += 1
        # from b
        take = 0
        while j < nb and take < chunk:
            item = b[j]; j += 1
            k = key(item)
            if k in seen:
                continue
            seen.add(k)
            out.append(item)
            take += 1
    return out


# -----------------------------------------------------------------------------
# Core detection & fitting
# -----------------------------------------------------------------------------
def detect_circle_hough(
    gray_img: np.ndarray,
    param1: int = 100,
    param2: int = 10,
    min_radius: int = 0,
    max_radius: int = 0,
    top_k: Optional[int] = None,
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Run OpenCV HoughCircles on a grayscale image.
    Returns circles as (cx, cy, r). If top_k is None, returns all.
    """
    circles = cv2.HoughCircles(
        gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return None
    arr = np.around(circles[0]).astype(int).tolist()
    return arr if top_k is None else arr[:top_k]

def fit_circle(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """
    Algebraic least-squares circle fit to points (xs, ys).
    Returns (cx, cy, r). Requires len(xs) >= 3.
    """
    if xs.size < 3 or ys.size < 3:
        raise ValueError("fit_circle needs at least 3 points")

    A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b = xs * xs + ys * ys
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(max(0.0, c[2] + cx * cx + cy * cy))
    return float(cx), float(cy), float(r)

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
    cx, cy = w // 2, h // 2
    factors = np.linspace(min_factor, max_factor, num_candidates)
    return [(cx, cy, int(f * short_side)) for f in factors]


def refine_circle_in_roi(
    gray_img: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    delta: float = 0.1,          # ROI half-margin as a fraction of r
    param1: int = 100,
    param2: int = 10,
    *,
    radius_pct: float = 0.02,    # constrain Hough radius to ± this fraction of r
    chunk: int = 50,             # interleave chunk size: N by closeness, N by support
    support_band: float = 1.5,   # px tolerance to count an edge hit
    support_samples: int = 720,  # samples along the circle for scoring
    w_radius: float = 2.0,
    w_center: float = 1.0,
    use_relative: bool = True,
) -> List[Tuple[int, int, int]]:
    """
    Detect ALL circles in an ROI around (cx,cy) with radius constrained to
    [r*(1 - radius_pct), r*(1 + radius_pct)].

    Ranking:
      • by_closeness: weighted (radius_diff [+ relative]) + (center_dist [+ relative])
      • by_support  : edge-support fraction (desc), then closeness (asc)
    Return order = interleaved chunks: <chunk> by_closeness, <chunk> by_support, repeat.
    """
    h, w = gray_img.shape

    # ROI bounds
    m = int(max(1, r * (1 + delta)))
    x0, y0 = max(0, int(cx - m)), max(0, int(cy - m))
    x1, y1 = min(w, int(cx + m)), min(h, int(cy + m))
    roi = gray_img[y0:y1, x0:x1]

    # Constrained radius window
    min_r = max(1, int(round(r * (1.0 - radius_pct))))
    max_r = max(min_r + 1, int(round(r * (1.0 + radius_pct))))

    # Detect ALL candidates in ROI
    raw = detect_circle_hough(roi, param1, param2, min_r, max_r, top_k=None) or []
    if not raw:
        return []

    # Edge-support distance map (once)
    edges = cv2.Canny(roi, 50, 150)
    inv = (edges == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    r_scale = max(1.0, float(r))
    scored = []
    for (cx_off, cy_off, rad) in raw:
        cx_roi, cy_roi, rad = float(cx_off), float(cy_off), float(rad)
        cx_full, cy_full = cx_roi + x0, cy_roi + y0

        # support (accumulator proxy)
        support = _edge_support_fraction(dist, cx_roi, cy_roi, rad,
                                         band=support_band, num_samples=support_samples)

        # closeness
        rad_term = abs(rad - r) / r_scale if use_relative else abs(rad - r)
        ctr_term = math.hypot(cx_full - cx, cy_full - cy) / r_scale if use_relative else \
                   math.hypot(cx_full - cx, cy_full - cy)
        combined = w_radius * rad_term + w_center * ctr_term

        scored.append(((int(cx_full), int(cy_full), int(rad)),
                       support, combined, rad_term, ctr_term))

    # Two rankings
    by_closeness = sorted(scored, key=lambda t: (t[2], -t[1], t[3], t[4]))
    by_support   = sorted(scored, key=lambda t: (-t[1], t[2], t[3], t[4]))

    # Interleave with CLOSENESS FIRST (distance 1–chunk, then support chunk, etc.)
    interleaved = _interleave_by_chunks(
        by_closeness, by_support,
        chunk=chunk,
        key=lambda s: (s[0][0], s[0][1], s[0][2])
    )

    return [t[0] for t in interleaved]


# -----------------------------------------------------------------------------
# Legacy drawing helpers (for previews)
# -----------------------------------------------------------------------------
def draw_circle(
    img: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    out = img.copy()
    cv2.circle(out, (int(cx), int(cy)), int(r), color, thickness)
    cv2.circle(out, (int(cx), int(cy)), max(3, int(r * 0.01)), color, -1)
    return out


def draw_multiple_circles(
    img: np.ndarray,
    circles: List[Tuple[int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
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
        cv2.circle(out, (int(cx), int(cy)), max(2, int(r * 0.01)), color, -1)
        cv2.putText(
            out, str(idx), (int(cx) + 5, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    return out


# -----------------------------------------------------------------------------
# Interactive + non-interactive unified function (backward compatible)
# -----------------------------------------------------------------------------

def interactive_detect_and_save(
    img_path: Path,
    out_dir: Optional[Path] = None,
    interactive: bool = False,
    *,
    base_output_dir: Optional[Path] = None,  # legacy alias
    n_clicks: int = 8,
    preview_size: int = 150,
    save_fig: bool = False,
    **_,
) -> Optional[Dict[str, float]]:
    """
    Detect the map circle. Returns {"center_x","center_y","radius"} or None if aborted.
    If no out_dir/base_output_dir is given, defaults to processed_maps/<image_stem>.
    """
    img_path = Path(img_path)

    # Default out_dir if none specified
    if out_dir is None:
        if base_output_dir is not None:
            out_dir = make_output_dir(img_path, base_output_dir)
        else:
            out_dir = Path("processed_maps") / img_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    gray = median_blur_and_gray(bgr)
    h, w = gray.shape

    cx = cy = r = None  # type: ignore

    if interactive:
        # --- Step 1: Click perimeter points ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Click {n_clicks} points around the circle perimeter.\n(close window to continue)")
        ax.axis("off")
        pts = plt.ginput(n_clicks, timeout=-1)

        if not pts or len(pts) < 3:
            print("[circle] Fewer than 3 points clicked; aborting.", flush=True)
            return None

        xs = np.array([x for x, _ in pts])
        ys = np.array([y for _, y in pts])
        cx0, cy0, r0 = fit_circle(xs, ys)

        print(f"[circle] initial fit → cx={cx0:.2f}, cy={cy0:.2f}, r={r0:.2f}", flush=True)
        circ = plt.Circle((cx0, cy0), r0, fill=False, linewidth=3, edgecolor='red')
        ax.add_patch(circ)
        fig.canvas.draw_idle()
        plt.show(block=True)

        # --- Step 2: Refine in ROI around (cx0,cy0,r0) ---
        candidates = refine_circle_in_roi(
            gray, cx0, cy0, r0,
            delta=0.1,
            radius_pct=0.05,
            chunk=50,
            w_radius=1.0,   # radius importance
            w_center=1.0,   # center importance (try 2.0 to prefer center more)
            use_relative=True,
        )

        if not candidates:
            candidates = [(int(cx0), int(cy0), int(r0))]

        # --- Step 3: Paged preview & selection ---
        selected, temp_dir = _paginate_and_select_circle(bgr, candidates, page_size=100)
        try:
            if selected is None:
                print("[circle] aborted by user.", flush=True)
                return None
            cx, cy, r = selected
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    else:
        # Non-interactive: try Hough; fallback heuristic
        short_side = min(h, w)
        hough = detect_circle_hough(gray, top_k=1)
        if hough:
            cx, cy, r = hough[0]
        else:
            cx, cy, r = w / 2.0, h / 2.0, short_side * 0.45

    result = {"center_x": float(cx), "center_y": float(cy), "radius": float(r)}

    # Persist
    update_json(out_dir / "params.json", result)

    if save_fig:
        overlay = bgr.copy()
        vis_draw_circle(overlay, (int(cx), int(cy)), int(r), palette=DEFAULT_PALETTE)
        draw_legend(overlay, palette=DEFAULT_PALETTE)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_circle.jpg"), overlay)
        print(f"Saved circle overlay → {out_dir}")

    return result



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
                 or generate_radius_candidates(gray, min(h, w), num_candidates=num_candidates)

    data = {"candidates": candidates, "image_width": w, "image_height": h}
    save_json(out_dir / "params_candidates.json", data)
    print(f"Saved {len(candidates)} candidates → {out_dir / 'params_candidates.json'}")


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
    data = json.loads((out_dir / "params_candidates.json").read_text())
    candidates = data["candidates"]
    w, h = data["image_width"], data["image_height"]

    img = cv2.imread(str(input_path))
    chosen = None
    for idx, (cx, cy, r) in enumerate(candidates, start=1):
        frame = draw_circle(img, cx, cy, r, color=(255, 0, 255))
        show_image(frame, title=f"Candidate {idx}/{len(candidates)}")
        if input(f"Accept {idx}? (y/N): ").lower().startswith("y"):
            chosen = (cx, cy, r)
            break

    if chosen is None:
        print("No candidate accepted; defaulting to first.")
        chosen = candidates[0]
    cx, cy, r = chosen

    final = draw_circle(img, cx, cy, r)
    cv2.imwrite(str(out_dir / "circle_final.jpg"), final)
    update_json(
        out_dir / "params.json",
        {
            "center_x": cx,
            "center_y": cy,
            "radius": r,
            "image_width": w,
            "image_height": h,
        },
    )
    print(f"Saved circle_final.jpg & updated params.json in {out_dir}")
