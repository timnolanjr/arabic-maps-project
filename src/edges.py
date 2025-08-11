# src/edges.py

import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils.image import median_blur_and_gray, show_image
from src.utils.io import update_json, make_output_dir


# -----------------------------------------------------------------------------
# Low-level detection
# -----------------------------------------------------------------------------
def detect_edges_canny(
    gray: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    return cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)


def detect_lines_hough(
    edge: np.ndarray,
    rho: float = 1,
    theta: float = np.pi/(360*5),  # 0.1° resolution
    threshold: int = 100
) -> List[Tuple[float, float]]:
    raw = cv2.HoughLines(edge, rho, theta, threshold)
    return [tuple(l[0]) for l in raw] if raw is not None else []


# -----------------------------------------------------------------------------
# Seed-line fit via PCA
# -----------------------------------------------------------------------------
def fit_edge_line(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    pts = np.vstack([xs, ys]).T
    centroid = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, np.argmax(eigvals)]
    n = np.array([-u[1], u[0]])
    n /= np.linalg.norm(n)
    theta0 = math.atan2(n[1], n[0]) % math.pi
    rho0 = float(n.dot(centroid))
    return rho0, theta0


# -----------------------------------------------------------------------------
# Angle filter (80°–110° normals)
# -----------------------------------------------------------------------------
def filter_horizontal_angles(
    lines: List[Tuple[float, float]],
    min_angle_deg: float = 80.0,
    max_angle_deg: float = 110.0
) -> List[Tuple[float, float]]:
    out = []
    for rho, theta in lines:
        deg = math.degrees(theta)
        if min_angle_deg <= deg <= max_angle_deg:
            out.append((rho, theta))
    return out


# -----------------------------------------------------------------------------
# Mean of rho & θ
# -----------------------------------------------------------------------------
def average_rho_theta(
    lines: List[Tuple[float, float]]
) -> Tuple[float, float]:
    if not lines:
        raise ValueError("No lines to average")
    rhos   = [ρ for ρ, _ in lines]
    thetas = [θ for _, θ in lines]
    rho_m = float(np.mean(rhos))
    theta_m = math.atan2(sum(math.sin(θ) for θ in thetas),
                         sum(math.cos(θ) for θ in thetas)) % math.pi
    return rho_m, theta_m


# -----------------------------------------------------------------------------
# Full-width endpoints & draw
# -----------------------------------------------------------------------------
def line_to_endpoints_full(
    rho: float,
    theta: float,
    img_shape: Tuple[int, int]
) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    h, w = img_shape
    a, b = math.cos(theta), math.sin(theta)
    if abs(b) < 1e-3:
        x = rho / a
        return (int(x), 0), (int(x), h)
    y0 = (rho - 0*a) / b
    y1 = (rho - w*a) / b
    return (0, int(y0)), (w, int(y1))


def draw_line(
    img: np.ndarray,
    rho: float,
    theta: float,
    color: Tuple[int,int,int] = (0,255,0),
    thickness: int = 3
) -> np.ndarray:
    out = img.copy()
    p1, p2 = line_to_endpoints_full(rho, theta, img.shape[:2])
    cv2.line(out, p1, p2, color, thickness)
    return out


# -----------------------------------------------------------------------------
# Interactive ROI-based pipeline
# -----------------------------------------------------------------------------
def interactive_detect_and_save(
    input_path: Path,
    base_output_dir: Path,
    n_clicks: int = 3,
    delta: float = 0.03,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_thresh: int = 100,
    top_k: int = 10,
    min_angle_deg: float = 80.0,
    max_angle_deg: float = 110.0,
    save_fig: bool = False
):
    out_dir = make_output_dir(input_path, base_output_dir)
    img  = cv2.imread(str(input_path))
    gray = median_blur_and_gray(img)
    H, _ = gray.shape

    # Step 1: click n points on the raw image
    disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(disp, origin="upper")
    ax.set_title(f"Click {n_clicks} points along the top edge\n(close to continue)")
    ax.axis("off")
    pts = plt.ginput(n_clicks, timeout=-1)
    # plt.close(fig)

    xs = np.array([x for x, _ in pts])
    ys = np.array([y for _, y in pts])

    rho0, theta0 = fit_edge_line(xs, ys)
    p1, p2 = line_to_endpoints_full(rho0, theta0, img.shape[:2])

    # ax.plot(
    #     [p1[0], p2[0]],
    #     [p1[1], p2[1]],
    #     color="red",
    #     linewidth=8,
    # )

    # render & block until closed
    plt.draw()
    plt.show(block=True)
    plt.close(fig)
    # Step 3: build ROI band ±delta*H around clicked ys
    y0 = max(0, int(min(ys) - delta * H))
    y1 = min(H, int(max(ys) + delta * H))
    roi = gray[y0:y1, :]

    # Step 4: Canny+Hough in ROI, adjust back to full image coords
    edges = detect_edges_canny(roi, canny_low, canny_high)
    raw   = detect_lines_hough(edges, threshold=hough_thresh)[:top_k]
    lines = [(ρ + y0 * math.sin(θ), θ) for ρ, θ in raw]

    # Step 5: filter horizontals & average
    horiz    = filter_horizontal_angles(lines, min_angle_deg, max_angle_deg)
    rho_m, theta_m = average_rho_theta(horiz)

    # Step 6: show & save final result
    if save_fig:
        final = draw_line(img, rho_m, theta_m, thickness=3)
        show_image(
            final,
            title=f"Final edge — ρ={rho_m:.2f}, θ={math.degrees(theta_m):.2f}°"
        )
        print(f"Saved edge_final.jpg {out_dir}")
        cv2.imwrite(str(out_dir/"edge_final.jpg"), final)

    
    update_json(out_dir/"params.json", {"rho": rho_m, "theta": theta_m})
    print(f"Saved updated params.json in {out_dir}")


# -----------------------------------------------------------------------------
# Batch (non-interactive) pipeline
# -----------------------------------------------------------------------------
def batch_detect_and_save(
    input_path: Path,
    base_output_dir: Path,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_thresh: int = 100,
    top_k: int = 10,
    min_angle_deg: float = 80.0,
    max_angle_deg: float = 110.0,
):
    out_dir = make_output_dir(input_path, base_output_dir)
    img  = cv2.imread(str(input_path))
    gray = median_blur_and_gray(img)

    edges = detect_edges_canny(gray, canny_low, canny_high)
    peaks = detect_lines_hough(edges, threshold=hough_thresh)[:top_k]
    horiz = filter_horizontal_angles(peaks, min_angle_deg, max_angle_deg)
    rho_m, theta_m = average_rho_theta(horiz)

    overlay = draw_line(img, rho_m, theta_m)
    cv2.imwrite(str(out_dir/"edge_overlay.jpg"), overlay)
    update_json(out_dir/"params.json", {"rho": rho_m, "theta": theta_m})
    print(f"Saved batch edge_overlay.jpg & updated params.json in {out_dir}")
