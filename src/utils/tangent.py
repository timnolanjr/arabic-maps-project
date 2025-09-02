from __future__ import annotations
from typing import Tuple
import math

def compute_tangent_point(
    cx: float,
    cy: float,
    r: float,
    rho: float,
    theta: float,
    *,
    prefer_top: bool = True,
) -> Tuple[float, float]:
    """
    Return a tangent point on a circle given a detected line in Hough normal form:
        x*cos(theta) + y*sin(theta) = rho

    If prefer_top is True, return the topmost of the two diametric candidates along
    the line's unit normal (cos(theta), sin(theta)). Image coordinates are assumed
    to have y increasing downward, so "top" means the smaller y value.

    If prefer_top is False, return the classic "closest to the line" choice.
    """
    nx = math.cos(theta)
    ny = math.sin(theta)

    if prefer_top:
        x1, y1 = cx - r * nx, cy - r * ny
        x2, y2 = cx + r * nx, cy + r * ny
        return (x1, y1) if y1 <= y2 else (x2, y2)

    # Closest-to-line variant (kept for backward compatibility)
    s = nx * cx + ny * cy - rho  # signed distance up to a scale
    x = cx - math.copysign(r, s) * nx
    y = cy - math.copysign(r, s) * ny
    return x, y
