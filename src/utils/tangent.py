import json
import math
from pathlib import Path

def compute_tangent_point(params_path: Path) -> None:
    """
    Read center_x, center_y, radius, rho, theta from params.json,
    compute the tangent point (top of the circle) for a horizontal edge line,
    and update JSON.
    """
    data = json.loads(params_path.read_text(encoding="utf-8"))
    cx, cy, r = data["center_x"], data["center_y"], data["radius"]
    theta     = data["theta"]

    # normal vector of the line
    nx, ny = math.cos(theta), math.sin(theta)

    # move *upwards* along -normal by r to hit the top of the circle
    tx = cx - r * nx
    ty = cy - r * ny

    data["tangent_x"] = tx
    data["tangent_y"] = ty

    params_path.write_text(json.dumps(data, indent=2), encoding="utf-8")