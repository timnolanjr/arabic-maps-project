from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import cv2

from src.utils.metadata import init_params_from_image
from src.circle import interactive_detect_and_save as detect_circle
from src.edges import interactive_detect_and_save as detect_edge
from src.utils.tangent import compute_tangent_point
from src.utils.vis import compose_geometry_overlay
from src.utils.palette import DEFAULT_PALETTE

def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                yield p
    elif path.is_file():
        yield path
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")

def process_image(img_path: Path, out_base: Path, *, interactive: bool = False) -> Path:
    out_dir = out_base / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.json"
    overlay_path = out_dir / "params_overlay.jpg"

    # 1) Init params
    params = init_params_from_image(img_path)

    # 2) Circle
    circle = detect_circle(img_path, out_dir=out_dir, interactive=interactive)
    if circle:
        params.update({
            "center_x": circle["center_x"],
            "center_y": circle["center_y"],
            "radius": circle["radius"],
        })

    # 3) Edge
    edge = detect_edge(img_path, out_dir=out_dir, interactive=interactive)
    if edge:
        params.update({
            "rho": edge["rho"],
            "theta": edge["theta"],
        })

    # 4) Tangent
    keys = {"center_x", "center_y", "radius", "rho", "theta"}
    if keys.issubset(params.keys()):
        tx, ty = compute_tangent_point(
            params["center_x"], params["center_y"], params["radius"], params["rho"], params["theta"]
        )
        params.update({"tangent_x": float(tx), "tangent_y": float(ty)})

    # Save params.json
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # 5) Draw overlay with unified palette
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    overlay = compose_geometry_overlay(bgr, params, palette=DEFAULT_PALETTE)
    cv2.imwrite(str(overlay_path), overlay)
    return out_dir

def run_pipeline(input_path: Path, out_base: Path, *, interactive: bool = False) -> None:
    out_base.mkdir(parents=True, exist_ok=True)
    for idx, img_path in enumerate(_iter_images(input_path), start=1):
        print(f"[{idx}] {img_path.name}")
        out_dir = process_image(img_path, out_base, interactive=interactive)
        print(f"  → wrote {out_dir}")
