# src/utils/metadata.py

import json
from pathlib import Path
import cv2

def init_params_from_image(input_path: Path, out_base: Path) -> None:
    """
    Extract basic metadata from a raw map image and write a params.json
    with empty placeholders for circle, edge, and tangent parameters.
    """
    out_dir = out_base / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # file metadata
    filetype = input_path.suffix.lstrip('.').lower()
    filesize = input_path.stat().st_size

    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {input_path!r}")

    # determine colorspace & dimensions
    if img.ndim == 2:
        colorspace = "gray"
        h, w = img.shape
    else:
        colorspace = "sRGB"
        h, w = img.shape[:2]

    params = {
        "filetype":     filetype,
        "filesize":     filesize,
        "image_width":  w,
        "image_height": h,
        "colorspace":   colorspace,
        "center_x":     None,
        "center_y":     None,
        "radius":       None,
        "rho":          None,
        "theta":        None,
        "tangent_x":    None,
        "tangent_y":    None,
    }

    (out_dir / "params.json").write_text(
        json.dumps(params, indent=2),
        encoding="utf-8"
    )
