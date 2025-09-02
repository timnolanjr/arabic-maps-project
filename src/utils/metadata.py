from pathlib import Path
import cv2
from src.utils.io import update_json  # <-- use the merge writer

def init_params_from_image(input_path: Path, out_base: Path) -> dict:
    """
    Extract basic metadata and merge into params.json.
    Safe to run before or after other stages: it won't overwrite non-null values.
    """
    out_dir = out_base
    out_dir.mkdir(parents=True, exist_ok=True)

    filetype = input_path.suffix.lstrip('.').lower()
    filesize = input_path.stat().st_size

    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {input_path!r}")

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

    json_path = out_dir / "params.json"
    update_json(json_path, params, skip_none=True)
    return params
