import json
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np

def make_output_dir(input_path: Path, base_output_dir: Path) -> Path:
    out_dir = base_output_dir / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_json(path: Path, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def update_json(
    path: Path,
    new_data: Dict[str, Any],
    *,
    skip_none: bool = True,   # <- default: do NOT overwrite existing values with None
    deep: bool = False,       # <- set True if you ever need recursive dict merges
) -> None:
    """
    Read JSON if it exists, merge in new_data, and write back atomically.

    Rules:
      - If skip_none=True: values of None in new_data will NOT overwrite existing non-null values.
      - If deep=True: merge nested dicts recursively, else overwrite at the top level.
    """
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (JSONDecodeError, OSError, ValueError):
        existing = {}

    def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in src.items():
            if skip_none and v is None and k in dst and dst[k] is not None:
                continue  # keep the existing non-null value
            if deep and isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v
        return dst

    merged = _merge(existing if isinstance(existing, dict) else {}, dict(new_data))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    tmp.replace(path)

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def write_image(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")
    return path

def default_overlay_path(image_path: Path) -> Path:
    ext = image_path.suffix.lower()
    out_ext = ".jpg" if ext in {".jpg", ".jpeg"} else ".png"
    return image_path.with_name("params_overlay" + out_ext)