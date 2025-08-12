# utils/io.py
import json
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np

def make_output_dir(input_path: Path, base_output_dir: Path) -> Path:
    """
    Create and return the per-image output folder.
    """
    out_dir = base_output_dir / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_json(path: Path, data: Dict[str, Any]):
    """
    Save a dict to JSON at the given path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def update_json(path: Path, new_data: Dict[str, Any]):
    """
    Read path if it exists, merge in new_data (overwriting existing keys),
    and write back.
    """
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing.update(new_data)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

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
