import json
from pathlib import Path
from typing import Dict, Any

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
