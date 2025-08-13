from __future__ import annotations
from pathlib import Path
import re
from typing import Tuple

def next_numbered_variant(path: Path) -> Path:
    """
    Generic non-overwriting helper (kept for backwards compatibility).
    foo.jpg -> foo.jpg, foo_0001.jpg, foo_0002.jpg, ...
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".jpg")
    if not path.exists():
        return path
    stem, ext = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i:04d}{ext}")
        if not candidate.exists():
            return candidate
        i += 1

def next_runpair_paths(
    out_dir: Path,
    stem: str,
    method: str,
    *,
    img_ext: str = ".jpg",
    json_ext: str = ".json",
    width: int = 4,
    start: int = 1,
) -> Tuple[Path, Path, int]:
    """
    Return (img_path, json_path, idx) such that both files share the same
    zero-padded index: <stem>_text_<method>_run_XXXX.<ext>

    Chooses the next index not used by either the jpg or json variant.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize extensions
    if not img_ext.startswith("."):
        img_ext = "." + img_ext
    if not json_ext.startswith("."):
        json_ext = "." + json_ext

    pat_img = re.compile(
        rf"^{re.escape(stem)}_text_{re.escape(method)}_run_(\d{{{width}}}){re.escape(img_ext)}$"
    )
    pat_json = re.compile(
        rf"^{re.escape(stem)}_text_{re.escape(method)}_run_(\d{{{width}}}){re.escape(json_ext)}$"
    )

    indices = set()
    for p in out_dir.glob(f"{stem}_text_{method}_run_*{img_ext}"):
        m = pat_img.match(p.name)
        if m:
            indices.add(int(m.group(1)))
    for p in out_dir.glob(f"{stem}_text_{method}_run_*{json_ext}"):
        m = pat_json.match(p.name)
        if m:
            indices.add(int(m.group(1)))

    idx = (max(indices) + 1) if indices else start
    tag = f"{idx:0{width}d}"
    img_path = out_dir / f"{stem}_text_{method}_run_{tag}{img_ext}"
    json_path = out_dir / f"{stem}_text_{method}_run_{tag}{json_ext}"
    return img_path, json_path, idx