from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import json
import cv2

from src.utils.metadata import init_params_from_image
from src.circle import interactive_detect_and_save as detect_circle
from src.edges import interactive_detect_and_save as detect_edge
from src.utils.tangent import compute_tangent_point
from src.utils.vis import compose_geometry_overlay
from src.utils.palette import DEFAULT_PALETTE


# Treat a map as "complete" if these keys are present and non-None AND an overlay exists
REQUIRED_PARAM_KEYS = {
    "center_x", "center_y", "radius",
    "rho", "theta",
    "tangent_x", "tangent_y",
}

def _out_dir_for(img_path: Path, out_base: Optional[Path]) -> Path:
    return (Path("processed_maps") / img_path.stem) if out_base is None else (out_base / img_path.stem)

def _is_params_complete(img_path: Path, out_base: Optional[Path]) -> bool:
    out_dir = _out_dir_for(img_path, out_base)
    params_path = out_dir / "params.json"
    overlay_jpg = out_dir / "params_overlay.jpg"
    overlay_png = out_dir / "params_overlay.png"

    if not params_path.exists():
        return False

    try:
        data = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    # all required keys present and non-None
    for k in REQUIRED_PARAM_KEYS:
        if data.get(k) is None:
            return False

    # overlay must exist
    return overlay_jpg.exists() or overlay_png.exists()

def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                yield p
    elif path.is_file():
        yield path
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")


def _init_params(img_path: Path, out_dir: Path) -> dict:
    """
    Wrap init_params_from_image so we ALWAYS return a dict.
    """
    try:
        out = init_params_from_image(img_path, out_dir)  # some versions accept out_dir
    except TypeError:
        out = init_params_from_image(img_path)           # older signature
    except Exception as e:
        print(f"[meta] init_params_from_image failed: {e}", flush=True)
        out = {}

    if not isinstance(out, dict):
        # Some implementations just write params.json and return None
        out = {}
    return out


def process_image(img_path: Path, out_base: Optional[Path], *, interactive: bool = False) -> Path:
    """
    Process one image. If out_base is None, defaults to processed_maps/<stem>.
    """
    if out_base is None:
        out_dir = Path("data/processed_maps") / img_path.stem
    else:
        out_dir = out_base / img_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    params_path = out_dir / "params.json"
    overlay_path = out_dir / "params_overlay.jpg"

    # 1) Init params
    print("[meta] initializing params…", flush=True)
    params = _init_params(img_path, out_dir)

    # 2) Circle
    print("[circle] detecting…", flush=True)
    circle = detect_circle(img_path, out_dir=out_dir, interactive=interactive)
    if circle:
        params.update(circle)

    # 3) Edge
    print("[edge] detecting…", flush=True)
    edge = detect_edge(img_path, out_dir=out_dir, interactive=interactive)
    if edge:
        params.update(edge)

    # 4) Tangent: always choose the TOP tangent point
    print("[tangent] computing…", flush=True)
    keys = {"center_x", "center_y", "radius", "rho", "theta"}
    if keys.issubset(params.keys()):
        tx, ty = compute_tangent_point(
            float(params["center_x"]),
            float(params["center_y"]),
            float(params["radius"]),
            float(params["rho"]),
            float(params["theta"]),
            prefer_top=True, 
        )
        params.update({"tangent_x": float(tx), "tangent_y": float(ty)})

    # Save params.json
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # 5) Draw overlay with unified palette
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    overlay = compose_geometry_overlay(bgr, params, palette=DEFAULT_PALETTE)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"  ✓ wrote {overlay_path}", flush=True)

    return out_dir


def run_pipeline(input_path: Path, out_base: Optional[Path] = None, *, interactive: bool = False) -> None:
    """
    Run pipeline over a file or directory. If out_base is None, defaults to processed_maps/.
    When input is a directory, ask per image whether to skip if it's already complete.
    """
    if out_base is not None:
        out_base.mkdir(parents=True, exist_ok=True)

    # Single file: behave as before
    if input_path.is_file():
        print(f"[1/1] {input_path.name}")
        out_dir = process_image(input_path, out_base, interactive=interactive)
        print(f"  → wrote {out_dir}")
        return

    # Directory: iterate and ask per-image
    imgs = list(_iter_images(input_path))
    total = len(imgs)
    if total == 0:
        print(f"No images found in {input_path}")
        return

    for idx, img_path in enumerate(imgs, start=1):
        print(f"[{idx}/{total}] {img_path.name}", flush=True)

        if _is_params_complete(img_path, out_base):
            if interactive:
                resp = input("  Already complete — skip? [Y/n]: ").strip().lower()
                if resp in {"", "y", "yes"}:
                    print("  ↷ skipped (complete).", flush=True)
                    continue
            else:
                print("  ↷ skipped (complete).", flush=True)
                continue

        out_dir = process_image(img_path, out_base, interactive=interactive)
        print(f"  → wrote {out_dir}", flush=True)
