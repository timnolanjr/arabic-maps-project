#!/usr/bin/env python3
from __future__ import annotations
import argparse, shlex, subprocess, sys
from pathlib import Path

BASELINES = {
    # Keep things minimal to isolate MSER behavior
    "defaults": dict(
        polarity="both",
        multiscale=False,
        scales=(),               # empty → no --scales flag
        morph=False,
        morph_close=3,
        morph_open=0,
        morph_iter=1,
        delta=5,
        min_area=60,
        max_area=14400,
        geom=False,              # geometry filter OFF by default
        geom_area_min_pct=0.00005,
        geom_area_max_pct=0.02,
        geom_ar_min=0.15,
        geom_ar_max=8.0,
        nms=False,               # NMS OFF by default
        nms_iou=0.3,
        circle=False,
        circle_min_cover=0.5,
        box_thickness=1,
    ),
    # Closer to “paper-like” exploratory settings:
    # - multiscale on, modest pyramid
    # - morphology closing to merge fragments
    # - geometry+NMS on (note: this can confound variable isolation)
    "paperish": dict(
        polarity="both",
        multiscale=True,
        scales=(1.0, 0.75),
        morph=True,
        morph_close=3,
        morph_open=0,
        morph_iter=1,
        delta=5,
        min_area=60,
        max_area=14400,
        geom=True,
        geom_area_min_pct=0.00005,
        geom_area_max_pct=0.02,
        geom_ar_min=0.15,
        geom_ar_max=8.0,
        nms=True,
        nms_iou=0.3,
        circle=False,
        circle_min_cover=0.5,
        box_thickness=1,
    ),
}

# Which single variables can we sweep?
CHOICES = {
    # MSER core
    "delta", "min_area", "max_area", "polarity",
    # Multiscale
    "scales",
    # Morph
    "morph_close", "morph_open", "morph_iter",
    # Geometry
    "geom_area_min_pct", "geom_area_max_pct", "geom_ar_min", "geom_ar_max",
    # NMS
    "nms_iou",
    # Circle (needs params.json)
    "circle_min_cover",
}

def _parse_values(var: str, values: str):
    """
    Parse --values into a list appropriate for the variable.
    - numeric vars: comma-separated numbers (floats allowed)
    - 'polarity': comma-separated in {dark,bright,both}
    - 'scales': semicolon-separated lists of comma-separated floats, e.g.
        '1.0;1.0,0.75;1.0,0.5'
    """
    var = var.lower().strip()
    if var == "polarity":
        return [v.strip().lower() for v in values.split(",") if v.strip()]
    if var == "scales":
        # split into sets: '1.0;1.0,0.75' → ['1.0', '1.0,0.75']
        sets = [s.strip() for s in values.split(";") if s.strip()]
        parsed = []
        for s in sets:
            parsed.append(tuple(float(x) for x in s.split(",") if x.strip()))
        return parsed
    # numeric
    out = []
    for tok in values.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # allow ints or floats
        out.append(float(tok) if ("." in tok or "e" in tok.lower()) else int(tok))
    return out

def _base_flags(base: dict, args) -> list[str]:
    """Translate a baseline dict into CLI flags (excluding the swept var)."""
    flags: list[str] = []
    # Polarity
    flags += ["--polarity", str(base["polarity"])]

    # Multiscale
    if base["multiscale"] or (args.var == "scales"):
        flags += ["--multiscale"]
    if base["scales"]:
        flags += ["--scales", ",".join(str(s) for s in base["scales"])]

    # Morph
    if base["morph"] or (args.var in {"morph_close", "morph_open", "morph_iter"}):
        flags += ["--morph",
                  "--morph-close", str(base["morph_close"]),
                  "--morph-open",  str(base["morph_open"]),
                  "--morph-iter",  str(base["morph_iter"])]

    # MSER numeric
    flags += ["--mser-delta",         str(base["delta"]),
              "--mser-min-area",      str(base["min_area"]),
              "--mser-max-area",      str(base["max_area"])]

    # Geometry
    if base["geom"] or (args.var in {"geom_area_min_pct", "geom_area_max_pct", "geom_ar_min", "geom_ar_max"}):
        # geometry ON
        # (turn off by adding --no-geom-filter otherwise)
        pass
    else:
        flags += ["--no-geom-filter"]

    # NMS
    if base["nms"] or (args.var == "nms_iou"):
        # NMS ON unless explicitly disabled elsewhere
        pass
    else:
        flags += ["--no-nms"]

    # Circle
    # Only enable circle filter when sweeping it (or baseline asked for it)
    if base["circle"] or (args.var == "circle_min_cover"):
        flags += ["--circle-filter", "--circle-min-cover", str(base["circle_min_cover"])]

    # Visual
    if base.get("box_thickness"):
        flags += ["--overlay-box-thickness", str(base["box_thickness"])]

    # Stamp params + runmeta (your CLI pairs run JPG/JSON automatically)
    flags += ["--stamp-params"]
    return flags

def _apply_var(flags: list[str], var: str, val, args) -> list[str]:
    """Add/override flags for the swept variable."""
    var = var.lower()
    if var == "delta":
        return flags + ["--mser-delta", str(val)]
    if var == "min_area":
        return flags + ["--mser-min-area", str(val)]
    if var == "max_area":
        return flags + ["--mser-max-area", str(val)]
    if var == "polarity":
        return flags + ["--polarity", str(val)]
    if var == "scales":
        return flags + ["--multiscale", "--scales", ",".join(f"{x:g}" for x in val)]
    if var in {"morph_close", "morph_open", "morph_iter"}:
        flags = [f for f in flags if f != "--morph"]  # ensure present once
        return ["--morph"] + flags + [f"--{var.replace('_','-')}", str(val)]
    if var in {"geom_area_min_pct", "geom_area_max_pct", "geom_ar_min", "geom_ar_max"}:
        # Geometry ON
        mapping = {
            "geom_area_min_pct": "--geom-area-min-pct",
            "geom_area_max_pct": "--geom-area-max-pct",
            "geom_ar_min": "--geom-ar-min",
            "geom_ar_max": "--geom-ar-max",
        }
        # Remove possible '--no-geom-filter'
        flags = [f for f in flags if f != "--no-geom-filter"]
        return flags + [mapping[var], str(val)]
    if var == "nms_iou":
        # NMS ON
        flags = [f for f in flags if f != "--no-nms"]
        return flags + ["--nms-iou", str(val)]
    if var == "circle_min_cover":
        # requires --params path
        flags = [f for f in flags if f != "--circle-filter"]
        flags = ["--circle-filter"] + flags
        return flags + ["--circle-min-cover", str(val)]
    raise ValueError(f"Unsupported var: {var}")

def main():
    ap = argparse.ArgumentParser(description="One-variable MSER sweep (paired JPG/JSON outputs).")
    ap.add_argument("--image", required=True, help="Image path")
    ap.add_argument("-o", "--out", default="data/processed_maps", help="Output base directory")
    ap.add_argument("--params", help="params.json path (needed if sweeping circle_min_cover)")
    ap.add_argument("--baseline", choices=("defaults","paperish"), default="defaults")
    ap.add_argument("--var", required=True, help=f"Variable to sweep. Choices: {', '.join(sorted(CHOICES))}")
    ap.add_argument("--values", required=True,
                    help="Values to try. For 'scales': semicolon-separated sets, e.g. '1.0;1.0,0.75;1.0,0.5'. "
                         "For others: comma-separated numbers or 'dark,bright,both' for polarity.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.var not in CHOICES:
        ap.error(f"--var must be one of: {', '.join(sorted(CHOICES))}")

    if args.var == "circle_min_cover" and not args.params:
        ap.error("--params is required when sweeping circle_min_cover")

    img = Path(args.image)
    if not img.exists():
        ap.error(f"Image not found: {img}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vals = _parse_values(args.var, args.values)
    if not vals:
        ap.error("No values parsed from --values")

    base = BASELINES[args.baseline]
    # Build common prefix
    common = ["python", "-m", "src.cli", "text", str(img), "-o", str(out_dir), "--method", "mser"]
    if args.params:
        common += ["--params", args.params]

    base_flags = _base_flags(base, args)

    for v in vals:
        run_id = f"{args.baseline}-{args.var}={v if args.var!='scales' else 'x'.join(str(x) for x in v)}"
        flags = _apply_var(base_flags[:], args.var, v, args)
        cmd = common + flags + ["--run-id", run_id]
        print("→", " ".join(shlex.quote(c) for c in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
