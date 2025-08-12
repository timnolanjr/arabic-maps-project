from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from src.pipeline_core import run_pipeline
from src.circle import interactive_detect_and_save as detect_circle
from src.edges import interactive_detect_and_save as detect_edge
from src.utils.vis import draw_text_boxes, draw_legend
from src.utils.palette import DEFAULT_PALETTE


def _path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def cmd_pipeline(args: argparse.Namespace) -> None:
    run_pipeline(_path(args.input), _path(args.out), interactive=args.interactive)


def cmd_circle(args: argparse.Namespace) -> None:
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detect_circle(_path(args.image), out_dir=out_dir, interactive=args.interactive)


def cmd_edges(args: argparse.Namespace) -> None:
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detect_edge(_path(args.image), out_dir=out_dir, interactive=args.interactive)


def cmd_text(args: argparse.Namespace) -> None:
    img_path = _path(args.image)
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so other subcommands don't require text_detection.py
    try:
        from src.text_detection import detect_text_regions  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Could not import 'detect_text_regions' from src.text_detection.\n"
            "Ensure src/text_detection.py defines:\n"
            "  detect_text_regions(bgr: np.ndarray, method: str, **kwargs) -> List[(x,y,w,h)]\n"
            f"Original import error: {e}"
        )

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {img_path}")

    boxes = detect_text_regions(
        bgr,
        method=args.method,
        nms=args.nms_filter,
        geom_filter=args.geom_filter,
        sw_filter=args.sw_filter,
        sw_threshold=args.sw_threshold,
        mser_delta=args.mser_delta,
        mser_min_area=args.mser_min_area,
        mser_max_area=args.mser_max_area,
        mser_max_variation=args.mser_max_variation,
        mser_min_diversity=args.mser_min_diversity,
    )

    overlay = bgr.copy()
    draw_text_boxes(overlay, boxes, palette=DEFAULT_PALETTE)
    draw_legend(overlay, palette=DEFAULT_PALETTE)
    out_path = out_dir / f"{img_path.stem}_text_{args.method}.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f" → wrote {out_path}")


def cmd_overlay(args: argparse.Namespace) -> None:
    from src.utils.overlay import render_overlay_from_paths
    img = _path(args.image)
    params = _path(args.params)
    out = _path(args.out)
    lang = args.lang
    render_overlay_from_paths(
        img, params, out,
        include_legend=not args.no_legend,
        include_text_boxes=not args.no_textboxes,
        lang=lang,
    )
    print(f" → wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arabmaps", description="Arabic circular maps toolkit CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pipeline", help="Run geometry pipeline on a file or directory")
    sp.add_argument("input", help="Image file or directory")
    sp.add_argument("-o", "--out", default="data/processed_maps", help="Output base directory")
    sp.add_argument("--interactive", action="store_true", help="Interactive circle/edge picking")
    sp.set_defaults(func=cmd_pipeline)

    sc = sub.add_parser("circle", help="Circle detection (interactive/non-interactive)")
    sc.add_argument("image", help="Image file")
    sc.add_argument("-o", "--out", default="data/processed_maps", help="Output directory")
    sc.add_argument("--interactive", action="store_true")
    sc.set_defaults(func=cmd_circle)

    se = sub.add_parser("edges", help="Edge detection (interactive/non-interactive)")
    se.add_argument("image", help="Image file")
    se.add_argument("-o", "--out", default="data/processed_maps", help="Output directory")
    se.add_argument("--interactive", action="store_true")
    se.set_defaults(func=cmd_edges)

    st = sub.add_parser("text", help="Run a text detector and save an overlay with boxes")
    st.add_argument("image", help="Image file")
    st.add_argument("-o", "--out", default="data/processed_maps", help="Output directory")
    st.add_argument("--method", default="mser", choices=["morph", "mser", "canny", "sobel", "gradient"])
    st.add_argument("--nms-filter", dest="nms_filter", action="store_true")
    st.add_argument("--geom-filter", action="store_true")
    st.add_argument("--sw-filter", action="store_true")
    st.add_argument("--sw-threshold", type=float, default=0.4)
    # MSER parameters (subset)
    st.add_argument("--mser-delta", type=int, default=5)
    st.add_argument("--mser-min-area", type=int, default=60)
    st.add_argument("--mser-max-area", type=int, default=14400)
    st.add_argument("--mser-max-variation", type=float, default=0.3)
    st.add_argument("--mser-min-diversity", type=float, default=0.2)
    st.set_defaults(func=cmd_text)

    so = sub.add_parser("overlay", help="Compose a final overlay from image + params.json")
    so.add_argument("image", help="Image file")
    so.add_argument("-p", "--params", required=True, help="Path to params.json")
    so.add_argument("-o", "--out", required=True, help="Output overlay path (e.g., assets/images/final_overlay.jpg)")
    so.add_argument("--lang", choices=["en", "ar"], default="en", help="Legend labels language")
    so.add_argument("--no-legend", action="store_true", help="Do not draw legend")
    so.add_argument("--no-textboxes", action="store_true", help="Do not draw text boxes even if present")
    so.set_defaults(func=cmd_overlay)

    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
