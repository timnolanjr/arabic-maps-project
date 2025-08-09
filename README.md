 Arabic Maps Project

A digital‑humanities toolkit for **large‑scale analysis of circular world maps** in medieval Arabic manuscripts. The project turns visual variation into data by detecting each map’s circular frame, identifying a reference edge, computing the tangent point, and (optionally) detecting Arabic text regions to support OCR and toponym analysis.

> Core idea: detect **circle → edge → tangent point**, then project labels into a consistent polar frame for cross‑manuscript comparison.

---

## Contents
- [Motivation](#motivation)
- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Data layout](#data-layout)
- [Outputs](#outputs)
- [CLI usage](#cli-usage)
- [Text detection (beta)](#text-detection-beta)
- [Configuration hints](#configuration-hints)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Motivation

Medieval Arabic circular maps (often with Mecca at the top) vary subtly by copyist and workshop. Differences in framing, scale, and label placement encode transmission histories and local practices. By standardizing geometry and extracting labels, we can **compare maps quantitatively** rather than only descriptively.

![Al-Idrīsī map, original (left) and flipped modern orientation (right)](assets/images/al-idrisi-1001inventions.jpg)

*Left: Original Al-Idrīsī 12th century map with Mecca at the top and Europe in the lower right.  
Right: Flipped view to match Western-style north-up orientation.*  [\[1\]](https://www.1001inventions.com/maps/)

This repo provides:
- A reproducible **geometric baseline** (circle/edge/tangent) per image.
- Lightweight **text detectors** to localize toponyms for OCR.
- A **unified pipeline** and small CLIs that save rich per‑image JSON.

---

## Highlights

- **Metadata extraction**: image type, size, dimensions, color space.
- **Circle detection** (`src/circle.py`): Hough‑based, with interactive refinement and batch candidate review.
- **Edge detection** (`src/edges.py`): Canny + Hough for (near‑horizontal) top edge; ROI‑guided interactive mode available.
- **Tangent computation** (`src/utils/tangent.py`): computes the upper tangent point from circle + edge parameters.
- **Text detection (beta)** (`src/text_detection.py`): morphology, MSER, Canny, Sobel, and gradient variants; optional NMS and geometric/stroke‑width filters.
- **Unified pipeline** (`pipeline.py`): runs metadata → circle → edge → tangent and writes overlays.
- **CLIs** in `scripts/`: interactive and batch tools for each stage.

---

## Quick start

```bash
# 1) Create and activate a virtual environment (example with venv)
python -m venv .venv
source .venv/bin/activate

# 2) Install
pip install --upgrade pip
pip install -r requirements.txt

# 3) Put raw images in data/raw_maps/
#    e.g., data/raw_maps/al-Qazwini_Arabic_MSS_575.jpg

# 4) Run the full geometry pipeline (non-interactive, prompts per image)
python pipeline.py data/raw_maps
```

You will get per‑image folders under `data/processed_maps/<image_stem>/` with a `params.json` and visual overlays.

---

## Installation

```bash
git clone https://github.com/timnolanjr/arabic-maps-project.git
cd arabic-maps-project
pip install -r requirements.txt
```

**Requirements (excerpt from `requirements.txt`):**
- numpy<2.0, pandas>=1.5, opencv-python>=4.11
- matplotlib, Pillow, tqdm
- (Optional OCR stack) mmcv/mmengine/mmdet/mmocr, easyocr, paddleocr
- PyTorch versions pinned for compatibility in `requirements.txt`

Python 3.9–3.11 is recommended. GPU is **not required** for the geometry steps.

---

## Data layout

```
arabic-maps-project/
├── data/
│   ├── raw_maps/            # your input TIFF/JPG/PNG scans
│   └── processed_maps/      # auto-created outputs (per-image folders)
├── scripts/                 # helper CLIs
├── src/                     # library code
│   ├── circle.py            # circle detection & review
│   ├── edges.py             # top-edge detection (batch/interactive)
│   ├── text_detection.py    # text detectors (beta)
│   └── utils/
│       ├── image.py         # grayscale & blur helpers + display
│       ├── io.py            # json I/O & per-image output dirs
│       ├── interactive.py   # point-click helpers
│       ├── metadata.py      # params.json init from image
│       └── tangent.py       # tangent point computation
├── pipeline.py              # unified multi-image pipeline
├── requirements.txt
└── README.md
```

---

## Outputs

For each input image `data/raw_maps/NAME.EXT`, the pipeline writes to `data/processed_maps/NAME/`:

- `params.json` – single source of truth for derived geometry and metadata:
  ```json
  {
    "filetype": "jpg",
    "filesize": 1234567,
    "image_width": 3000,
    "image_height": 3000,
    "colorspace": "sRGB",
    "center_x": 1500,
    "center_y": 1500,
    "radius": 1200,
    "rho": 850.1,
    "theta": 1.57,
    "tangent_x": 1500.0,
    "tangent_y": 300.0
  }
  ```
- `circle_final.jpg` / `edge_final.jpg` / `edge_overlay.jpg` – visualizations (depending on interactive/batch path).
- `final_overlay.jpg` – combined overlay with circle, edge, cardinal markers, and annotation (created in `pipeline.py`).

![Al-Qazwīnī Map Example](assets/images/al-Qazwīnī_Arabic_MSS_575.jpg)

*Map extracted from a sixteenth-century Ottoman manuscript of Persian cosmographer and geographer al-Qazwīnī’s “The Marvels of Creation.”*


---

## CLI usage

### 1) Unified pipeline
Process a directory of images (prompts per file; skips completed work):
```bash
python pipeline.py data/raw_maps -o data/processed_maps
```

### 2) Circle detection
Batch candidates (non‑interactive), then review interactively:
```bash
# Generate candidates
python scripts/test_circle.py data/raw_maps/map1.jpg

# Review and pick final circle
python scripts/pick_circle.py data/raw_maps/map1.jpg -o data/processed_maps
```

Or fully interactive with perimeter clicks + ROI refinement:
```bash
python scripts/test_circle.py data/raw_maps/map1.jpg --interactive -o data/processed_maps
```

### 3) Edge detection
Batch mode (global Canny+Hough with horizontal‑angle filter):
```bash
python scripts/test_edges.py data/raw_maps/map1.jpg -o data/processed_maps
```

    Interactive, ROI‑guided (click ~3 points along the top edge, then refine):
```bash
python scripts/test_edges.py data/raw_maps/map1.jpg --interactive -o data/processed_maps
```

### 4) Text detection (beta)
Run a detector over a glob of images and save drawn boxes per image:
```bash
# Methods: morph | mser | canny | sobel | gradient
python scripts/test_text_detection.py "data/raw_maps/*.jpg" --method mser -o data/processed_maps
```

Common flags (see `scripts/test_text_detection.py` for full list):
- `--interactive` – show a window for each result instead of saving.
- `--nms-filter` – apply non‑max suppression to reduce overlapping boxes.

**MSER‑specific flags (subset):**
```
--mser-delta 5
--mser-min-area 60
--mser-max-area 14400
--mser-max-variation 0.3
--mser-min-diversity 0.2
--mser-max-evolution 1000
--mser-area-threshold 1.01
--mser-min-margin 0.003
--mser-edge-blur-size 3
--mser-geom-filter                # enable geometry filter
--mser-sw-filter                  # enable stroke-width stability filter
--mser-sw-threshold 0.4
```

---

## Text detection (beta)

All methods share a common interface and produce a list of bounding boxes `(x, y, w, h)`. Available detectors:
- **Morphology:** adaptive threshold + dilation → contours.
- **MSER:** region stability; optional **geometry** filter (aspect ratio, eccentricity, solidity, extent, Euler number) and optional **stroke‑width stability** filter (skeleton + distance transform statistics).
- **Canny/Sobel/Gradient:** edge/gradient maps + morphology → contours.

Use `--nms-filter` to reduce overlaps. Expect **false positives**; these detectors are intended as **recall‑oriented** proposals ahead of OCR or human vetting.

---

## Configuration hints

- Save global defaults (e.g., MSER thresholds) in a small YAML and load them in your scripts for reproducibility.
- For very large scans, downscale to a working resolution (e.g., ~2–3k px short side) before detection to stabilize parameters and speed up previews.
- Keep `params.json` authoritative; downstream steps should only append keys (no destructive edits).

---

## Development

- Code lives under `src/`. Utilities are in `src/utils/`.
- Prefer **pure functions** that accept arrays/paths and return data/arrays.
- Use **type hints** and **docstrings** consistently.
- Use `logging` instead of `print` for pipelines that may run headless.
- Consider `pyproject.toml` packaging with console entry points for CLIs.

---

## Testing

- `pytest` is set to include `src/` on the Python path (`pytest.ini`).
- Add unit tests for:
  - geometric helpers (fit, endpoints, conversions),
  - text detector post‑filters,
  - JSON I/O (idempotent update/merge semantics).

---

## Troubleshooting

- **Matplotlib windows don’t show / freeze:** use a non‑headless backend, or run interactive scripts locally (not in headless servers). For batch work, omit `--interactive`.
- **No circles found:** check image contrast; try *interactive circle* mode to seed fit; widen Hough thresholds or use candidate generation + review.
- **Edge detection unstable:** use *interactive ROI* mode and increase `--delta` to widen the search band.
- **Text detection too many boxes:** enable `--nms-filter`, then try `--mser-geom-filter` and/or `--mser-sw-filter`.
- **Large TIFFs:** convert to JPEG/PNG for experimentation; keep originals for final runs.

---

## Roadmap

1. **Detection & OCR**
   - Add deep‑learning text detectors (e.g., DBNet/PSENet via MMOCR) as optional backends.
   - Integrate Arabic OCR (EasyOCR, PaddleOCR, MMOCR recognizers) with post‑processing.
   - Automate **toponym normalization & transliteration**; build gazetteer links.

2. **Geometry & Alignment**
   - Export polar coordinates for boxes/labels using circle + tangent reference.
   - Implement multi‑map **similarity metrics** (shape, label placement) and clustering.

3. **Productivity & DX**
   - Central configuration file; dataset manifests.
   - Rich logging; progress bars; resumable batch runs.

4. **Visualization & Sharing**
   - Streamlit dashboard for per‑map review and cross‑map comparison.
   - Optional Google Sheets/Drive sync for metadata status tracking.

---

## License

MIT License.

