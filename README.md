# Arabic Maps Project

A digital-humanities toolkit for **large-scale analysis of circular world maps** in medieval Arabic manuscripts. Currently, the project detects each map’s circular frame, detects the orientation of manuscript's upper edge, computes the tangent point, and detects Arabic text regions to support OCR and toponym analysis.

---

## Contents
- [Motivation](#motivation)
- [Highlights](#highlights)
- [Example Results](#example-results)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Data layout](#data-layout)
- [CLI usage](#cli-usage)
- [Outputs](#outputs)
- [Roadmap](#roadmap)
- [Development](#development)
- [License](#license)

---

## Motivation

Medieval Arabic circular world maps—often south-up, with Mecca near the top—vary by copyist and time. Shifts in framing, scale, orientation, and label placement may encode transmission histories and local practices, but they’re hard to compare by eye.

This toolkit makes those choices **measurable**. It detects a map’s circular frame, establishes a common south-up orientation via a reference edge and tangent line, and yields a polar coordinate system. With standardized geometry, we can **compare maps quantitatively** across authors, manuscripts, and centuries to trace workshop influence, reuse, and change over time.

![Al-Idrīsī map, original (left) and flipped modern orientation (right)](assets/images/al-idrisi-1001inventions.jpg)

*Left: Original Al-Idrīsī 12th-century map with Mecca at the top and Europe in the lower right.  
Right: Flipped view to match a north-up orientation.* [\[1\]](https://www.1001inventions.com/maps/)

---

## Highlights

- **Unified CLI** (`src/cli.py`) with subcommands:
  - `pipeline` → metadata → circle → edges → tangent (writes `params.json` + overlay)
  - `circle` / `edges` → run geometry steps individually (interactive or batch)
  - `text` → **MSER** text detection (recall-first), with optional multiscale, morphology merge, circle coverage, geometry filter, and NMS; draws a legend and can stamp a param summary in the overlay.
- **Geometry**
  - `src/circle.py` (Hough + interactive refinement)
  - `src/edges.py` (Canny+Hough; near-horizontal top edge; interactive ROI mode)
  - `src/utils/tangent.py` (tangent “South” point from circle + edge)
- **Text (modular stack in `src/text/`)**
  - `backends/mser.py` (OpenCV MSER with robust param wiring)
  - `filters/{geometry,circle,nms,morphology}.py`
  - `visualize.py` (stage legend UL, optional param title TR)
  - `config.py` (dataclasses for MSER, channels, multiscale, morphology, geometry, circle, NMS, visual)
  - `detect.py` (orchestrator)
- **Paired outputs** use `src/utils/naming.py` to write:
  ```
  processed_maps/<stem>/<stem>_text_<method>_run_XXXX.jpg
  processed_maps/<stem>/<stem>_text_<method>_run_XXXX.json
  ```


## Example Results

### Map Preprocessing

![Al-Qazwīnī Map Example](assets/images/al-Qazwīnī_Arabic_MSS_575.jpg)

*Map extracted from a sixteenth-century Ottoman manuscript of Persian cosmographer and geographer al-Qazwīnī’s “The Marvels of Creation.”*

We run the preprocessing pipeline on al-Qazwīnī’s work with the following code snippet in the terminal:

```bash
python pipeline.py "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg"
```

![Parameter Overlay Map Example](assets/images/params_overlay.jpg)


*`params_overlay.jpg` depicting geometric parameters (circle, center, top-edge, tangent South / جنوب / الجَنوب, cardinal directions*

So, we can successfully detect the circular frame of the map and the point on the map we call South / جنوب / الجَنوب by finding the circle's tangent point to the manuscript edge. Also returns `params.json` (see: [Outputs](#outputs)).

### Text Detection and Optical Character Recognition

The current challenge is to extract the toponyms from each map, an exploration in Optical Character Recognition (OCR). Some things that make this process non-trivial are:

- The vast majority of OCR literature and text detection methods focus on English and Left-to-Right reading systems.
- No models are tuned to our use-case - most train on natural scenes (images, not maps) or more strucuted documents (e.g., books).
- We need a system that is precise (few false positives), but more importantly one that has high recall (few false *negatives*).
- I personally don't speak Arabic so I anticipate a bottleneck. May need to recruit JB to tune the model once we've got a better plan.

![Text Detect Overlay Map Example](assets/images/al-Qazwīnī_Arabic_MSS_575_mser.jpg)

*`_mser.jpg` depicting text detection's current state of affairs: Catching >90% of all text with MSER method, with many false-positives.*

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

You will get per-image folders under `data/processed_maps/<image_stem>/` with a `params.json` and visual overlays.

---

## CLI usage

### Full Pipeline

Runs metadata → circle → edge → tangent; writes per-image outputs under `data/processed_maps/img/`
```
python -m src.cli pipeline data/raw_maps -o data/processed_maps
```

Pipeline over a single file (non-interactive)
```
python -m src.cli pipeline "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg" -o data/processed_maps
```

Interactive pipeline: click perimeter points (circle) and ~3 points along top edge (edge)
```
python -m src.cli pipeline "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg" -o data/processed_maps --interactive
```


### Circle Detection

Non-interactive (Hough-based) circle detection; updates `params.json`
```
python -m src.cli circle "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg" -o data/processed_maps
```

Interactive circle detection: click N points on the perimeter, then choose a refined candidate
```
python -m src.cli circle "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg" -o data/processed_maps --interactive
```

### Top Edge Detection

Non-interactive (Canny+Hough) top-edge detection; updates `params.json`
```
python -m src.cli edges  "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg" -o data/processed_maps
```

Interactive edge detection: click ~3 points along the manuscript’s upper edge, then refine in an ROI band
```
python -m src.cli edges  "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg" -o data/processed_maps --interactive
```

### Text Detection

Method choices: `morph | mser | canny | sobel | gradient`
```
python -m src.cli text   "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg" -o data/processed_maps --method mser
```

**Core MSER knobs**:
- `--method mser`
- `--polarity {dark,bright,both}` (we default to **both**)
- `--channels gray[,r,g,b,labL]` (union of per-channel detections)
- `--multiscale --scales s1,s2[,s3]` (union of per-scale detections)
- `--mser-delta INT` (default 5)
- `--mser-min-area INT` (pixels; e.g., 60)
- `--mser-max-area INT` (pixels; e.g., 14400)
- `--mser-max-variation FLOAT` (default ~0.3)
- `--mser-min-diversity FLOAT` (default ~0.2)
- `--mser-max-evolution INT` (default 200)
- `--mser-area-threshold FLOAT` (default 1.01)
- `--mser-min-margin FLOAT` (default 0.003)
- `--mser-edge-blur-size INT` (e.g., 3–5)

**Optional post-filters**:
- **Morphology merge:** `--morph --morph-close 3 --morph-open 0 --morph-iter 1`
- **Circle coverage:** `--circle-filter --params <params.json> --circle-min-cover 0.5` \
  *(optional tolerance: `--circle-grow 1.02` if enabled)*
- **Geometry filter:** `--geom-area-min-pct 5e-5 --geom-area-max-pct 0.02 --geom-ar-min 0.15 --geom-ar-max 8.0` \
  (use `--no-geom-filter` to disable)
- **NMS:** `--nms-iou 0.3` (use `--no-nms` to disable)


### Image Overlay Only

Writes a final overlay (with legend + metadata) to the given path.
```
python -m src.cli overlay "data/raw_maps/al-Qazwīnī_Arabic_MSS_575.jpg" \
  -p "data/processed_maps/al-Qazwīnī_Arabic_MSS_575/params.json" \
  -o assets/images/final_overlay.jpg \
  --lang en
```
---

## Data layout

```
arabic-maps-project/
├── data/
│   ├── raw_maps/                         # input scans (TIFF/JPG/PNG), south-up
│   └── processed_maps/<stem>/            # per-image outputs
│       ├── params.json                   # center/radius, rho/theta, tangent, metadata
│       ├── params_overlay.jpg            # circle + edge + tangent (legend UL)
│       ├── <stem>_text_mser_run_0001.jpg
│       └── <stem>_text_mser_run_0001.json
├── src/
│   ├── cli.py                            # CLI: pipeline, circle, edges, text
│   ├── circle.py                         # circle detection (batch/interactive)
│   ├── edges.py                          # top-edge detection (batch/interactive)
│   ├── utils/                            # overlay, IO, palette, naming, etc.
│   └── text/
│       ├── backends/mser.py              # MSER backend
│       ├── filters/{geometry,circle,nms,morphology}.py
│       ├── visualize.py                  # stage overlays (legend/counts)
│       ├── config.py                     # dataclasses for all knobs
│       └── detect.py                     # minimal orchestrator (for extensions)
├── scripts/
│   └── sweep_mser.py                     # 1-var sweeps with paired outputs
└── README.md
```

---

## Outputs

For each input image `data/raw_maps/NAME.EXT`, the pipeline writes to `data/processed_maps/NAME/`:

- `params.json` – single source of truth for derived geometry and metadata (these are from `assets/images/al-Qazwīnī_Arabic_MSS_575.jpg`):
  ```json
  {
  "filetype": "jpg",
  "filesize": 6121401,
  "image_width": 6760,
  "image_height": 5080,
  "colorspace": "sRGB",
  "center_x": 3358,
  "center_y": 2528,
  "radius": 2027,
  "rho": 59.0,
  "theta": 1.5702727078651233,
  "tangent_x": 3356.93862447785,
  "tangent_y": 501.00027787816384
} 
- `params_overlay.jpg` – combined overlay with circle, edge, cardinal markers, and annotation.


---

## Roadmap

1. **Detection & OCR**
   - Add deep-learning text detectors (e.g., DBNet/PSENet via MMOCR) as optional backends.
   - Integrate Arabic OCR (EasyOCR, PaddleOCR, MMOCR recognizers) with post-processing.
   - Automate **toponym normalization & transliteration**.

2. **Geometry & Alignment**
   - Export polar coordinates for boxes/labels using circle + tangent reference.
   - Implement multi-map **similarity metrics** (shape, label placement) and clustering.

3. **Address Misaligned Maps**
   - Pipeline assumes "perfect circle," but there are maps with two flaws: (1) non-circular (oblong), (2) hemispheres misaligned.
   - Develop small "map splitter" utility to split maps whose hemicircles are misaligned. 

4. **Visualization & Sharing**
   - Streamlit dashboard for per-map review and cross-map comparison.
   - Optional Google Sheets/Drive sync for metadata status tracking.

---

## Development

- Code lives under `src/`. Utilities are in `src/utils/`. Old smoke tests are in `scripts/`.
- Prefer **pure functions** that accept arrays/paths and return data/arrays.
- Use **type hints** and **docstrings** consistently.

---

## License

MIT License.
