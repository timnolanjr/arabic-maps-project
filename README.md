# Arabic Maps Project

A digital humanities toolkit for large-scale analysis of circular world maps in medieval Arabic manuscripts.

## Motivation

Arabic circular maps, with Mecca at the center, have long been studied
for their geographical insights, but scholars rarely compare them at the level
of individual manuscript copyists.  Each surviving map is a distinct artifact
shaped by scribal choices—subtle shifts in framing, scale, or label placement
that reflect workshops, regional tastes, or evolving worldviews.

By automatically detecting each map’s circular frame and projecting toponym
labels into polar coordinates, we can transform these subtle variations into
data.  Quantitative comparison across hundreds of maps may reveal patterns
of circulation, temporal shifts in cultural
perspective, and copyist creativity that text-only analysis alone cannot
capture.  Our toolkit brings a new, data-driven lens to the rich tradition of
Arabic cartography.

![Al-Idrīsī map, original (left) and flipped modern orientation (right)](assets/images/al-idrisi-1001inventions.jpg)

*Left: Original Al-Idrīsī 12th century map with Mecca at the top and Europe in the lower right.  
Right: Flipped view to match Western-style north-up orientation.*  [\[1\]](https://www.1001inventions.com/maps/)

![Al-Qazwīnī Map Example](assets/images/al-Qazwīnī_Arabic_MSS_575.jpg)

*Map extracted from a sixteenth-century Ottoman manuscript of Persian cosmographer and geographer al-Qazwīnī’s “The Marvels of Creation.”*

## Current Status

1. **Metadata Initialization**  
   Extracts file metadata (format, size, dimensions, colorspace) into `params.json`.
2. **Circle Detection**  
   Interactive (`scripts/test_circle.py`) & batch Hough‐circle detection (`src/circle.py`).
3. **Edge Detection**  
   Interactive (`scripts/test_edges.py`) & batch Hough‐line detection (`src/edges.py`).
4. **Tangent Computation**  
   Calculates the point on the detected circle tangent to the detected edge.

**Next Step**: Text Detection redult filtering - MSER fit is very good right now but produces many false positives. Researching filtering methods.


## Pipeline Overview

```
data/raw_maps/* (.jpg,.tif,.png)
    │
    ├─ Init metadata → data/raw_maps/image/params.json {"filetype", "filesize", "image_width", "image_height", "colorspace"}
    │
    ├─ Circle Detection → circle_final.jpg + params.json {"center_x", "center_y", "radius"}
    │
    ├─ Edge Detection   → edge_final.jpg   + params.json {"rho", "theta"}
    │
    ├─ Tangent Computation → params.json {"tangent_x", "tangent_y"}
    │
    └─ Overlay Generation → final_overlay.jpg
```

## Roadmap

- **Text Detection**
  MSER fit is very good right now but produces many false positives. Researching filtering methods.
- **OCR Integration**  
  Implement wrappers for EasyOCR, QARI, and Strabo pipelines to extract Arabic toponyms.
- **Toponym Extraction & Cleaning**  
  Normalize OCR output, deduplicate place names, and compute label centroids.
- **Map Comparison & Clustering**  
  Compute similarity metrics, cluster maps by style and content, and visualize variation.
- **Metadata Sync**  
  (Optional) Enhance Google Sheets/Drive integration with progress tracking and automated logging.

## Directory Structure

```
arabic-maps-project/
├── data/
│   ├── raw_maps/              # original TIFF/JPG/PNG scans
│   └── processed_maps/        # per-image output folders
├── scripts/
│   ├── init_params.py         # metadata init CLI
│   ├── test_circle.py         # circle detection CLI
│   ├── test_edges.py          # edge detection CLI
│   └── compute_tangent.py     # tangent computation CLI
├── src/
│   ├── circle.py              # Hough-circle logic
│   ├── edges.py               # Hough-line logic
│   └── utils/
│       ├── io.py              # JSON & directory helpers
│       ├── image.py           # blur & display helpers
│       ├── interactive.py     # click-ROI helpers
│       ├── metadata.py        # metadata init (new)
│       └── tangent.py         # tangent logic (new)
├── pipeline.py                # unified processing script (new)
├── requirements.txt
└── README.md
```

## License

MIT License.

