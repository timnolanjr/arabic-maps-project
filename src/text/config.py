from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence

@dataclass
class MserParams:
    delta: int = 5
    min_area: int = 60
    max_area: int = 14400
    max_variation: float = 0.3
    min_diversity: float = 0.2
    max_evolution: int = 200
    area_threshold: float = 1.01
    min_margin: float = 0.003
    edge_blur_size: int = 3
    polarity: str = "both"  # "dark" | "bright" | "both"

@dataclass
class GeometryFilter:
    enabled: bool = True
    area_pct: Tuple[float, float] = (0.00005, 0.02)
    aspect_ratio: Tuple[float, float] = (0.15, 8.0)

@dataclass
class CircleFilter:
    enabled: bool = False
    min_cover: float = 0.5

@dataclass
class NmsConfig:
    enabled: bool = True
    iou: float = 0.3

@dataclass
class MultiScaleConfig:
    enabled: bool = False
    scales: Sequence[float] = (1.0, 0.75, 0.5)

@dataclass
class MorphologyConfig:
    enabled: bool = False
    close_ksize: int = 3
    open_ksize: int = 0
    iterations: int = 1

@dataclass
class VisualConfig:
    include_removed: bool = True
    scale: Optional[float] = None
    show_labels: bool = False
    box_thickness: Optional[int] = 3

@dataclass
class TextDetectConfig:
    method: str = "mser"
    mser: MserParams = field(default_factory=MserParams)
    geometry: GeometryFilter = field(default_factory=GeometryFilter)
    circle: CircleFilter = field(default_factory=CircleFilter)
    nms: NmsConfig = field(default_factory=NmsConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    multiscale: MultiScaleConfig = field(default_factory=MultiScaleConfig)
    morph: MorphologyConfig = field(default_factory=MorphologyConfig)