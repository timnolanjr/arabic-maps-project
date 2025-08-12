from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

RGB = Tuple[int, int, int]
BGR = Tuple[int, int, int]

@dataclass(frozen=True)
class Color:
    name: str
    rgb: RGB  # 0-255

    @property
    def bgr(self) -> BGR:
        r, g, b = self.rgb
        return (b, g, r)

    @property
    def mpl(self) -> Tuple[float, float, float]:
        r, g, b = self.rgb
        return (r / 255.0, g / 255.0, b / 255.0)

@dataclass(frozen=True)
class Stroke:
    thickness: int = 2
    alpha: float = 0.9

@dataclass(frozen=True)
class Palette:
    # Geometry layers
    circle: Color = Color("circle", (0, 200, 0))                 # green
    rho_theta: Color = Color("rho_theta", (30, 144, 255))        # blue for reference edge (ρ, θ)
    center: Color = Color("center", (255, 215, 0))               # gold
    tangent_point: Color = Color("tangent_point", (220, 20, 60)) # red (South)

    # Text & boxes
    text_box: Color = Color("text_box", (255, 0, 255))           # magenta
    text_label: Color = Color("text_label", (255, 255, 255))     # white
    legend_bg: Color = Color("legend_bg", (0, 0, 0))             # black

    # Stroke presets
    thick: Stroke = Stroke(thickness=3, alpha=0.95)
    normal: Stroke = Stroke(thickness=2, alpha=0.9)
    thin: Stroke = Stroke(thickness=1, alpha=0.85)

DEFAULT_PALETTE = Palette()
