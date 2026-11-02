from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NoiseConfig:
    octaves: int = 6
    lacunarity: float = 2.1
    persistence: float = 0.52
    base_scale: float = 0.0085
    ridge_mix: float = 0.28
    warp_strength: float = 42.0
    warp_scale: float = 0.0045
    mountain_bias: float = 0.15


@dataclass
class HydrologyConfig:
    epsilon: float = 1.5e-4
    lake_min_depth: float = 0.045
    lake_min_cells: int = 70
    lake_max_fraction: float = 0.08
    lake_max_count: int = 10
    river_accumulation_quantile: float = 0.972
    min_river_length: int = 10


@dataclass
class CliffConfig:
    slope_threshold: float = 0.55
    slope_percentile: float = 0.97
    min_component_cells: int = 10


@dataclass
class ForestConfig:
    target_density: float = 0.22
    max_slope: float = 0.42
    elev_min: float = 0.12
    elev_max: float = 0.82
    moisture_weight: float = 0.55
    clustering_scale: float = 0.035
    min_spacing_cells: int = 2


@dataclass
class RenderConfig:
    pixel_scale: int = 3
    show_contours: bool = True
    contour_levels: int = 8
    output_path: str = "output/forest_map.png"
    debug_panels_path: str = "output/forest_debug.png"


@dataclass
class TerrainConfig:
    seed: int = 42
    width: int = 256
    height: int = 256
    cell_size_m: float = 8.0
    relief_amplitude_m: float = 95.0
    dem_pre_smooth_sigma: float = 0.85
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)
    cliffs: CliffConfig = field(default_factory=CliffConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(cls: type, data: dict[str, Any] | None):
    if not data:
        return cls()
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in fields})


def load_config(path: str | Path | None = None) -> TerrainConfig:
    if path is None:
        return TerrainConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TerrainConfig(
        seed=int(raw.get("seed", 42)),
        width=int(raw.get("width", 256)),
        height=int(raw.get("height", 256)),
        cell_size_m=float(raw.get("cell_size_m", 8.0)),
        relief_amplitude_m=float(raw.get("relief_amplitude_m", 95.0)),
        dem_pre_smooth_sigma=float(raw.get("dem_pre_smooth_sigma", 0.85)),
        noise=_merge(NoiseConfig, raw.get("noise")),
        hydrology=_merge(HydrologyConfig, raw.get("hydrology")),
        cliffs=_merge(CliffConfig, raw.get("cliffs")),
        forest=_merge(ForestConfig, raw.get("forest")),
        render=_merge(RenderConfig, raw.get("render")),
    )
