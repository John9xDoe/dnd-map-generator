from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class ErosionConfig:
    enabled: bool = True
    thermal_iterations: int = 10
    talus_tangent: float = 0.08
    thermal_rate: float = 0.25
    fluvial_iterations: int = 24
    stream_power_k: float = 0.010
    area_exp: float = 0.5
    slope_exp: float = 1.0
    max_incision_per_step: float = 0.015
    uplift: float = 0.0006


@dataclass
class NoiseConfig:
    octaves: int = 6
    lacunarity: float = 2.05
    persistence: float = 0.48
    base_scale: float = 0.0072
    ridge_mix: float = 0.26
    warp_strength: float = 36.0
    warp_scale: float = 0.0038
    mountain_bias: float = 0.14


@dataclass
class HydrologyConfig:
    epsilon: float = 1.2e-4
    lake_min_depth: float = 0.05
    lake_min_cells: int = 80
    lake_max_fraction: float = 0.07
    lake_max_count: int = 8
    river_accumulation_quantile: float = 0.982
    min_river_length: int = 14
    river_bank_cells: int = 1


@dataclass
class CliffConfig:
    slope_threshold: float = 0.28
    slope_percentile: float = 0.97
    min_component_cells: int = 12
    morph_close_iters: int = 3
    band_width_cells: int = 3


@dataclass
class ForestConfig:
    target_density: float = 0.15
    max_slope: float = 0.55
    elev_min: float = 0.08
    elev_max: float = 0.86
    moisture_weight: float = 0.62
    clustering_scale: float = 0.028
    min_spacing_cells: int = 2
    water_buffer_cells: int = 3


@dataclass
class AgentConfig:
    enabled: bool = True
    n_mountain_agents: int = 6
    mountain_steps: int = 90
    mountain_raise: float = 0.045
    mountain_width: int = 2
    n_river_agents: int = 5
    river_min_length: int = 28
    river_max_attempts: int = 12
    river_start_width: float = 0.8
    river_widen_every: int = 10
    river_carve: float = 0.035


@dataclass
class QualityConfig:
    enabled: bool = True
    max_retries: int = 6
    max_trees_on_water: int = 0
    min_river_cells: int = 80
    max_lake_fraction: float = 0.12
    min_lake_fraction: float = 0.01
    min_passable_fraction: float = 0.35
    min_largest_cc_fraction: float = 0.55
    max_slope_passable: float = 0.65


@dataclass
class RenderConfig:
    pixel_scale: int = 3
    show_contours: bool = True
    contour_levels: int = 7
    use_lsystem_trees: bool = True
    output_path: str = "output/forest_map.png"
    debug_panels_path: str = "output/forest_debug.png"
    expressive_range_path: str = "output/expressive_range.png"


@dataclass
class TerrainConfig:
    seed: int = 42
    width: int = 256
    height: int = 256
    cell_size_m: float = 8.0
    relief_amplitude_m: float = 58.0
    dem_pre_smooth_sigma: float = 0.85
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)
    cliffs: CliffConfig = field(default_factory=CliffConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
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
        relief_amplitude_m=float(raw.get("relief_amplitude_m", 58.0)),
        dem_pre_smooth_sigma=float(raw.get("dem_pre_smooth_sigma", 0.85)),
        noise=_merge(NoiseConfig, raw.get("noise")),
        erosion=_merge(ErosionConfig, raw.get("erosion")),
        agents=_merge(AgentConfig, raw.get("agents")),
        hydrology=_merge(HydrologyConfig, raw.get("hydrology")),
        cliffs=_merge(CliffConfig, raw.get("cliffs")),
        forest=_merge(ForestConfig, raw.get("forest")),
        quality=_merge(QualityConfig, raw.get("quality")),
        render=_merge(RenderConfig, raw.get("render")),
    )
