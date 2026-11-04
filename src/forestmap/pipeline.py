"""End-to-end forest map generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import TerrainConfig, load_config
from .erosion import evolve_landscape
from .features import FeatureResult, extract_features
from .heightmap import compute_slope, generate_heightmap, smooth_dem
from .hydrology import HydrologyResult, analyze_hydrology
from .render import save_outputs


@dataclass
class MapResult:
    dem: np.ndarray
    hydro: HydrologyResult
    features: FeatureResult
    outputs: dict[str, Path]
    stats: dict[str, Any]


def generate_map(config: TerrainConfig | str | Path | None = None) -> MapResult:
    cfg = config if isinstance(config, TerrainConfig) else load_config(config)

    dem = generate_heightmap(cfg.width, cfg.height, cfg.seed, cfg.noise)
    dem = smooth_dem(dem, cfg.dem_pre_smooth_sigma)
    # Landscape evolution: talus healing + stream-power valleys.
    dem = evolve_landscape(dem, cfg.erosion)
    dem = smooth_dem(dem, max(0.35, cfg.dem_pre_smooth_sigma * 0.45))

    slope = compute_slope(dem, cfg.cell_size_m, cfg.relief_amplitude_m)
    hydro = analyze_hydrology(dem, cfg.hydrology)
    features = extract_features(
        dem=dem,
        slope=slope,
        accumulation=hydro.accumulation,
        lake_mask=hydro.lake_mask,
        river_mask=hydro.river_mask,
        cell_size_m=cfg.cell_size_m,
        seed=cfg.seed,
        cliffs_cfg=cfg.cliffs,
        forest_cfg=cfg.forest,
    )
    outputs = save_outputs(dem, hydro, features, cfg.render, cfg.seed)

    n_cells = cfg.width * cfg.height
    stats: dict[str, Any] = {
        "seed": cfg.seed,
        "size": [cfg.width, cfg.height],
        "cell_size_m": cfg.cell_size_m,
        "elev_range": [float(dem.min()), float(dem.max())],
        "lake_cells": int(hydro.lake_mask.sum()),
        "lake_fraction": float(hydro.lake_mask.mean()),
        "river_cells": int(hydro.river_mask.sum()),
        "cliff_cells": int(features.cliff_mask.sum()),
        "tree_count": int(len(features.trees)),
        "trees_on_water": int(
            sum(
                1
                for r, c in features.trees
                if features.water_block[
                    int(np.clip(round(r), 0, dem.shape[0] - 1)),
                    int(np.clip(round(c), 0, dem.shape[1] - 1)),
                ]
            )
        )
        if len(features.trees)
        else 0,
        "max_accumulation": float(hydro.accumulation.max()),
        "mean_slope": float(slope.mean()),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "cells": n_cells,
    }
    return MapResult(
        dem=dem,
        hydro=hydro,
        features=features,
        outputs=outputs,
        stats=stats,
    )
