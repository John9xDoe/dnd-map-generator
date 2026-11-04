"""End-to-end forest map generation with agents + generate-and-test."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agents import apply_terrain_agents
from .config import TerrainConfig, load_config
from .erosion import evolve_landscape
from .features import FeatureResult, extract_features
from .heightmap import compute_slope, generate_heightmap, smooth_dem
from .hydrology import HydrologyResult, analyze_hydrology
from .quality import evaluate_map
from .render import save_outputs


@dataclass
class MapResult:
    dem: np.ndarray
    hydro: HydrologyResult
    features: FeatureResult
    outputs: dict[str, Path]
    stats: dict[str, Any]


def _apply_constraints(
    hydro: HydrologyResult,
    dem: np.ndarray,
    constraints: dict[str, np.ndarray] | None,
) -> tuple[HydrologyResult, np.ndarray]:
    """SketchaWorld-style hard constraints: painted masks override generation."""
    if not constraints:
        return hydro, dem
    dem = dem.copy()
    if "lake" in constraints:
        hydro.lake_mask = hydro.lake_mask | constraints["lake"]
        dem[constraints["lake"]] = np.minimum(dem[constraints["lake"]], 0.25)
    if "river" in constraints:
        hydro.river_mask = hydro.river_mask | constraints["river"]
    if "mountain" in constraints:
        dem[constraints["mountain"]] = np.maximum(
            dem[constraints["mountain"]], dem[constraints["mountain"]] + 0.08
        )
        dem = (dem - dem.min()) / max(float(dem.max() - dem.min()), 1e-12)
    return hydro, dem


def generate_map_once(
    cfg: TerrainConfig,
    *,
    save: bool = True,
    constraints: dict[str, np.ndarray] | None = None,
) -> MapResult:
    dem = generate_heightmap(cfg.width, cfg.height, cfg.seed, cfg.noise)
    dem = smooth_dem(dem, cfg.dem_pre_smooth_sigma)
    dem = evolve_landscape(dem, cfg.erosion)
    dem = smooth_dem(dem, max(0.35, cfg.dem_pre_smooth_sigma * 0.45))
    dem, agent_rivers = apply_terrain_agents(dem, cfg.seed, cfg.agents)

    slope = compute_slope(dem, cfg.cell_size_m, cfg.relief_amplitude_m)
    hydro = analyze_hydrology(dem, cfg.hydrology)
    # Merge agent-carved channels into hydrology mask.
    hydro.river_mask = hydro.river_mask | agent_rivers
    hydro, dem = _apply_constraints(hydro, dem, constraints)
    if constraints and ("mountain" in constraints or "lake" in constraints):
        slope = compute_slope(dem, cfg.cell_size_m, cfg.relief_amplitude_m)

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
    fitness = evaluate_map(dem, slope, hydro, features, cfg.quality)

    outputs: dict[str, Path] = {}
    if save:
        outputs = save_outputs(dem, hydro, features, cfg.render, cfg.seed)

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
        "trees_on_water": fitness["trees_on_water"],
        "max_accumulation": float(hydro.accumulation.max()),
        "mean_slope": float(slope.mean()),
        "fitness": fitness,
        "accepted": bool(fitness["passed"]),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "cells": cfg.width * cfg.height,
    }
    return MapResult(dem=dem, hydro=hydro, features=features, outputs=outputs, stats=stats)


def generate_map(
    config: TerrainConfig | str | Path | None = None,
    *,
    constraints: dict[str, np.ndarray] | None = None,
) -> MapResult:
    """Generate-and-test loop: retry with seed+k until quality gates pass."""
    base = config if isinstance(config, TerrainConfig) else load_config(config)
    cfg = deepcopy(base)

    best: MapResult | None = None
    attempts = 0
    max_tries = cfg.quality.max_retries if cfg.quality.enabled else 1

    for k in range(max_tries):
        attempts = k + 1
        cfg.seed = int(base.seed) + k
        # Only save the accepted (or last) attempt.
        save = (k == max_tries - 1) or not cfg.quality.enabled
        result = generate_map_once(cfg, save=False, constraints=constraints)
        result.stats["attempts"] = attempts
        if best is None or result.stats["fitness"]["score"] > best.stats["fitness"]["score"]:
            best = result
        if not cfg.quality.enabled or result.stats["fitness"]["passed"]:
            best = result
            break

    assert best is not None
    # Persist best
    best.outputs = save_outputs(
        best.dem, best.hydro, best.features, base.render, int(best.stats["seed"])
    )
    best.stats["outputs"] = {k: str(v) for k, v in best.outputs.items()}
    best.stats["attempts"] = attempts
    best.stats["accepted"] = bool(best.stats["fitness"]["passed"])
    return best
