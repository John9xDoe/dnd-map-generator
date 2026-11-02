"""Derived land-cover features: cliffs and forest stand placement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CliffConfig, ForestConfig
from .hydrology import D8
from .noise import fbm


@dataclass
class FeatureResult:
    slope: np.ndarray
    cliff_mask: np.ndarray
    moisture: np.ndarray
    suitability: np.ndarray
    trees: np.ndarray  # (N, 2) row/col float positions


def _label_and_filter(mask: np.ndarray, min_cells: int) -> np.ndarray:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    lid = 0
    keep_mask = np.zeros_like(mask)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c]:
                continue
            lid += 1
            stack = [(r, c)]
            labels[r, c] = lid
            cells = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in D8:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = lid
                        stack.append((nr, nc))
                        cells.append((nr, nc))
            if len(cells) >= min_cells:
                for rr, cc in cells:
                    keep_mask[rr, cc] = True
    return keep_mask


def extract_cliffs(slope: np.ndarray, cfg: CliffConfig) -> np.ndarray:
    # Absolute steepness OR local extreme — works across relief amplitudes.
    pct = float(np.quantile(slope, cfg.slope_percentile))
    thr = max(float(cfg.slope_threshold), pct)
    raw = slope >= thr
    return _label_and_filter(raw, cfg.min_component_cells)


def topographic_wetness(
    accumulation: np.ndarray,
    slope: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    """
    Beven & Kirkby TWI proxy: ln(a / tan(β)).
    High values → hollows / toeslopes that stay moist → denser forest.
    """
    # Specific catchment area proxy (cells * cell area / cell width).
    a = accumulation * cell_size_m
    beta = np.maximum(slope, 1e-4)
    twi = np.log(a / beta)
    twi = (twi - twi.min()) / max(float(twi.max() - twi.min()), 1e-12)
    return twi


def place_trees(
    dem: np.ndarray,
    slope: np.ndarray,
    moisture: np.ndarray,
    blocked: np.ndarray,
    seed: int,
    cfg: ForestConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Suitability-driven thinned point process.

    Real forest constraints encoded:
    - no stands in water / cliffs
    - avoid steep slopes (shallow soils, windthrow)
    - elevation band (temperate forest belt)
    - moisture (TWI) attracts denser canopy near streams/toeslopes
    - clustered noise → groves instead of uniform sprinkle
    - min spacing → crown competition / map readability
    """
    h, w = dem.shape
    elev_ok = (dem >= cfg.elev_min) & (dem <= cfg.elev_max)
    slope_factor = np.clip(1.0 - slope / max(cfg.max_slope, 1e-6), 0.0, 1.0)

    ys, xs = np.mgrid[0:h, 0:w]
    cluster = fbm(
        xs.astype(np.float64) * cfg.clustering_scale,
        ys.astype(np.float64) * cfg.clustering_scale,
        seed=seed + 404,
        octaves=4,
        lacunarity=2.2,
        persistence=0.55,
    )

    suitability = (
        elev_ok.astype(np.float64)
        * slope_factor
        * ((1.0 - cfg.moisture_weight) + cfg.moisture_weight * moisture)
        * (0.35 + 0.65 * cluster)
    )
    suitability = np.where(blocked | (slope > cfg.max_slope), 0.0, suitability)

    rng = np.random.default_rng(seed + 7)
    # Soft threshold so only relatively good cells compete.
    soft = np.quantile(suitability[suitability > 0], 0.35) if np.any(suitability > 0) else 1.0
    candidates = np.argwhere(suitability >= soft)
    if candidates.size == 0:
        return suitability, np.zeros((0, 2), dtype=np.float64)

    scores = suitability[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-scores)
    candidates = candidates[order]

    target = int(cfg.target_density * h * w / max(cfg.min_spacing_cells**2, 1))
    target = max(20, min(target, len(candidates)))

    occupied = np.zeros((h, w), dtype=bool)
    spacing = max(1, int(cfg.min_spacing_cells))
    chosen: list[tuple[float, float]] = []

    for r, c in candidates:
        r, c = int(r), int(c)
        if occupied[r, c]:
            continue
        # Probabilistic accept scaled by suitability.
        p = float(suitability[r, c])
        if rng.random() > min(1.0, 0.25 + 0.75 * p):
            continue
        jitter_r = float(r) + float(rng.uniform(-0.35, 0.35))
        jitter_c = float(c) + float(rng.uniform(-0.35, 0.35))
        chosen.append((jitter_r, jitter_c))
        r0 = max(0, r - spacing)
        r1 = min(h, r + spacing + 1)
        c0 = max(0, c - spacing)
        c1 = min(w, c + spacing + 1)
        occupied[r0:r1, c0:c1] = True
        if len(chosen) >= target:
            break

    trees = np.asarray(chosen, dtype=np.float64) if chosen else np.zeros((0, 2))
    return suitability, trees


def extract_features(
    dem: np.ndarray,
    slope: np.ndarray,
    accumulation: np.ndarray,
    lake_mask: np.ndarray,
    river_mask: np.ndarray,
    cell_size_m: float,
    seed: int,
    cliffs_cfg: CliffConfig,
    forest_cfg: ForestConfig,
) -> FeatureResult:
    cliff_mask = extract_cliffs(slope, cliffs_cfg)
    moisture = topographic_wetness(accumulation, slope, cell_size_m)
    # Rivers locally boost moisture for riparian stands.
    moisture = np.clip(moisture + 0.15 * river_mask.astype(np.float64), 0.0, 1.0)
    blocked = lake_mask | river_mask | cliff_mask
    suitability, trees = place_trees(dem, slope, moisture, blocked, seed, forest_cfg)
    return FeatureResult(
        slope=slope,
        cliff_mask=cliff_mask,
        moisture=moisture,
        suitability=suitability,
        trees=trees,
    )
