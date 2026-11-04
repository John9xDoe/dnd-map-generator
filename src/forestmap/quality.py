"""Generate-and-test gates and DnD-oriented fitness."""

from __future__ import annotations

import numpy as np

from .config import QualityConfig
from .features import FeatureResult, _binary_dilate
from .hydrology import HydrologyResult


def _largest_cc_fraction(mask: np.ndarray) -> float:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    total = int(mask.sum())
    if total == 0:
        return 0.0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            best = max(best, size)
    return best / total


def passable_mask(
    slope: np.ndarray,
    hydro: HydrologyResult,
    features: FeatureResult,
    max_slope: float,
) -> np.ndarray:
    return (
        (~hydro.lake_mask)
        & (~hydro.river_mask)
        & (~features.cliff_mask)
        & (slope <= max_slope)
    )


def count_camp_sites(
    dem: np.ndarray,
    slope: np.ndarray,
    hydro: HydrologyResult,
    features: FeatureResult,
    *,
    flat_slope: float = 0.25,
    near_water: int = 4,
) -> int:
    water = hydro.lake_mask | hydro.river_mask
    near = _binary_dilate(water, near_water) & ~features.water_block
    camps = (
        near
        & (slope <= flat_slope)
        & (~features.cliff_mask)
        & (~hydro.lake_mask)
        & (~hydro.river_mask)
    )
    h, w = dem.shape
    taken = np.zeros((h, w), dtype=bool)
    n = 0
    for r, c in zip(*np.where(camps)):
        if taken[r, c]:
            continue
        n += 1
        taken[max(0, r - 3) : min(h, r + 4), max(0, c - 3) : min(w, c + 4)] = True
    return n


def evaluate_map(
    dem: np.ndarray,
    slope: np.ndarray,
    hydro: HydrologyResult,
    features: FeatureResult,
    cfg: QualityConfig,
) -> dict:
    trees_on_water = 0
    if len(features.trees):
        h, w = dem.shape
        for r, c in features.trees:
            ir = int(np.clip(round(r), 0, h - 1))
            ic = int(np.clip(round(c), 0, w - 1))
            if features.water_block[ir, ic]:
                trees_on_water += 1

    lake_frac = float(hydro.lake_mask.mean())
    river_cells = int(hydro.river_mask.sum())
    passable = passable_mask(slope, hydro, features, cfg.max_slope_passable)
    pass_frac = float(passable.mean())
    cc_frac = _largest_cc_fraction(passable)
    camps = count_camp_sites(dem, slope, hydro, features)

    reasons: list[str] = []
    if trees_on_water > cfg.max_trees_on_water:
        reasons.append(f"trees_on_water={trees_on_water}")
    if river_cells < cfg.min_river_cells:
        reasons.append(f"too_few_rivers={river_cells}")
    if lake_frac > cfg.max_lake_fraction:
        reasons.append(f"too_much_lake={lake_frac:.3f}")
    if lake_frac < cfg.min_lake_fraction:
        reasons.append(f"too_little_lake={lake_frac:.3f}")
    if pass_frac < cfg.min_passable_fraction:
        reasons.append(f"low_passable={pass_frac:.3f}")
    if cc_frac < cfg.min_largest_cc_fraction:
        reasons.append(f"fragmented={cc_frac:.3f}")
    if (features.cliff_mask & hydro.lake_mask).any():
        reasons.append("cliffs_in_lakes")

    score = (
        0.35 * pass_frac
        + 0.25 * cc_frac
        + 0.15 * min(1.0, camps / 25.0)
        + 0.15 * min(1.0, river_cells / 2000.0)
        + 0.10 * max(0.0, 1.0 - abs(lake_frac - 0.05) / 0.05)
    )
    if reasons:
        score *= 0.35

    return {
        "score": float(score),
        "passable_fraction": pass_frac,
        "largest_cc_fraction": cc_frac,
        "camp_sites": camps,
        "river_cells": river_cells,
        "lake_fraction": lake_frac,
        "trees_on_water": trees_on_water,
        "reasons": reasons,
        "passed": len(reasons) == 0,
    }
