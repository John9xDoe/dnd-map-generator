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
    water_block: np.ndarray  # dilated water (lakes+rivers+banks)


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= padded[1 + dr : 1 + dr + out.shape[0], 1 + dc : 1 + dc + out.shape[1]]
        out = grown
    return out


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        kept = np.ones_like(out, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                kept &= padded[1 + dr : 1 + dr + out.shape[0], 1 + dc : 1 + dc + out.shape[1]]
        out = kept
    return out


def _morph_close(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Close = dilate then erode — fills 1-pixel holes in cliff bands."""
    return _binary_erode(_binary_dilate(mask, iterations), iterations)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed False pockets inside True components (background flood)."""
    h, w = mask.shape
    fill = mask.astype(bool).copy()
    visited = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if not fill[r, c] and not visited[r, c]:
                stack.append((r, c))
                visited[r, c] = True
    for c in range(w):
        for r in (0, h - 1):
            if not fill[r, c] and not visited[r, c]:
                stack.append((r, c))
                visited[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in D8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and not fill[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))
    # Unvisited False cells are holes → set True.
    holes = (~fill) & (~visited)
    fill[holes] = True
    return fill


def _label_and_filter(mask: np.ndarray, min_cells: int) -> np.ndarray:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    lid = 0
    keep_mask = np.zeros_like(mask, dtype=bool)
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
    """
    Continuous escarpment bands: threshold → morph close → fill holes →
    thicken → drop tiny speckles.
    """
    pct = float(np.quantile(slope, cfg.slope_percentile))
    thr = max(float(cfg.slope_threshold), pct)
    raw = slope >= thr
    try:
        from scipy import ndimage as ndi

        structure = np.ones((3, 3), dtype=bool)
        closed = ndi.binary_closing(
            raw, structure=structure, iterations=max(1, cfg.morph_close_iters)
        )
        # Remove pepper noise, then close again so the scarp is a solid ribbon.
        closed = ndi.binary_opening(closed, structure=structure, iterations=1)
        closed = ndi.binary_closing(
            closed, structure=structure, iterations=max(1, cfg.morph_close_iters)
        )
        filled = ndi.binary_fill_holes(closed)
        band = ndi.binary_dilation(
            filled, structure=structure, iterations=max(0, cfg.band_width_cells - 1)
        )
    except Exception:
        closed = _morph_close(raw, max(1, cfg.morph_close_iters))
        filled = _fill_holes(closed)
        band = _binary_dilate(filled, max(0, cfg.band_width_cells - 1))
    return _label_and_filter(band.astype(bool), cfg.min_component_cells)


def topographic_wetness(
    accumulation: np.ndarray,
    slope: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
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
    soft = (
        np.quantile(suitability[suitability > 0], 0.35)
        if np.any(suitability > 0)
        else 1.0
    )
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
        if occupied[r, c] or blocked[r, c]:
            continue
        p = float(suitability[r, c])
        if rng.random() > min(1.0, 0.25 + 0.75 * p):
            continue
        # Small jitter, but never onto blocked water/cliff cells.
        for _ in range(6):
            jitter_r = float(r) + float(rng.uniform(-0.25, 0.25))
            jitter_c = float(c) + float(rng.uniform(-0.25, 0.25))
            ir = int(np.clip(round(jitter_r), 0, h - 1))
            ic = int(np.clip(round(jitter_c), 0, w - 1))
            if not blocked[ir, ic]:
                chosen.append((jitter_r, jitter_c))
                break
        else:
            continue
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
    # Escarpments cannot sit in standing water / channels.
    cliff_mask = cliff_mask & ~(lake_mask | river_mask)
    moisture = topographic_wetness(accumulation, slope, cell_size_m)
    moisture = np.clip(moisture + 0.15 * river_mask.astype(np.float64), 0.0, 1.0)

    water = lake_mask | river_mask
    water_block = _binary_dilate(water, max(1, forest_cfg.water_buffer_cells))
    blocked = water_block | cliff_mask
    suitability, trees = place_trees(dem, slope, moisture, blocked, seed, forest_cfg)

    # Final hard filter: drop any tree whose nearest cell is water-blocked.
    if len(trees):
        keep = []
        h, w = dem.shape
        for r, c in trees:
            ir = int(np.clip(round(r), 0, h - 1))
            ic = int(np.clip(round(c), 0, w - 1))
            if not water_block[ir, ic] and not cliff_mask[ir, ic]:
                keep.append((r, c))
        trees = np.asarray(keep, dtype=np.float64) if keep else np.zeros((0, 2))

    return FeatureResult(
        slope=slope,
        cliff_mask=cliff_mask,
        moisture=moisture,
        suitability=suitability,
        trees=trees,
        water_block=water_block,
    )
