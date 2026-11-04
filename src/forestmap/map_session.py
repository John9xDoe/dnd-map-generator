"""In-memory map session: generation, layers, brush edits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .config import TerrainConfig
from .erosion import evolve_landscape
from .features import FeatureResult, extract_features
from .heightmap import compute_slope, generate_heightmap, smooth_dem
from .hydrology import HydrologyResult, analyze_hydrology
from .render import render_symbolic_map


LAYER_NAMES = (
    "Карта (знаки)",
    "DEM",
    "Уклон",
    "Глубина котловин",
    "Сток (log)",
    "Влажность TWI",
    "Пригодность леса",
    "Озёра",
    "Реки",
    "Тропы",
    "Обрывы",
)

BRUSH_RIVER = "river"
BRUSH_TRAIL = "trail"
BRUSH_LAKE = "lake"
BRUSH_MOUNTAIN = "mountain"
BRUSH_ERASE = "erase"


@dataclass
class MapSession:
    cfg: TerrainConfig
    dem: np.ndarray
    slope: np.ndarray
    hydro: HydrologyResult
    features: FeatureResult
    trail_mask: np.ndarray
    symbolic: Image.Image | None = None
    stats: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dem.shape


def _stats(dem, slope, hydro, features, trails) -> dict:
    return {
        "lake_cells": int(hydro.lake_mask.sum()),
        "river_cells": int(hydro.river_mask.sum()),
        "cliff_cells": int(features.cliff_mask.sum()),
        "trail_cells": int(trails.sum()),
        "tree_count": int(len(features.trees)),
        "mean_slope": float(slope.mean()),
        "max_accumulation": float(hydro.accumulation.max()),
    }


def build_session(cfg: TerrainConfig, *, dem: np.ndarray | None = None) -> MapSession:
    if dem is None:
        dem = generate_heightmap(cfg.width, cfg.height, cfg.seed, cfg.noise)
        dem = smooth_dem(dem, cfg.dem_pre_smooth_sigma)
        dem = evolve_landscape(dem, cfg.erosion)
        dem = smooth_dem(dem, max(0.35, cfg.dem_pre_smooth_sigma * 0.45))
    else:
        dem = dem.astype(np.float64)
        dem = (dem - dem.min()) / max(float(dem.max() - dem.min()), 1e-12)

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
    trails = np.zeros(dem.shape, dtype=bool)
    symbolic = render_symbolic_map(dem, hydro, features, cfg.render, cfg.seed)
    return MapSession(
        cfg=cfg,
        dem=dem,
        slope=slope,
        hydro=hydro,
        features=features,
        trail_mask=trails,
        symbolic=symbolic,
        stats=_stats(dem, slope, hydro, features, trails),
    )


def rebuild_from_dem(session: MapSession, cfg: TerrainConfig) -> MapSession:
    """Recompute hydro/features from edited DEM, keep painted trails."""
    trails = session.trail_mask.copy()
    # Merge painted lakes/rivers into DEM sinks before hydro? Lakes painted as
    # masks are reapplied after hydro so user strokes survive.
    painted_lakes = session.hydro.lake_mask.copy()
    painted_rivers = session.hydro.river_mask.copy()

    new = build_session(cfg, dem=session.dem.copy())
    new.trail_mask = trails
    new.hydro.lake_mask |= painted_lakes
    new.hydro.river_mask |= painted_rivers
    new.symbolic = render_symbolic_map(
        new.dem, new.hydro, new.features, cfg.render, cfg.seed
    )
    new.stats = _stats(new.dem, new.slope, new.hydro, new.features, new.trail_mask)
    return new


def refresh_symbolic(session: MapSession) -> None:
    session.symbolic = render_symbolic_map(
        session.dem,
        session.hydro,
        session.features,
        session.cfg.render,
        session.cfg.seed,
    )
    session.stats = _stats(
        session.dem,
        session.slope,
        session.hydro,
        session.features,
        session.trail_mask,
    )


def _brush_disk(h: int, w: int, r: float, c: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - r) ** 2 + (xx - c) ** 2 <= radius**2


def apply_brush(
    session: MapSession,
    *,
    row: float,
    col: float,
    radius: float,
    kind: str,
    strength: float = 0.08,
) -> None:
    h, w = session.dem.shape
    mask = _brush_disk(h, w, row, col, radius)
    if not np.any(mask):
        return

    if kind == BRUSH_RIVER:
        session.hydro.river_mask[mask] = True
        session.hydro.lake_mask[mask] = False
        session.trail_mask[mask] = False
    elif kind == BRUSH_TRAIL:
        session.trail_mask[mask] = True
    elif kind == BRUSH_LAKE:
        session.hydro.lake_mask[mask] = True
        session.hydro.river_mask[mask] = False
        session.dem[mask] = np.minimum(session.dem[mask], float(session.dem[mask].mean()) - 0.01)
        session.dem[mask] = np.clip(session.dem[mask], 0.0, 1.0)
    elif kind == BRUSH_MOUNTAIN:
        session.dem[mask] = np.clip(session.dem[mask] + strength, 0.0, 1.0)
        session.slope = compute_slope(
            session.dem, session.cfg.cell_size_m, session.cfg.relief_amplitude_m
        )
    elif kind == BRUSH_ERASE:
        session.hydro.river_mask[mask] = False
        session.hydro.lake_mask[mask] = False
        session.trail_mask[mask] = False


def layer_to_rgb(session: MapSession, layer: str) -> np.ndarray:
    """Return HxWx3 uint8 preview for a named layer."""
    h, w = session.dem.shape

    def colorize(data: np.ndarray, cmap: str) -> np.ndarray:
        d = data.astype(np.float64)
        d = (d - d.min()) / max(float(d.max() - d.min()), 1e-12)
        # Lightweight colormaps without matplotlib dependency at paint-time.
        if cmap == "terrain":
            c0 = np.array([92, 120, 72])
            c1 = np.array([168, 150, 128])
        elif cmap == "magma":
            c0 = np.array([20, 5, 30])
            c1 = np.array([250, 200, 80])
        elif cmap == "blues":
            c0 = np.array([245, 248, 252])
            c1 = np.array([30, 90, 170])
        elif cmap == "viridis":
            c0 = np.array([40, 10, 60])
            c1 = np.array([220, 230, 80])
        elif cmap == "ylgnbu":
            c0 = np.array([255, 255, 200])
            c1 = np.array([20, 80, 140])
        elif cmap == "greens":
            c0 = np.array([245, 250, 245])
            c1 = np.array([30, 110, 50])
        else:
            c0 = np.array([0, 0, 0])
            c1 = np.array([255, 255, 255])
        rgb = c0 * (1 - d[..., None]) + c1 * d[..., None]
        return np.clip(rgb, 0, 255).astype(np.uint8)

    if layer == "Карта (знаки)":
        if session.symbolic is None:
            refresh_symbolic(session)
        img = np.array(session.symbolic.convert("RGB"))
        # Overlay trails in brown.
        if session.trail_mask.any():
            scale = max(1, img.shape[0] // h)
            for r, c in zip(*np.where(session.trail_mask)):
                r0, c0 = r * scale, c * scale
                img[r0 : r0 + scale, c0 : c0 + scale] = (150, 105, 55)
        return img

    if layer == "DEM":
        return colorize(session.dem, "terrain")
    if layer == "Уклон":
        return colorize(session.slope, "magma")
    if layer == "Глубина котловин":
        return colorize(session.hydro.fill_depth, "blues")
    if layer == "Сток (log)":
        return colorize(np.log1p(session.hydro.accumulation), "viridis")
    if layer == "Влажность TWI":
        return colorize(session.features.moisture, "ylgnbu")
    if layer == "Пригодность леса":
        return colorize(session.features.suitability, "greens")
    if layer == "Озёра":
        rgb = np.full((h, w, 3), 40, dtype=np.uint8)
        rgb[session.hydro.lake_mask] = (70, 130, 190)
        return rgb
    if layer == "Реки":
        rgb = np.full((h, w, 3), 40, dtype=np.uint8)
        rgb[session.hydro.river_mask] = (60, 140, 220)
        return rgb
    if layer == "Тропы":
        rgb = np.full((h, w, 3), 40, dtype=np.uint8)
        rgb[session.trail_mask] = (160, 110, 60)
        return rgb
    if layer == "Обрывы":
        rgb = np.full((h, w, 3), 40, dtype=np.uint8)
        rgb[session.features.cliff_mask] = (170, 120, 70)
        return rgb

    return colorize(session.dem, "terrain")
