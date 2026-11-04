"""In-memory map session: generation, layers, brush edits, hard constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .config import TerrainConfig
from .features import FeatureResult
from .heightmap import compute_slope
from .hydrology import HydrologyResult
from .pipeline import generate_map_once
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
    "Лагеря (DnD)",
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
    constraints: dict[str, np.ndarray] = field(default_factory=dict)
    symbolic: Image.Image | None = None
    stats: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dem.shape


def _stats(dem, slope, hydro, features, trails, extra: dict | None = None) -> dict:
    d = {
        "lake_cells": int(hydro.lake_mask.sum()),
        "river_cells": int(hydro.river_mask.sum()),
        "cliff_cells": int(features.cliff_mask.sum()),
        "trail_cells": int(trails.sum()),
        "tree_count": int(len(features.trees)),
        "mean_slope": float(slope.mean()),
        "max_accumulation": float(hydro.accumulation.max()),
    }
    if extra:
        d.update(extra)
    return d


def build_session(
    cfg: TerrainConfig,
    *,
    constraints: dict[str, np.ndarray] | None = None,
) -> MapSession:
    # Disable multi-retry in interactive GUI for snappiness; still evaluate fitness.
    cfg = cfg
    q_enabled = cfg.quality.enabled
    cfg.quality.enabled = False
    result = generate_map_once(cfg, save=False, constraints=constraints)
    cfg.quality.enabled = q_enabled

    trails = np.zeros(result.dem.shape, dtype=bool)
    if constraints and "trail" in constraints:
        trails = constraints["trail"].copy()

    return MapSession(
        cfg=cfg,
        dem=result.dem,
        slope=compute_slope(result.dem, cfg.cell_size_m, cfg.relief_amplitude_m),
        hydro=result.hydro,
        features=result.features,
        trail_mask=trails,
        constraints={k: v.copy() for k, v in (constraints or {}).items()},
        symbolic=render_symbolic_map(
            result.dem, result.hydro, result.features, cfg.render, cfg.seed
        ),
        stats=_stats(
            result.dem,
            result.features.slope,
            result.hydro,
            result.features,
            trails,
            {
                "fitness": result.stats.get("fitness", {}),
                "accepted": result.stats.get("accepted", True),
            },
        ),
    )


def rebuild_from_dem(session: MapSession, cfg: TerrainConfig) -> MapSession:
    """Recompute with SketchaWorld hard constraints preserved."""
    constraints = {k: v.copy() for k, v in session.constraints.items()}
    # Current painted hydro also becomes constraints.
    constraints["lake"] = session.hydro.lake_mask.copy()
    constraints["river"] = session.hydro.river_mask.copy()
    if session.trail_mask.any():
        constraints["trail"] = session.trail_mask.copy()
    new = build_session(cfg, constraints=constraints)
    new.trail_mask = constraints.get("trail", new.trail_mask)
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
        session.stats,
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
    """Paint feature AND lock it as a hard constraint (SketchaWorld-style)."""
    h, w = session.dem.shape
    mask = _brush_disk(h, w, row, col, radius)
    if not np.any(mask):
        return

    def lock(name: str) -> None:
        if name not in session.constraints:
            session.constraints[name] = np.zeros((h, w), dtype=bool)
        session.constraints[name][mask] = True

    if kind == BRUSH_RIVER:
        session.hydro.river_mask[mask] = True
        session.hydro.lake_mask[mask] = False
        session.trail_mask[mask] = False
        lock("river")
    elif kind == BRUSH_TRAIL:
        session.trail_mask[mask] = True
        lock("trail")
    elif kind == BRUSH_LAKE:
        session.hydro.lake_mask[mask] = True
        session.hydro.river_mask[mask] = False
        session.dem[mask] = np.minimum(
            session.dem[mask], float(session.dem[mask].mean()) - 0.01
        )
        session.dem[mask] = np.clip(session.dem[mask], 0.0, 1.0)
        lock("lake")
    elif kind == BRUSH_MOUNTAIN:
        session.dem[mask] = np.clip(session.dem[mask] + strength, 0.0, 1.0)
        session.slope = compute_slope(
            session.dem, session.cfg.cell_size_m, session.cfg.relief_amplitude_m
        )
        lock("mountain")
    elif kind == BRUSH_ERASE:
        session.hydro.river_mask[mask] = False
        session.hydro.lake_mask[mask] = False
        session.trail_mask[mask] = False
        for name in ("river", "lake", "trail", "mountain"):
            if name in session.constraints:
                session.constraints[name][mask] = False


def layer_to_rgb(session: MapSession, layer: str) -> np.ndarray:
    from .quality import count_camp_sites, passable_mask

    h, w = session.dem.shape

    def colorize(data: np.ndarray, cmap: str) -> np.ndarray:
        d = data.astype(np.float64)
        d = (d - d.min()) / max(float(d.max() - d.min()), 1e-12)
        palettes = {
            "terrain": ([92, 120, 72], [168, 150, 128]),
            "magma": ([20, 5, 30], [250, 200, 80]),
            "blues": ([245, 248, 252], [30, 90, 170]),
            "viridis": ([40, 10, 60], [220, 230, 80]),
            "ylgnbu": ([255, 255, 200], [20, 80, 140]),
            "greens": ([245, 250, 245], [30, 110, 50]),
        }
        c0, c1 = (np.array(a, dtype=np.float64) for a in palettes.get(cmap, ([0, 0, 0], [255, 255, 255])))
        rgb = c0 * (1 - d[..., None]) + c1 * d[..., None]
        return np.clip(rgb, 0, 255).astype(np.uint8)

    if layer == "Карта (знаки)":
        if session.symbolic is None:
            refresh_symbolic(session)
        img = np.array(session.symbolic.convert("RGB"))
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
    if layer == "Лагеря (DnD)":
        rgb = colorize(session.dem, "terrain")
        # Mark passable faintly and camps as yellow dots via camp heuristic cells.
        from .features import _binary_dilate

        water = session.hydro.lake_mask | session.hydro.river_mask
        near = _binary_dilate(water, 4) & ~session.features.water_block
        camps = (
            near
            & (session.slope <= 0.25)
            & (~session.features.cliff_mask)
            & (~session.hydro.lake_mask)
            & (~session.hydro.river_mask)
        )
        rgb[passable_mask(session.slope, session.hydro, session.features, 0.65)] = (
            rgb[passable_mask(session.slope, session.hydro, session.features, 0.65)] * 0.85
            + np.array([40, 80, 40]) * 0.15
        ).astype(np.uint8)
        rgb[camps] = (230, 200, 60)
        return rgb
    return colorize(session.dem, "terrain")
