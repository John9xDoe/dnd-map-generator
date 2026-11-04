"""Relief construction: warped fBm + ridged mountains, then local metrics."""

from __future__ import annotations

import numpy as np

from .config import NoiseConfig
from .noise import coordinate_grids, fbm, ridged_fbm


def generate_heightmap(
    width: int,
    height: int,
    seed: int,
    cfg: NoiseConfig,
) -> np.ndarray:
    """
    Build a normalized DEM in [0, 1].

    Physics-inspired construction:
    - low-frequency base = continental / hillslope envelope
    - domain warp breaks axis-aligned regularity (natural landforms)
    - ridged noise adds watershed divides / escarpment seeds
    - mild power curve biases more lowland (forest-friendly) than alpine
    """
    xs, ys = coordinate_grids(width, height)

    wx = fbm(
        xs * cfg.warp_scale,
        ys * cfg.warp_scale,
        seed=seed + 11,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
    )
    wy = fbm(
        xs * cfg.warp_scale + 37.0,
        ys * cfg.warp_scale - 19.0,
        seed=seed + 29,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
    )

    qx = xs * cfg.base_scale + cfg.warp_strength * (wx - 0.5)
    qy = ys * cfg.base_scale + cfg.warp_strength * (wy - 0.5)

    base = fbm(
        qx,
        qy,
        seed=seed,
        octaves=cfg.octaves,
        lacunarity=cfg.lacunarity,
        persistence=cfg.persistence,
    )
    ridges = ridged_fbm(
        qx * 1.35,
        qy * 1.35,
        seed=seed + 77,
        octaves=max(3, cfg.octaves - 1),
        lacunarity=cfg.lacunarity,
        persistence=cfg.persistence,
    )

    dem = (1.0 - cfg.ridge_mix) * base + cfg.ridge_mix * ridges
    # Soft mountain bias: lift already-high cells a bit (orographic feel).
    dem = dem + cfg.mountain_bias * (ridges ** 2)
    dem = (dem - dem.min()) / max(float(dem.max() - dem.min()), 1e-12)
    # More lowland area for temperate forest maps.
    dem = np.power(dem, 1.15)
    return dem.astype(np.float64)


def compute_slope(
    dem: np.ndarray,
    cell_size_m: float,
    relief_amplitude_m: float = 1.0,
) -> np.ndarray:
    """Physical slope magnitude |∇z| (rise/run) using DEM scaled to meters."""
    dem_m = dem.astype(np.float64) * float(relief_amplitude_m)
    gy, gx = np.gradient(dem_m, cell_size_m, cell_size_m)
    return np.sqrt(gx * gx + gy * gy)


def smooth_dem(dem: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian pre-smooth removes micro-pits; keeps macro basins/ridges."""
    if sigma is None or sigma <= 0:
        return dem
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        # Separable binomial-ish fallback without SciPy.
        k = np.array([1, 2, 1], dtype=np.float64)
        k /= k.sum()
        out = dem.astype(np.float64)
        passes = max(1, int(round(sigma * 2)))
        for _ in range(passes):
            pad = np.pad(out, ((0, 0), (1, 1)), mode="edge")
            out = k[0] * pad[:, :-2] + k[1] * pad[:, 1:-1] + k[2] * pad[:, 2:]
            pad = np.pad(out, ((1, 1), (0, 0)), mode="edge")
            out = k[0] * pad[:-2, :] + k[1] * pad[1:-1, :] + k[2] * pad[2:, :]
        out = (out - out.min()) / max(float(out.max() - out.min()), 1e-12)
        return out
    out = gaussian_filter(dem.astype(np.float64), sigma=sigma, mode="nearest")
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-12)
    return out


def compute_aspect(dem: np.ndarray, cell_size_m: float) -> np.ndarray:
    gy, gx = np.gradient(dem, cell_size_m, cell_size_m)
    return np.arctan2(-gy, gx)
