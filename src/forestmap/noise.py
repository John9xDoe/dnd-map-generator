"""Deterministic value-noise + fBm / ridged noise (no external noise package)."""

from __future__ import annotations

import numpy as np


def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer lattice hash → float in [0, 1)."""
    n = (ix.astype(np.int64) * 374761393) ^ (iy.astype(np.int64) * 668265263) ^ int(seed)
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0x7FFFFFFF).astype(np.float64) / float(0x7FFFFFFF)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    xf = _fade(x - x0)
    yf = _fade(y - y0)

    n00 = _hash2(x0, y0, seed)
    n10 = _hash2(x0 + 1, y0, seed)
    n01 = _hash2(x0, y0 + 1, seed)
    n11 = _hash2(x0 + 1, y0 + 1, seed)

    nx0 = n00 * (1.0 - xf) + n10 * xf
    nx1 = n01 * (1.0 - xf) + n11 * xf
    return nx0 * (1.0 - yf) + nx1 * yf


def fbm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    octaves: int,
    lacunarity: float,
    persistence: float,
) -> np.ndarray:
    total = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for i in range(octaves):
        total += amp * value_noise(x * freq, y * freq, seed + i * 1013)
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / max(norm, 1e-12)


def ridged_fbm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    octaves: int,
    lacunarity: float,
    persistence: float,
) -> np.ndarray:
    """1 - |noise| → sharp ridges useful for cliffs / divides."""
    total = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for i in range(octaves):
        n = value_noise(x * freq, y * freq, seed + 9000 + i * 733)
        total += amp * (1.0 - np.abs(2.0 * n - 1.0))
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / max(norm, 1e-12)


def coordinate_grids(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:height, 0:width]
    return xs.astype(np.float64), ys.astype(np.float64)
