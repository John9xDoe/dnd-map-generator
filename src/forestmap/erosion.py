"""
Landscape evolution inspired by fluvial geomorphology:

- Thermal / talus relaxation: mass moves downslope above a critical gradient
  (angle of repose) — heals one-pixel holes in scarps.
- Stream-power fluvial incision: E ∝ A^m · S^n — carves dendritic valleys that
  align with real drainage networks.

Encoded from standard geomorphology (drainage basin + hillslope–channel
coupling; Whipple/Tucker stream power; Schott-style iterative SPE), not copied
from any textbook PDF.
"""

from __future__ import annotations

from collections import deque
import numpy as np

from .config import ErosionConfig

# Local D8 (avoid importing hydrology → config cycle).
D8 = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
SQRT2 = 1.4142135623730951


def _d8_flow_and_area(dem: np.ndarray) -> np.ndarray:
    h, w = dem.shape
    flow = np.full((h, w), -1, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            z = dem[r, c]
            best = 0.0
            best_d = -1
            for d, (dr, dc) in enumerate(D8):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    best, best_d = 1e9, d
                    continue
                dist = SQRT2 if (dr and dc) else 1.0
                drop = (z - dem[nr, nc]) / dist
                if drop > best:
                    best, best_d = drop, d
            flow[r, c] = best_d if best > 0 else -1

    acc = np.ones((h, w), dtype=np.float64)
    indeg = np.zeros((h, w), dtype=np.int16)
    for r in range(h):
        for c in range(w):
            d = int(flow[r, c])
            if d < 0:
                continue
            dr, dc = D8[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                indeg[nr, nc] += 1
    q = deque((r, c) for r in range(h) for c in range(w) if indeg[r, c] == 0)
    while q:
        r, c = q.popleft()
        d = int(flow[r, c])
        if d < 0:
            continue
        dr, dc = D8[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        acc[nr, nc] += acc[r, c]
        indeg[nr, nc] -= 1
        if indeg[nr, nc] == 0:
            q.append((nr, nc))
    return acc


def thermal_erosion(dem: np.ndarray, cfg: ErosionConfig) -> np.ndarray:
    """Vectorized 4-neighbour talus relaxation on normalized DEM."""
    out = dem.astype(np.float64).copy()
    thr = cfg.talus_tangent
    rate = cfg.thermal_rate
    for _ in range(cfg.thermal_iterations):
        delta = np.zeros_like(out)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(np.roll(out, -dr, axis=0), -dc, axis=1)
            # Fix borders: do not wrap.
            if dr == -1:
                shifted[-1, :] = out[-1, :]
            elif dr == 1:
                shifted[0, :] = out[0, :]
            if dc == -1:
                shifted[:, -1] = out[:, -1]
            elif dc == 1:
                shifted[:, 0] = out[:, 0]
            diff = out - shifted
            excess = np.maximum(0.0, diff - thr)
            move = excess * (0.25 * rate)
            delta -= move
            # Deposit onto neighbour grid.
            deposit = np.roll(np.roll(move, dr, axis=0), dc, axis=1)
            if dr == -1:
                deposit[0, :] = 0
            elif dr == 1:
                deposit[-1, :] = 0
            if dc == -1:
                deposit[:, 0] = 0
            elif dc == 1:
                deposit[:, -1] = 0
            delta += deposit
        out += delta
    return out


def fluvial_stream_power(dem: np.ndarray, cfg: ErosionConfig) -> np.ndarray:
    out = dem.astype(np.float64).copy()
    mean0 = float(out.mean())
    for _ in range(cfg.fluvial_iterations):
        acc = _d8_flow_and_area(out)
        gy, gx = np.gradient(out)
        slope = np.sqrt(gx * gx + gy * gy)
        a = np.maximum(acc, 1.0)
        incision = cfg.stream_power_k * (a**cfg.area_exp) * (np.maximum(slope, 1e-6) ** cfg.slope_exp)
        incision = np.minimum(incision, cfg.max_incision_per_step)
        uplift = cfg.uplift * (out - out.mean())
        out = out - incision + uplift
        out -= float(out.mean()) - mean0
        out = np.maximum(out, 0.0)
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-12)
    return out


def evolve_landscape(dem: np.ndarray, cfg: ErosionConfig | None = None) -> np.ndarray:
    if cfg is None:
        cfg = ErosionConfig()
    if not cfg.enabled:
        return dem
    z = thermal_erosion(dem, cfg)
    z = fluvial_stream_power(z, cfg)
    heal = ErosionConfig(
        enabled=True,
        thermal_iterations=max(3, cfg.thermal_iterations // 3),
        talus_tangent=cfg.talus_tangent * 0.9,
        thermal_rate=cfg.thermal_rate * 0.6,
        fluvial_iterations=0,
    )
    z = thermal_erosion(z, heal)
    z = (z - z.min()) / max(float(z.max() - z.min()), 1e-12)
    return z.astype(np.float64)
