"""
Doran & Parberry–inspired terrain agents (simplified).

After base noise + SPE:
  1. Mountain agents raise ridge wedges.
  2. River agents dig downhill from ridges, widening toward outlets.
"""

from __future__ import annotations

import numpy as np

from .config import AgentConfig


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def mountain_agents(dem: np.ndarray, rng: np.random.Generator, cfg: AgentConfig) -> np.ndarray:
    out = dem.astype(np.float64).copy()
    h, w = out.shape
    for _ in range(cfg.n_mountain_agents):
        flat = out.ravel()
        probs = flat - flat.min() + 1e-3
        probs = probs / probs.sum()
        idx = int(rng.choice(flat.size, p=probs))
        r, c = divmod(idx, w)
        gy, gx = np.gradient(out)
        angle = float(np.arctan2(gy[r, c], gx[r, c])) + np.pi / 2
        angle += float(rng.uniform(-0.4, 0.4))
        width = cfg.mountain_width
        for step in range(cfg.mountain_steps):
            for dr in range(-width, width + 1):
                for dc in range(-width, width + 1):
                    if abs(dr) + abs(dc) > width + 1:
                        continue
                    rr, cc = _clamp(r + dr, 0, h - 1), _clamp(c + dc, 0, w - 1)
                    fall = 1.0 - (abs(dr) + abs(dc)) / (width + 2)
                    out[rr, cc] += (
                        cfg.mountain_raise
                        * fall
                        * (1.0 - step / (cfg.mountain_steps + 1))
                    )
            angle += float(rng.uniform(-0.25, 0.25))
            r = _clamp(int(round(r + np.sin(angle))), 0, h - 1)
            c = _clamp(int(round(c + np.cos(angle))), 0, w - 1)
            if rng.random() < 0.08:
                width = max(1, width + int(rng.integers(-1, 2)))
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-12)
    return out


def _steepest_down(dem: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    h, w = dem.shape
    best = None
    best_drop = 0.0
    z = dem[r, c]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return nr, nc
            drop = z - dem[nr, nc]
            dist = 1.414 if (dr and dc) else 1.0
            score = drop / dist
            if score > best_drop:
                best_drop = score
                best = (nr, nc)
    return best if best_drop > 0 else None


def river_agents(
    dem: np.ndarray,
    rng: np.random.Generator,
    cfg: AgentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    out = dem.astype(np.float64).copy()
    h, w = out.shape
    river = np.zeros((h, w), dtype=bool)
    thr = float(np.quantile(out, 0.82))
    ridges = np.argwhere(out >= thr)
    if len(ridges) == 0:
        return out, river

    carved = 0
    attempts = 0
    while carved < cfg.n_river_agents and attempts < cfg.n_river_agents * cfg.river_max_attempts:
        attempts += 1
        r0, c0 = ridges[int(rng.integers(0, len(ridges)))]
        r, c = int(r0), int(c0)
        path: list[tuple[int, int]] = [(r, c)]
        visited = {(r, c)}
        for _ in range(h + w):
            nxt = _steepest_down(out, r, c)
            if nxt is None:
                break
            nr, nc = nxt
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if (nr, nc) in visited:
                break
            path.append((nr, nc))
            visited.add((nr, nc))
            r, c = nr, nc
            if r <= 1 or c <= 1 or r >= h - 2 or c >= w - 2:
                break
        if len(path) < cfg.river_min_length:
            continue

        width = cfg.river_start_width
        for i, (rr, cc) in enumerate(path):
            if i > 0 and i % cfg.river_widen_every == 0:
                width += 0.35
            rad = max(0, int(round(width)))
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if dr * dr + dc * dc > width * width:
                        continue
                    tr, tc = rr + dr, cc + dc
                    if 0 <= tr < h and 0 <= tc < w:
                        out[tr, tc] -= cfg.river_carve * (
                            1.0 - 0.35 * (abs(dr) + abs(dc)) / (width + 1)
                        )
                        river[tr, tc] = True
        carved += 1

    out = np.clip(out, 0.0, None)
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-12)
    return out, river


def apply_terrain_agents(
    dem: np.ndarray,
    seed: int,
    cfg: AgentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg is None:
        cfg = AgentConfig()
    if not cfg.enabled:
        return dem, np.zeros(dem.shape, dtype=bool)
    rng = np.random.default_rng(seed + 911)
    dem = mountain_agents(dem, rng, cfg)
    dem, rivers = river_agents(dem, rng, cfg)
    return dem, rivers
