"""
Hydrology on a DEM following GIS practice:

1. Priority-Flood depression fill (Barnes et al., 2014) with epsilon micro-gradient
2. D8 steepest-descent flow directions
3. Flow accumulation (upstream catchment size)
4. Lakes = sufficiently deep filled pits (closed basins)
5. Rivers = high-accumulation paths drained off lakes / map edge
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import HydrologyConfig

# D8: index → (dr, dc). Order matches flow encoding 0..7.
D8 = (
    (-1, 0),  # N
    (-1, 1),  # NE
    (0, 1),  # E
    (1, 1),  # SE
    (1, 0),  # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1),  # NW
)
SQRT2 = np.sqrt(2.0)


@dataclass
class HydrologyResult:
    filled: np.ndarray
    fill_depth: np.ndarray
    flow_dir: np.ndarray  # int8, -1 = outlet/sink
    accumulation: np.ndarray
    lake_mask: np.ndarray
    lake_labels: np.ndarray
    river_mask: np.ndarray
    river_paths: list[list[tuple[int, int]]]


def priority_flood_fill(dem: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Fill depressions so every cell drains to the border.
    Epsilon imposes a tiny outward gradient across former flats/pits.
    """
    h, w = dem.shape
    filled = dem.astype(np.float64).copy()
    in_q = np.zeros((h, w), dtype=bool)
    pq: list[tuple[float, int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            heapq.heappush(pq, (filled[r, c], r, c))
            in_q[r, c] = True
    for c in range(w):
        for r in (0, h - 1):
            if not in_q[r, c]:
                heapq.heappush(pq, (filled[r, c], r, c))
                in_q[r, c] = True

    while pq:
        elev, r, c = heapq.heappop(pq)
        for dr, dc in D8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or in_q[nr, nc]:
                continue
            # Raise pit cells just above the spill elevation.
            nxt = filled[nr, nc]
            if nxt < elev + epsilon:
                nxt = elev + epsilon
                filled[nr, nc] = nxt
            in_q[nr, nc] = True
            heapq.heappush(pq, (nxt, nr, nc))
    return filled


def d8_flow_directions(filled: np.ndarray) -> np.ndarray:
    """Steepest descent among 8 neighbors. -1 marks local/border sinks."""
    h, w = filled.shape
    flow = np.full((h, w), -1, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            z = filled[r, c]
            best_drop = 0.0
            best_dir = -1
            for d, (dr, dc) in enumerate(D8):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    # Leaving the map counts as infinite drop → outlet.
                    best_dir = d
                    best_drop = 1e9
                    continue
                dist = SQRT2 if (dr and dc) else 1.0
                drop = (z - filled[nr, nc]) / dist
                if drop > best_drop:
                    best_drop = drop
                    best_dir = d
            flow[r, c] = best_dir if best_drop > 0 else -1
    return flow


def flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """Each cell starts with 1 unit of rain; drain downslope."""
    h, w = flow_dir.shape
    acc = np.ones((h, w), dtype=np.float64)
    indeg = np.zeros((h, w), dtype=np.int16)

    for r in range(h):
        for c in range(w):
            d = int(flow_dir[r, c])
            if d < 0:
                continue
            dr, dc = D8[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                indeg[nr, nc] += 1

    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if indeg[r, c] == 0:
                q.append((r, c))

    while q:
        r, c = q.popleft()
        d = int(flow_dir[r, c])
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


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    lid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c]:
                continue
            lid += 1
            stack = [(r, c)]
            labels[r, c] = lid
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
    return labels, lid


def extract_lakes(
    dem: np.ndarray,
    filled: np.ndarray,
    cfg: HydrologyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed basins deep/wide enough to hold standing water.

    Shallow micro-pits are ignored (they become river/through-flow after fill).
    Among qualifying basins we keep the largest by water volume until area /
    count caps — matching temperate forest landscapes with few true lakes.
    """
    depth = filled - dem
    candidates = depth >= cfg.lake_min_depth
    # Ignore thin border-touching "lakes" (usually ocean/outflow flats).
    candidates[0, :] = False
    candidates[-1, :] = False
    candidates[:, 0] = False
    candidates[:, -1] = False

    labels, n = _label_components(candidates)
    scored: list[tuple[float, int, int]] = []
    for i in range(1, n + 1):
        cells = int(np.count_nonzero(labels == i))
        if cells < cfg.lake_min_cells:
            continue
        volume = float(depth[labels == i].sum())
        scored.append((volume, cells, i))
    scored.sort(reverse=True)

    budget = int(dem.size * cfg.lake_max_fraction)
    used = 0
    keep_ids: list[int] = []
    for volume, cells, i in scored:
        if len(keep_ids) >= cfg.lake_max_count:
            break
        if used + cells > budget and keep_ids:
            break
        keep_ids.append(i)
        used += cells

    new_labels = np.zeros_like(labels)
    for new_id, old in enumerate(keep_ids, start=1):
        new_labels[labels == old] = new_id
    lake_mask = new_labels > 0
    return lake_mask, new_labels


def extract_rivers(
    flow_dir: np.ndarray,
    accumulation: np.ndarray,
    lake_mask: np.ndarray,
    cfg: HydrologyConfig,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = flow_dir.shape
    thr = float(np.quantile(accumulation, cfg.river_accumulation_quantile))
    thr = max(thr, 8.0)
    seeds = np.argwhere((accumulation >= thr) & (~lake_mask))
    # Prefer mid/high catchment seeds; sort so longer trunks first.
    seeds = seeds[np.argsort(-accumulation[seeds[:, 0], seeds[:, 1]])]

    river_mask = np.zeros((h, w), dtype=bool)
    paths: list[list[tuple[int, int]]] = []
    visited_start = np.zeros((h, w), dtype=bool)

    for r0, c0 in seeds:
        r0, c0 = int(r0), int(c0)
        if visited_start[r0, c0] or river_mask[r0, c0]:
            continue
        path: list[tuple[int, int]] = []
        r, c = r0, c0
        guard = 0
        while guard < h * w:
            guard += 1
            if not (0 <= r < h and 0 <= c < w):
                break
            if lake_mask[r, c]:
                path.append((r, c))
                break
            path.append((r, c))
            visited_start[r, c] = True
            d = int(flow_dir[r, c])
            if d < 0:
                break
            dr, dc = D8[d]
            r, c = r + dr, c + dc
            if path and (r, c) == path[-1]:
                break

        if len(path) < cfg.min_river_length:
            continue
        for r, c in path:
            if 0 <= r < h and 0 <= c < w and not lake_mask[r, c]:
                river_mask[r, c] = True
        paths.append(path)

    return river_mask, paths


def analyze_hydrology(dem: np.ndarray, cfg: HydrologyConfig) -> HydrologyResult:
    filled = priority_flood_fill(dem, cfg.epsilon)
    fill_depth = filled - dem
    lake_mask, lake_labels = extract_lakes(dem, filled, cfg)
    flow_dir = d8_flow_directions(filled)
    # Lakes act as sinks for accumulation aesthetics: keep routing through filled
    # DEM so rivers still find outlets, but mask lakes when painting channels.
    accumulation = flow_accumulation(flow_dir)
    river_mask, river_paths = extract_rivers(flow_dir, accumulation, lake_mask, cfg)
    return HydrologyResult(
        filled=filled,
        fill_depth=fill_depth,
        flow_dir=flow_dir,
        accumulation=accumulation,
        lake_mask=lake_mask,
        lake_labels=lake_labels,
        river_mask=river_mask,
        river_paths=river_paths,
    )
