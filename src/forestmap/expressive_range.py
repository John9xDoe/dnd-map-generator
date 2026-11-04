"""Expressive-range analysis (PCG Book Ch.12)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import TerrainConfig, load_config
from .pipeline import generate_map_once


def collect_metrics(cfg: TerrainConfig, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        c = deepcopy(cfg)
        c.seed = int(seed)
        c.quality.enabled = False  # measure raw generator range
        c.render.show_contours = False
        result = generate_map_once(c, save=False)
        fit = result.stats.get("fitness", {})
        rows.append(
            {
                "seed": seed,
                "lake_fraction": result.stats["lake_fraction"],
                "river_cells": result.stats["river_cells"],
                "mean_slope": result.stats["mean_slope"],
                "tree_count": result.stats["tree_count"],
                "cliff_cells": result.stats["cliff_cells"],
                "passable_fraction": fit.get("passable_fraction", 0.0),
                "largest_cc_fraction": fit.get("largest_cc_fraction", 0.0),
                "camp_sites": fit.get("camp_sites", 0),
                "fitness": fit.get("score", 0.0),
            }
        )
    return rows


def plot_expressive_range(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lake = [r["lake_fraction"] for r in rows]
    slope = [r["mean_slope"] for r in rows]
    rivers = [r["river_cells"] for r in rows]
    fit = [r["fitness"] for r in rows]
    camps = [r["camp_sites"] for r in rows]
    passable = [r["passable_fraction"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    sc = axes[0, 0].scatter(lake, slope, c=fit, cmap="viridis", s=40)
    axes[0, 0].set_xlabel("lake_fraction")
    axes[0, 0].set_ylabel("mean_slope")
    axes[0, 0].set_title("lake vs slope (color=fitness)")
    fig.colorbar(sc, ax=axes[0, 0], fraction=0.046)

    axes[0, 1].scatter(rivers, camps, c=fit, cmap="plasma", s=40)
    axes[0, 1].set_xlabel("river_cells")
    axes[0, 1].set_ylabel("camp_sites")
    axes[0, 1].set_title("drainage vs camps")

    axes[1, 0].scatter(passable, fit, c=slope, cmap="magma", s=40)
    axes[1, 0].set_xlabel("passable_fraction")
    axes[1, 0].set_ylabel("fitness")
    axes[1, 0].set_title("DnD passability vs fitness")

    axes[1, 1].hist(fit, bins=min(12, max(4, len(rows) // 2)), color="#4a7c59")
    axes[1, 1].set_xlabel("fitness")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("fitness distribution")

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_expressive_range(
    config_path: str | Path,
    seeds: list[int],
    out_png: str | Path = "output/expressive_range.png",
    out_csv: str | Path = "output/expressive_range.csv",
) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg.width = min(cfg.width, 160)
    cfg.height = min(cfg.height, 160)
    # Speed: fewer SPE / agent iterations for batch.
    cfg.erosion.fluvial_iterations = min(cfg.erosion.fluvial_iterations, 12)
    cfg.agents.n_mountain_agents = min(cfg.agents.n_mountain_agents, 4)
    cfg.agents.n_river_agents = min(cfg.agents.n_river_agents, 3)
    cfg.agents.mountain_steps = min(cfg.agents.mountain_steps, 50)

    rows = collect_metrics(cfg, seeds)
    png = plot_expressive_range(rows, out_png)
    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    return {"rows": rows, "png": str(png), "csv": str(csv_path)}
