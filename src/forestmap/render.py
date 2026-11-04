"""Cartographic-style symbolic rendering (provisional map signs on a canvas)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import RenderConfig
from .features import FeatureResult
from .hydrology import HydrologyResult


def _font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _hypsometric(dem: np.ndarray) -> np.ndarray:
    """Muted land tint: low moist green → mid olive → high rock tan."""
    low = np.array([92, 120, 72], dtype=np.float64)
    mid = np.array([145, 138, 92], dtype=np.float64)
    high = np.array([168, 150, 128], dtype=np.float64)
    z = dem.astype(np.float64)
    t_low = np.clip(z / 0.55, 0.0, 1.0)[..., None]
    t_high = np.clip((z - 0.55) / 0.45, 0.0, 1.0)[..., None]
    low_mid = low * (1.0 - t_low) + mid * t_low
    mid_high = mid * (1.0 - t_high) + high * t_high
    rgb = np.where(z[..., None] < 0.55, low_mid, mid_high)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _draw_contours(
    draw: ImageDraw.ImageDraw,
    dem: np.ndarray,
    scale: int,
    levels: int,
) -> None:
    if levels <= 0:
        return
    h, w = dem.shape
    qs = np.linspace(0.15, 0.9, levels)
    for q in qs:
        level = float(np.quantile(dem, q))
        mask = dem >= level
        edge = mask.copy()
        edge[1:-1, 1:-1] = mask[1:-1, 1:-1] & ~(
            mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        )
        rs, cs = np.where(edge)
        # Thin points for readability.
        for r, c in zip(rs[::2], cs[::2]):
            x = int(c) * scale + scale // 2
            y = int(r) * scale + scale // 2
            draw.point((x, y), fill=(110, 105, 88, 110))


def _draw_lake(draw: ImageDraw.ImageDraw, lake_mask: np.ndarray, scale: int) -> None:
    h, w = lake_mask.shape
    for r in range(h):
        for c in range(w):
            if not lake_mask[r, c]:
                continue
            x0, y0 = c * scale, r * scale
            draw.rectangle([x0, y0, x0 + scale, y0 + scale], fill=(70, 120, 168))
            # Wave sign (условный знак озера).
            if (r + c) % 3 == 0 and scale >= 3:
                y = y0 + scale // 2
                draw.arc([x0 + 1, y - 1, x0 + scale - 1, y + scale // 2], 0, 180, fill=(200, 220, 235))


def _river_width(accumulation: np.ndarray, r: int, c: int) -> int:
    a = float(accumulation[r, c])
    if a < 40:
        return 1
    if a < 120:
        return 2
    if a < 400:
        return 3
    return 4


def _draw_rivers(
    draw: ImageDraw.ImageDraw,
    hydro: HydrologyResult,
    scale: int,
) -> None:
    for path in hydro.river_paths:
        if len(path) < 2:
            continue
        pts: list[tuple[int, int]] = []
        widths: list[int] = []
        for r, c in path:
            if hydro.lake_mask[r, c]:
                break
            pts.append((c * scale + scale // 2, r * scale + scale // 2))
            widths.append(_river_width(hydro.accumulation, r, c))
        for i in range(len(pts) - 1):
            w = max(widths[i], widths[i + 1])
            draw.line([pts[i], pts[i + 1]], fill=(55, 105, 170), width=w)


def _cliff_hachures(
    draw: ImageDraw.ImageDraw,
    cliff_mask: np.ndarray,
    dem: np.ndarray,
    scale: int,
) -> None:
    """
    Continuous escarpment: filled cliff band + dense downslope hachures
    along the scarp front (no speckled single-pixel holes).
    """
    h, w = cliff_mask.shape
    # Solid band underlay so holes don't show through hypsometry.
    for r, c in zip(*np.where(cliff_mask)):
        x0, y0 = int(c) * scale, int(r) * scale
        draw.rectangle(
            [x0, y0, x0 + scale, y0 + scale],
            fill=(118, 96, 72, 210),
        )

    gy, gx = np.gradient(dem.astype(np.float64))
    # Scarp front = cliff cell with a non-cliff neighbour.
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not cliff_mask[r, c]:
                continue
            if cliff_mask[r - 1 : r + 2, c - 1 : c + 2].all():
                continue
            # Skip every other rim cell only when scale is tiny; otherwise dense.
            if scale <= 2 and ((r + c) & 1):
                continue
            x = c * scale + scale // 2
            y = r * scale + scale // 2
            vx, vy = float(gx[r, c]), float(gy[r, c])
            norm = max((vx * vx + vy * vy) ** 0.5, 1e-9)
            ux, uy = vx / norm, vy / norm
            length = max(scale + 1, int(1.6 * scale))
            x1 = int(x + ux * length)
            y1 = int(y + uy * length)
            draw.line([(x, y), (x1, y1)], fill=(78, 56, 38, 230), width=max(1, scale // 2))
            px, py = -uy, ux
            draw.line(
                [
                    (int(x - px * (scale * 0.7)), int(y - py * (scale * 0.7))),
                    (int(x + px * (scale * 0.7)), int(y + py * (scale * 0.7))),
                ],
                fill=(60, 44, 30, 230),
                width=1,
            )


def _draw_tree(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    scale: int,
    kind: int,
    *,
    use_lsystem: bool = True,
) -> None:
    """Forest signs: L-system glyphs (Ch.5) or simple geometric fallback."""
    if use_lsystem:
        from .lsystem import draw_lsystem_tree

        leaf = (42, 92, 48) if kind % 2 == 0 else (58, 110, 55)
        draw_lsystem_tree(
            draw, x, y, kind=kind, scale=float(max(2, scale)), leaf_color=leaf
        )
        return
    s = max(2, scale)
    if kind == 0:
        top = (int(x), int(y - 1.6 * s))
        left = (int(x - 1.1 * s), int(y + 0.4 * s))
        right = (int(x + 1.1 * s), int(y + 0.4 * s))
        draw.polygon([top, left, right], fill=(42, 92, 48), outline=(28, 64, 34))
        draw.line([(int(x), int(y + 0.4 * s)), (int(x), int(y + 1.2 * s))], fill=(70, 50, 30), width=1)
    else:
        r = int(1.1 * s)
        draw.ellipse(
            [int(x - r), int(y - r - s // 2), int(x + r), int(y + r - s // 2)],
            fill=(58, 110, 55),
            outline=(35, 75, 40),
        )
        draw.line(
            [(int(x), int(y + r - s // 2)), (int(x), int(y + r + s // 2))],
            fill=(70, 50, 30),
            width=1,
        )


def _legend(draw: ImageDraw.ImageDraw, width_px: int, height_px: int, scale: int) -> None:
    font = _font(13)
    font_title = _font(14)
    box_w, box_h = 168, 118
    x0 = width_px - box_w - 10
    y0 = 10
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(245, 242, 230), outline=(80, 80, 70))
    draw.text((x0 + 8, y0 + 6), "Условные знаки", fill=(40, 40, 35), font=font_title)
    draw.rectangle([x0 + 10, y0 + 28, x0 + 28, y0 + 40], fill=(70, 120, 168))
    draw.text((x0 + 36, y0 + 26), "озеро", fill=(40, 40, 35), font=font)
    draw.line([(x0 + 10, y0 + 52), (x0 + 28, y0 + 52)], fill=(55, 105, 170), width=2)
    draw.text((x0 + 36, y0 + 44), "река", fill=(40, 40, 35), font=font)
    draw.line([(x0 + 10, y0 + 68), (x0 + 22, y0 + 78)], fill=(92, 70, 48), width=1)
    draw.line([(x0 + 10, y0 + 68), (x0 + 18, y0 + 66)], fill=(70, 52, 36), width=1)
    draw.text((x0 + 36, y0 + 62), "обрыв", fill=(40, 40, 35), font=font)
    _draw_tree(draw, x0 + 18, y0 + 98, max(2, scale), 0)
    draw.text((x0 + 36, y0 + 86), "лес", fill=(40, 40, 35), font=font)


def render_symbolic_map(
    dem: np.ndarray,
    hydro: HydrologyResult,
    features: FeatureResult,
    cfg: RenderConfig,
    seed: int,
) -> Image.Image:
    scale = max(1, int(cfg.pixel_scale))
    h, w = dem.shape
    base = _hypsometric(dem)
    # Slightly darken cliffs / water later overlaid.
    img = Image.fromarray(base, mode="RGB").resize(
        (w * scale, h * scale), resample=Image.Resampling.NEAREST
    )
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    if cfg.show_contours:
        _draw_contours(draw, dem, scale, cfg.contour_levels)

    # Land cover first, then water on top — trees never visually sit on water.
    _cliff_hachures(draw, features.cliff_mask, dem, scale)

    water_block = getattr(features, "water_block", hydro.lake_mask | hydro.river_mask)
    rng = np.random.default_rng(seed + 99)
    hh, ww = dem.shape
    for r, c in features.trees:
        ir = int(np.clip(round(float(r)), 0, hh - 1))
        ic = int(np.clip(round(float(c)), 0, ww - 1))
        if water_block[ir, ic] or features.cliff_mask[ir, ic]:
            continue
        # Also reject if any neighbour cell in symbol footprint is water.
        pad = 1
        r0, r1 = max(0, ir - pad), min(hh, ir + pad + 1)
        c0, c1 = max(0, ic - pad), min(ww, ic + pad + 1)
        if water_block[r0:r1, c0:c1].any():
            continue
        x = float(c) * scale + scale / 2
        y = float(r) * scale + scale / 2
        _draw_tree(
            draw,
            x,
            y,
            scale,
            int(rng.integers(0, 2)),
            use_lsystem=bool(getattr(cfg, "use_lsystem_trees", True)),
        )

    _draw_lake(draw, hydro.lake_mask, scale)
    _draw_rivers(draw, hydro, scale)

    _legend(draw, img.width, img.height, scale)
    return img.convert("RGB")


def render_debug_panels(
    dem: np.ndarray,
    hydro: HydrologyResult,
    features: FeatureResult,
) -> Image.Image:
    """Side-by-side scientific panels for verifying the pipeline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], dem, "DEM (высота)", "terrain"),
        (axes[0, 1], features.slope, "Уклон |∇z|", "magma"),
        (axes[0, 2], hydro.fill_depth, "Глубина заполнения котловин", "Blues"),
        (axes[1, 0], np.log1p(hydro.accumulation), "log(сток)", "viridis"),
        (axes[1, 1], features.moisture, "Влажность (TWI)", "YlGnBu"),
        (axes[1, 2], features.suitability, "Пригодность для леса", "Greens"),
    ]
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overlay masks on DEM-like composite in moisture panel annotation via contours.
    mask = np.zeros((*dem.shape, 3), dtype=float)
    mask[..., 1] = features.cliff_mask * 0.8
    mask[..., 2] = hydro.lake_mask * 0.9
    mask[..., 0] = hydro.river_mask * 0.9
    axes[0, 0].imshow(mask, alpha=0.35, origin="upper")

    fig.canvas.draw()
    fig.savefig("/tmp/forest_debug_panels.png", dpi=120)
    plt.close(fig)
    return Image.open("/tmp/forest_debug_panels.png").convert("RGB")


def save_outputs(
    dem: np.ndarray,
    hydro: HydrologyResult,
    features: FeatureResult,
    cfg: RenderConfig,
    seed: int,
) -> dict[str, Path]:
    out_map = Path(cfg.output_path)
    out_dbg = Path(cfg.debug_panels_path)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    out_dbg.parent.mkdir(parents=True, exist_ok=True)

    symbolic = render_symbolic_map(dem, hydro, features, cfg, seed)
    symbolic.save(out_map)

    debug = render_debug_panels(dem, hydro, features)
    debug.save(out_dbg)
    return {"map": out_map, "debug": out_dbg}
