#!/usr/bin/env python3
"""CLI: generate a symbolic forest relief map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from forestmap.config import load_config  # noqa: E402
from forestmap.pipeline import generate_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Генератор реалистичной лесной карты (рельеф → гидрология → лес)."
    )
    p.add_argument(
        "-c",
        "--config",
        default=str(ROOT / "configs" / "terrain_config.json"),
        help="Путь к JSON-конфигу",
    )
    p.add_argument("--seed", type=int, default=None, help="Переопределить seed")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument(
        "--show",
        action="store_true",
        help="Показать карту в окне pygame (если есть дисплей)",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Открыть графический редактор (tkinter)",
    )
    p.add_argument(
        "--expressive-range",
        type=int,
        nargs="?",
        const=16,
        default=None,
        metavar="N",
        help="Построить expressive-range по N сидам начиная с --seed",
    )
    p.add_argument(
        "--no-quality",
        action="store_true",
        help="Отключить generate-and-test отбор",
    )
    return p.parse_args()


def maybe_show_pygame(image_path: Path) -> None:
    import os

    try:
        import pygame
        from PIL import Image
    except ImportError:
        print("pygame/Pillow недоступны для --show")
        return

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("Нет DISPLAY — пропускаю окно pygame, карта сохранена в файл.")
        return

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Forest map — условные знаки")
    surface = pygame.image.fromstring(img.tobytes(), img.size, "RGB")
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    print("Закройте окно или нажмите Esc/Q.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
    pygame.quit()


def main() -> int:
    args = parse_args()
    if args.gui:
        from forestmap.gui.app import run_app

        run_app()
        return 0

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.width is not None:
        cfg.width = args.width
    if args.height is not None:
        cfg.height = args.height
    if args.no_quality:
        cfg.quality.enabled = False

    if args.expressive_range is not None:
        from forestmap.expressive_range import run_expressive_range

        n = int(args.expressive_range)
        seeds = list(range(int(cfg.seed), int(cfg.seed) + n))
        print(f"Expressive range: {n} сидов {seeds[0]}…{seeds[-1]}")
        out = run_expressive_range(
            args.config,
            seeds,
            out_png=cfg.render.expressive_range_path,
        )
        print(json.dumps({"png": out["png"], "csv": out["csv"], "n": len(out["rows"])}, indent=2))
        # brief table
        for row in out["rows"][:5]:
            print(
                f"  seed={row['seed']} fitness={row['fitness']:.3f} "
                f"lake={row['lake_fraction']:.3f} camps={row['camp_sites']}"
            )
        if len(out["rows"]) > 5:
            print(f"  … +{len(out['rows']) - 5} rows")
        return 0

    print(
        f"Генерация карты {cfg.width}×{cfg.height}, seed={cfg.seed}, "
        f"ячейка={cfg.cell_size_m} м …"
    )
    result = generate_map(cfg)
    print(json.dumps(result.stats, ensure_ascii=False, indent=2, default=str))
    print(f"Карта:  {result.outputs['map']}")
    print(f"Отладка: {result.outputs['debug']}")

    if args.show:
        maybe_show_pygame(result.outputs["map"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
