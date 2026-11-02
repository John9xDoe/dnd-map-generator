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
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.width is not None:
        cfg.width = args.width
    if args.height is not None:
        cfg.height = args.height

    print(
        f"Генерация карты {cfg.width}×{cfg.height}, seed={cfg.seed}, "
        f"ячейка={cfg.cell_size_m} м …"
    )
    result = generate_map(cfg)
    print(json.dumps(result.stats, ensure_ascii=False, indent=2))
    print(f"Карта:  {result.outputs['map']}")
    print(f"Отладка: {result.outputs['debug']}")

    if args.show:
        maybe_show_pygame(result.outputs["map"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
