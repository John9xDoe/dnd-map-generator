"""Bracketed L-systems for provisional tree glyphs (PCG Book Ch.5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import ImageDraw


@dataclass
class LSystem:
    axiom: str
    rules: dict[str, str]
    angle_deg: float = 25.0
    iterations: int = 3


PINE = LSystem(
    axiom="X",
    rules={"X": "F[+X][-X]FX", "F": "FF"},
    angle_deg=28.0,
    iterations=3,
)
BROADLEAF = LSystem(
    axiom="X",
    rules={"X": "F-[[X]+X]+F[+FX]-X", "F": "FF"},
    angle_deg=22.5,
    iterations=3,
)


_EXPAND_CACHE: dict[tuple[str, int], str] = {}


def expand(ls: LSystem) -> str:
    cache_key = (ls.axiom, tuple(sorted(ls.rules.items())), ls.iterations)
    if cache_key in _EXPAND_CACHE:
        return _EXPAND_CACHE[cache_key]
    s = ls.axiom
    for _ in range(ls.iterations):
        s = "".join(ls.rules.get(ch, ch) for ch in s)
    _EXPAND_CACHE[cache_key] = s
    return s


def draw_lsystem_tree(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    *,
    kind: int = 0,
    scale: float = 3.0,
    trunk_color: tuple[int, int, int] = (70, 50, 30),
    leaf_color: tuple[int, int, int] = (42, 92, 48),
) -> None:
    """Turtle-interpret a small L-system rooted at (x,y), growing upward."""
    ls = PINE if kind % 2 == 0 else BROADLEAF
    # Map icons: keep L-system shallow for speed.
    local = LSystem(ls.axiom, ls.rules, ls.angle_deg, iterations=2)
    prog = expand(local)
    step = max(1.0, 0.45 * scale)
    angle = -90.0  # up
    stack: list[tuple[float, float, float]] = []
    cx, cy = x, y
    for ch in prog:
        if ch == "F":
            rad = math.radians(angle)
            nx = cx + step * math.cos(rad)
            ny = cy + step * math.sin(rad)
            # Trunk-ish lower segments darker; tips greener.
            col = leaf_color if abs(angle + 90) > 25 else trunk_color
            draw.line([(cx, cy), (nx, ny)], fill=col, width=max(1, int(scale * 0.35)))
            cx, cy = nx, ny
        elif ch == "+":
            angle += local.angle_deg
        elif ch == "-":
            angle -= local.angle_deg
        elif ch == "[":
            stack.append((cx, cy, angle))
        elif ch == "]":
            if stack:
                cx, cy, angle = stack.pop()
        elif ch == "X":
            # Apex leaf mark.
            draw.ellipse(
                [cx - scale * 0.35, cy - scale * 0.35, cx + scale * 0.35, cy + scale * 0.35],
                fill=leaf_color,
            )
