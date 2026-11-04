#!/usr/bin/env python3
"""GUI entrypoint (also used by PyInstaller)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = ROOT / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from forestmap.gui.app import run_app


def main() -> int:
    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
