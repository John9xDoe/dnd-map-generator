#!/usr/bin/env bash
# Сборка однофайлового исполняемого приложения (Linux / macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi
"$PYTHON" -m pip install -r requirements.txt pyinstaller
"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --windowed \
  --name ForestMapGenerator \
  --paths "$ROOT/src" \
  --add-data "$ROOT/configs/terrain_config.json:configs" \
  --hidden-import PIL._tkinter_finder \
  gui_main.py

OUT="$ROOT/dist/ForestMapGenerator"
# Удобное имя .exe по запросу (на Linux это всё равно ELF-бинарник).
cp -f "$OUT" "$ROOT/dist/ForestMapGenerator.exe"
echo "Built: $OUT"
echo "Alias: $ROOT/dist/ForestMapGenerator.exe"
ls -lh "$ROOT/dist/"
