@echo off
REM Build Windows .exe (run on Windows with Python 3.11+)
cd /d %~dp0
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name ForestMapGenerator --paths src --add-data "configs/terrain_config.json;configs" --hidden-import PIL._tkinter_finder gui_main.py
echo Built dist\ForestMapGenerator.exe
dir dist
