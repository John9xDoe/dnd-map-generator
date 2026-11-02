"""Tk GUI: map (zoom/pan/layers) left, parameters + brushes right."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np
from PIL import Image, ImageTk

from forestmap.config import TerrainConfig, load_config
from forestmap.map_session import (
    BRUSH_ERASE,
    BRUSH_LAKE,
    BRUSH_MOUNTAIN,
    BRUSH_RIVER,
    BRUSH_TRAIL,
    LAYER_NAMES,
    MapSession,
    apply_brush,
    build_session,
    layer_to_rgb,
    rebuild_from_dem,
    refresh_symbolic,
)


def _default_config_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "configs" / "terrain_config.json",
        here.parents[3] / "configs" / "terrain_config.json",
        here.parents[2] / "configs" / "terrain_config.json",
    ]
    # PyInstaller bundle
    import sys

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.insert(0, Path(sys._MEIPASS) / "configs" / "terrain_config.json")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


class MapCanvas(ttk.Frame):
    """Left pane: scrollable/zoomable map with brush painting."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_paint: Callable[[float, float], None],
        on_hover: Callable[[float, float], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.on_paint = on_paint
        self.on_hover = on_hover
        self.zoom = 1.0
        self._photo: ImageTk.PhotoImage | None = None
        self._base_rgb: np.ndarray | None = None
        self._img_id: int | None = None
        self._painting = False
        self._panning = False
        self._pan_last: tuple[int, int] | None = None
        self.grid_h = 256
        self.grid_w = 256

        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(toolbar, text="Масштаб").pack(side=tk.LEFT)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_scale = ttk.Scale(
            toolbar,
            from_=0.25,
            to=6.0,
            variable=self.zoom_var,
            command=self._on_zoom_slider,
            length=160,
        )
        self.zoom_scale.pack(side=tk.LEFT, padx=6)
        self.zoom_label = ttk.Label(toolbar, text="100%")
        self.zoom_label.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Сбросить вид", command=self.reset_view).pack(
            side=tk.RIGHT, padx=4
        )

        wrap = ttk.Frame(self)
        wrap.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(wrap, bg="#2b2b2b", highlightthickness=0)
        self.hbar = ttk.Scrollbar(wrap, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.status = ttk.Label(self, text="Клетка: —")
        self.status.pack(fill=tk.X, padx=4, pady=2)

        self.canvas.bind("<ButtonPress-1>", self._paint_start)
        self.canvas.bind("<B1-Motion>", self._paint_move)
        self.canvas.bind("<ButtonRelease-1>", self._paint_end)
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._pan_end)
        self.canvas.bind("<ButtonPress-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._pan_end)
        self.canvas.bind("<Motion>", self._motion)
        self.canvas.bind("<Control-ButtonPress-1>", self._pan_start)
        self.canvas.bind("<Control-B1-Motion>", self._pan_move)
        self.canvas.bind("<Control-ButtonRelease-1>", self._pan_end)
        # Linux / Windows wheel
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._wheel_linux(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._wheel_linux(e, -1))

    def set_image(self, rgb: np.ndarray, grid_h: int, grid_w: int) -> None:
        self._base_rgb = rgb
        self.grid_h, self.grid_w = grid_h, grid_w
        self._redraw()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.zoom_var.set(1.0)
        self.zoom_label.config(text="100%")
        self._redraw()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _on_zoom_slider(self, _v: str) -> None:
        self.zoom = float(self.zoom_var.get())
        self.zoom_label.config(text=f"{int(self.zoom * 100)}%")
        self._redraw()

    def _redraw(self) -> None:
        if self._base_rgb is None:
            return
        h, w, _ = self._base_rgb.shape
        nh, nw = max(1, int(h * self.zoom)), max(1, int(w * self.zoom))
        img = Image.fromarray(self._base_rgb, mode="RGB").resize(
            (nw, nh), Image.Resampling.NEAREST
        )
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("map")
        self._img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo, tags="map")
        self.canvas.configure(scrollregion=(0, 0, nw, nh))

    def _event_to_grid(self, event) -> tuple[float, float] | None:
        if self._base_rgb is None:
            return None
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        disp_h = max(1, int(self._base_rgb.shape[0] * self.zoom))
        disp_w = max(1, int(self._base_rgb.shape[1] * self.zoom))
        if not (0 <= x < disp_w and 0 <= y < disp_h):
            return None
        # Map display pixels → grid cells (layer may be upscaled symbolic).
        col = x / disp_w * self.grid_w
        row = y / disp_h * self.grid_h
        return row, col

    def _paint_start(self, event) -> None:
        if event.state & 0x0004:  # Control → pan
            return
        self._painting = True
        self._emit_paint(event)

    def _paint_move(self, event) -> None:
        if self._painting:
            self._emit_paint(event)

    def _paint_end(self, _event) -> None:
        self._painting = False

    def _emit_paint(self, event) -> None:
        cell = self._event_to_grid(event)
        if cell:
            self.on_paint(*cell)

    def _pan_start(self, event) -> None:
        self._panning = True
        self._pan_last = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def _pan_move(self, event) -> None:
        if self._panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _pan_end(self, _event) -> None:
        self._panning = False
        self._pan_last = None

    def _motion(self, event) -> None:
        cell = self._event_to_grid(event)
        if cell:
            r, c = cell
            self.status.config(text=f"Клетка: row={r:.1f}, col={c:.1f} · zoom={int(self.zoom*100)}%")
            if self.on_hover:
                self.on_hover(r, c)

    def _wheel(self, event) -> None:
        delta = 1 if event.delta > 0 else -1
        self._zoom_at(event, delta)

    def _wheel_linux(self, event, direction: int) -> None:
        self._zoom_at(event, direction)

    def _zoom_at(self, event, direction: int) -> None:
        factor = 1.15 if direction > 0 else 1 / 1.15
        new_zoom = float(np.clip(self.zoom * factor, 0.25, 6.0))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        # Keep cursor position stable-ish.
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.zoom = new_zoom
        self.zoom_var.set(new_zoom)
        self.zoom_label.config(text=f"{int(self.zoom * 100)}%")
        self._redraw()
        self.canvas.scan_mark(int(event.x), int(event.y))
        # Nudge scroll toward cursor cell
        if self._base_rgb is not None:
            self.canvas.xview_moveto(max(0, (x * factor - event.x) / max(1, self._base_rgb.shape[1] * self.zoom)))
            self.canvas.yview_moveto(max(0, (y * factor - event.y) / max(1, self._base_rgb.shape[0] * self.zoom)))


class Sidebar(ttk.Frame):
    def __init__(self, master: tk.Misc, app: "ForestMapApp") -> None:
        super().__init__(master, width=340)
        self.app = app
        self.pack_propagate(False)

        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        self.body = ttk.Frame(canvas)
        self.body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.body, anchor=tk.NW)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

        self.vars: dict[str, tk.Variable] = {}
        b = self.body

        ttk.Label(b, text="Генерация", font=("DejaVu Sans", 12, "bold")).pack(
            anchor=tk.W, padx=8, pady=(8, 4)
        )
        self._add_int(b, "seed", "Seed", 42)
        self._add_int(b, "width", "Ширина", 256)
        self._add_int(b, "height", "Высота", 256)
        self._add_float(b, "cell_size_m", "Ячейка, м", 8.0)
        self._add_float(b, "relief_amplitude_m", "Амплитуда рельефа, м", 58.0)

        ttk.Separator(b).pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(b, text="Шум / рельеф", font=("DejaVu Sans", 11, "bold")).pack(
            anchor=tk.W, padx=8
        )
        self._add_int(b, "noise.octaves", "Октавы", 6)
        self._add_float(b, "noise.persistence", "Persistence", 0.48)
        self._add_float(b, "noise.ridge_mix", "Ridges", 0.26)
        self._add_float(b, "noise.warp_strength", "Warp", 36.0)
        self._add_float(b, "noise.base_scale", "Масштаб шума", 0.0072)

        ttk.Separator(b).pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(b, text="Озёра / реки", font=("DejaVu Sans", 11, "bold")).pack(
            anchor=tk.W, padx=8
        )
        self._add_float(b, "hydrology.lake_min_depth", "Мин. глубина озера", 0.05)
        self._add_float(b, "hydrology.lake_max_fraction", "Макс. доля озёр", 0.07)
        self._add_int(b, "hydrology.lake_max_count", "Макс. озёр", 8)
        self._add_float(
            b, "hydrology.river_accumulation_quantile", "Квантиль рек", 0.974
        )

        ttk.Separator(b).pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(b, text="Обрывы / лес", font=("DejaVu Sans", 11, "bold")).pack(
            anchor=tk.W, padx=8
        )
        self._add_float(b, "cliffs.slope_threshold", "Порог обрыва", 0.42)
        self._add_float(b, "forest.target_density", "Плотность леса", 0.18)
        self._add_float(b, "forest.max_slope", "Макс. уклон леса", 0.55)
        self._add_float(b, "forest.moisture_weight", "Вес влажности", 0.62)

        btns = ttk.Frame(b)
        btns.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btns, text="Сгенерировать", command=app.generate).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(
            btns, text="Пересчитать гидрологию с DEM", command=app.recompute_hydro
        ).pack(fill=tk.X, pady=2)
        ttk.Button(btns, text="Обновить знаковую карту", command=app.refresh_map).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(btns, text="Сохранить PNG…", command=app.save_png).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(b).pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(b, text="Слой отображения", font=("DejaVu Sans", 11, "bold")).pack(
            anchor=tk.W, padx=8
        )
        self.layer_var = tk.StringVar(value=LAYER_NAMES[0])
        for name in LAYER_NAMES:
            ttk.Radiobutton(
                b, text=name, value=name, variable=self.layer_var, command=app.redraw_layer
            ).pack(anchor=tk.W, padx=16)

        ttk.Separator(b).pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(b, text="Кисти", font=("DejaVu Sans", 11, "bold")).pack(
            anchor=tk.W, padx=8
        )
        self.brush_var = tk.StringVar(value=BRUSH_RIVER)
        for val, label in (
            (BRUSH_RIVER, "Река"),
            (BRUSH_TRAIL, "Тропа"),
            (BRUSH_LAKE, "Озеро"),
            (BRUSH_MOUNTAIN, "Горы (поднять DEM)"),
            (BRUSH_ERASE, "Ластик (вода/тропы)"),
        ):
            ttk.Radiobutton(
                b, text=label, value=val, variable=self.brush_var
            ).pack(anchor=tk.W, padx=16)
        self._add_float(b, "brush_radius", "Радиус кисти (клетки)", 3.0)
        self._add_float(b, "brush_strength", "Сила гор", 0.06)
        ttk.Label(
            b,
            text="ЛКМ — рисовать · ПКМ/СКМ/Ctrl+ЛКМ — пан · колесо — зум",
            wraplength=300,
            font=("DejaVu Sans", 8),
        ).pack(anchor=tk.W, padx=8, pady=8)

        self.stats_var = tk.StringVar(value="Статистика: —")
        ttk.Label(b, textvariable=self.stats_var, wraplength=300).pack(
            anchor=tk.W, padx=8, pady=(0, 12)
        )

    def _row(self, parent, label: str) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
        return row

    def _add_int(self, parent, key: str, label: str, default: int) -> None:
        row = self._row(parent, label)
        var = tk.IntVar(value=default)
        self.vars[key] = var
        ttk.Entry(row, textvariable=var, width=12).pack(side=tk.RIGHT)

    def _add_float(self, parent, key: str, label: str, default: float) -> None:
        row = self._row(parent, label)
        var = tk.DoubleVar(value=default)
        self.vars[key] = var
        ttk.Entry(row, textvariable=var, width=12).pack(side=tk.RIGHT)

    def load_from_config(self, cfg: TerrainConfig) -> None:
        mapping = {
            "seed": cfg.seed,
            "width": cfg.width,
            "height": cfg.height,
            "cell_size_m": cfg.cell_size_m,
            "relief_amplitude_m": cfg.relief_amplitude_m,
            "noise.octaves": cfg.noise.octaves,
            "noise.persistence": cfg.noise.persistence,
            "noise.ridge_mix": cfg.noise.ridge_mix,
            "noise.warp_strength": cfg.noise.warp_strength,
            "noise.base_scale": cfg.noise.base_scale,
            "hydrology.lake_min_depth": cfg.hydrology.lake_min_depth,
            "hydrology.lake_max_fraction": cfg.hydrology.lake_max_fraction,
            "hydrology.lake_max_count": cfg.hydrology.lake_max_count,
            "hydrology.river_accumulation_quantile": cfg.hydrology.river_accumulation_quantile,
            "cliffs.slope_threshold": cfg.cliffs.slope_threshold,
            "forest.target_density": cfg.forest.target_density,
            "forest.max_slope": cfg.forest.max_slope,
            "forest.moisture_weight": cfg.forest.moisture_weight,
        }
        for k, v in mapping.items():
            if k in self.vars:
                self.vars[k].set(v)

    def write_to_config(self, cfg: TerrainConfig) -> TerrainConfig:
        cfg.seed = int(self.vars["seed"].get())
        cfg.width = int(self.vars["width"].get())
        cfg.height = int(self.vars["height"].get())
        cfg.cell_size_m = float(self.vars["cell_size_m"].get())
        cfg.relief_amplitude_m = float(self.vars["relief_amplitude_m"].get())
        cfg.noise.octaves = int(self.vars["noise.octaves"].get())
        cfg.noise.persistence = float(self.vars["noise.persistence"].get())
        cfg.noise.ridge_mix = float(self.vars["noise.ridge_mix"].get())
        cfg.noise.warp_strength = float(self.vars["noise.warp_strength"].get())
        cfg.noise.base_scale = float(self.vars["noise.base_scale"].get())
        cfg.hydrology.lake_min_depth = float(self.vars["hydrology.lake_min_depth"].get())
        cfg.hydrology.lake_max_fraction = float(
            self.vars["hydrology.lake_max_fraction"].get()
        )
        cfg.hydrology.lake_max_count = int(self.vars["hydrology.lake_max_count"].get())
        cfg.hydrology.river_accumulation_quantile = float(
            self.vars["hydrology.river_accumulation_quantile"].get()
        )
        cfg.cliffs.slope_threshold = float(self.vars["cliffs.slope_threshold"].get())
        cfg.forest.target_density = float(self.vars["forest.target_density"].get())
        cfg.forest.max_slope = float(self.vars["forest.max_slope"].get())
        cfg.forest.moisture_weight = float(self.vars["forest.moisture_weight"].get())
        # Faster GUI previews
        cfg.render.pixel_scale = 2
        cfg.render.show_contours = False
        return cfg


class ForestMapApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Forest Map Generator")
        root.geometry("1280x800")
        root.minsize(960, 600)

        conf_path = _default_config_path()
        self.cfg = load_config(conf_path) if conf_path.exists() else TerrainConfig()
        self.session: MapSession | None = None
        self._paint_dirty = False

        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned, width=360)
        paned.add(left, weight=4)
        paned.add(right, weight=1)

        self.map_view = MapCanvas(left, on_paint=self.on_paint)
        self.map_view.pack(fill=tk.BOTH, expand=True)
        self.sidebar = Sidebar(right, self)
        self.sidebar.pack(fill=tk.BOTH, expand=True)
        self.sidebar.load_from_config(self.cfg)

        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Сгенерировать", command=self.generate)
        file_menu.add_command(label="Сохранить PNG…", command=self.save_png)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=root.destroy)
        menubar.add_cascade(label="Файл", menu=file_menu)
        root.config(menu=menubar)

        self.root.after(100, self.generate)

    def generate(self) -> None:
        try:
            self.cfg = self.sidebar.write_to_config(self.cfg)
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            self.session = build_session(self.cfg)
            self._paint_dirty = False
            self.redraw_layer()
            self._update_stats()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка генерации", str(exc))
        finally:
            self.root.config(cursor="")

    def recompute_hydro(self) -> None:
        if not self.session:
            return
        try:
            self.cfg = self.sidebar.write_to_config(self.cfg)
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            self.session = rebuild_from_dem(self.session, self.cfg)
            self._paint_dirty = False
            self.redraw_layer()
            self._update_stats()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка", str(exc))
        finally:
            self.root.config(cursor="")

    def refresh_map(self) -> None:
        if not self.session:
            return
        refresh_symbolic(self.session)
        self._paint_dirty = False
        self.redraw_layer()
        self._update_stats()

    def redraw_layer(self) -> None:
        if not self.session:
            return
        if self._paint_dirty and self.sidebar.layer_var.get() == "Карта (знаки)":
            refresh_symbolic(self.session)
            self._paint_dirty = False
        rgb = layer_to_rgb(self.session, self.sidebar.layer_var.get())
        self.map_view.set_image(rgb, *self.session.shape)

    def on_paint(self, row: float, col: float) -> None:
        if not self.session:
            return
        radius = float(self.sidebar.vars["brush_radius"].get())
        strength = float(self.sidebar.vars["brush_strength"].get())
        kind = self.sidebar.brush_var.get()
        apply_brush(
            self.session, row=row, col=col, radius=radius, kind=kind, strength=strength
        )
        self._paint_dirty = True
        # Live update of current layer (cheap for masks / dem views).
        layer = self.sidebar.layer_var.get()
        if layer == "Карта (знаки)":
            # Defer full symbol redraw; show mask layers immediately if useful.
            refresh_symbolic(self.session)
            self._paint_dirty = False
        self.redraw_layer()
        self._update_stats()

    def save_png(self) -> None:
        if not self.session:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile="forest_map.png",
        )
        if not path:
            return
        rgb = layer_to_rgb(self.session, self.sidebar.layer_var.get())
        Image.fromarray(rgb).save(path)
        messagebox.showinfo("Сохранено", path)

    def _update_stats(self) -> None:
        if not self.session:
            return
        s = self.session.stats
        self.sidebar.stats_var.set(
            f"Озёра: {s['lake_cells']} · Реки: {s['river_cells']} · "
            f"Тропы: {s['trail_cells']} · Обрывы: {s['cliff_cells']} · "
            f"Деревья: {s['tree_count']}"
        )


def run_app() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    ForestMapApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()
