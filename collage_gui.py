from __future__ import annotations

import re
import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import ttk

from PIL import Image, ImageTk

import collage


class CollageGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Photo Collage")
        self.geometry("980x720")

        self._infos: list[collage.ImgInfo] | None = None
        self._last_image = None
        self._last_seed: int | None = None

        self.input_var = tk.StringVar(value=r"D:\STUDY\photo pack\input")
        self.output_var = tk.StringVar(value=str(Path.cwd() / "collage.jpg"))
        self.recursive_var = tk.BooleanVar(value=False)

        self.preview_res_var = tk.StringVar(value="720")
        self.output_size_var = tk.StringVar(value="9:16 (1080x1920)")

        self.avg_var = tk.DoubleVar(value=0.80)

        self.avg_label_var = tk.StringVar(value=f"{self.avg_var.get():.2f}")

        self.images_var = tk.StringVar(value="")
        self._files_total: int | None = None
        self._files_used: int | None = None

        self.status_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="")

        self._preview_tk: ImageTk.PhotoImage | None = None
        self._last_key: tuple | None = None

        self._progress: ttk.Progressbar | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        tab_base = ttk.Frame(notebook)
        tab_base.columnconfigure(1, weight=1)
        tab_base.rowconfigure(0, weight=1)
        notebook.add(tab_base, text="基础拼图")

        tab_year = YearCollageTab(notebook)
        notebook.add(tab_year, text="年字拼图")

        left = ttk.Frame(tab_base, padding=12)
        left.grid(row=0, column=0, sticky="nsw")

        right = ttk.Frame(tab_base, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        ttk.Label(left, text="输入文件夹").grid(row=0, column=0, sticky="w")
        in_row = ttk.Frame(left)
        in_row.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        in_row.columnconfigure(0, weight=1)
        ttk.Entry(in_row, textvariable=self.input_var, width=42).grid(row=0, column=0, sticky="ew")
        ttk.Button(in_row, text="选择...", command=self._choose_input).grid(row=0, column=1, padx=(6, 0))
        ttk.Checkbutton(left, text="递归扫描子文件夹", variable=self.recursive_var).grid(row=2, column=0, sticky="w", pady=(0, 10))

        ttk.Label(left, textvariable=self.images_var, foreground="#444").grid(row=3, column=0, sticky="w", pady=(0, 10))

        ttk.Label(left, text="输出文件").grid(row=4, column=0, sticky="w")
        out_row = ttk.Frame(left)
        out_row.grid(row=5, column=0, sticky="ew", pady=(4, 10))
        out_row.columnconfigure(0, weight=1)
        ttk.Entry(out_row, textvariable=self.output_var, width=42).grid(row=0, column=0, sticky="ew")
        ttk.Button(out_row, text="选择...", command=self._choose_output).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(left, text="输出尺寸(所见即所得，支持手动输入 WxH)").grid(row=6, column=0, sticky="w")
        out_size = ttk.Combobox(
            left,
            textvariable=self.output_size_var,
            values=[
                "9:16 (1080x1920)",
                "9:16 (1440x2560)",
                "9:16 (2160x3840)",
                "9:16 (4320x7680)",
                "2:3 (1080x1620)",
                "2:3 (1440x2160)",
                "2:3 (2160x3240)",
                "2:3 (5304x7952)",
            ],
            state="normal",
            width=18,
        )
        out_size.grid(row=7, column=0, sticky="w", pady=(4, 10))

        ttk.Label(left, text="预览宽度(越小越快)").grid(row=8, column=0, sticky="w")
        prev_res = ttk.Combobox(
            left,
            textvariable=self.preview_res_var,
            values=["360", "540", "720", "1080"],
            state="readonly",
            width=10,
        )
        prev_res.grid(row=9, column=0, sticky="w", pady=(4, 10))

        ttk.Label(left, text="平均度(越大越平均，步进0.01)").grid(row=10, column=0, sticky="w")
        avg = ttk.Scale(left, from_=0.0, to=1.0, variable=self.avg_var, orient="horizontal", command=self._on_avg_changed)
        avg.grid(row=11, column=0, sticky="ew")
        ttk.Label(left, textvariable=self.avg_label_var).grid(row=12, column=0, sticky="w", pady=(2, 10))

        btn_row = ttk.Frame(left)
        btn_row.grid(row=13, column=0, sticky="ew", pady=(8, 8))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        self.preview_btn = ttk.Button(btn_row, text="生成预览", command=self._on_preview)
        self.preview_btn.grid(row=0, column=0, sticky="ew")
        self.refresh_btn = ttk.Button(btn_row, text="刷新排布", command=self._on_refresh)
        self.refresh_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.save_btn = ttk.Button(left, text="保存输出(高质量)", command=self._on_save)
        self.save_btn.grid(row=14, column=0, sticky="ew")

        self._progress = ttk.Progressbar(left, mode="indeterminate")
        self._progress.grid(row=15, column=0, sticky="ew", pady=(8, 0))

        ttk.Separator(left, orient="horizontal").grid(row=16, column=0, sticky="ew", pady=(10, 10))
        ttk.Label(left, textvariable=self.status_var, foreground="#444").grid(row=17, column=0, sticky="w")
        ttk.Label(left, textvariable=self.stats_var, foreground="#444").grid(row=18, column=0, sticky="w", pady=(6, 0))

        self.preview_label = ttk.Label(right)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self._on_avg_changed(None)

    def _snap01(self, value: float) -> float:
        v = max(0.0, min(1.0, float(value)))
        return round(v / 0.01) * 0.01

    def _on_avg_changed(self, _evt) -> None:
        snapped = self._snap01(self.avg_var.get())
        if abs(snapped - float(self.avg_var.get())) > 1e-9:
            self.avg_var.set(snapped)
        self.avg_label_var.set(f"{snapped:.2f}")

    def _canvas_by_aspect(self, width: int, aspect: float) -> collage.CanvasSpec:
        w = int(width)
        if aspect <= 0:
            h = int(round(w * 16 / 9))
            return collage.CanvasSpec(w, max(1, h))
        h = int(round(w / aspect))
        return collage.CanvasSpec(w, max(1, h))

    def _preview_canvas(self) -> collage.CanvasSpec:
        w = int(self.preview_res_var.get())
        parsed = self._parse_size(self.output_size_var.get())
        if parsed is not None:
            out_w, out_h = parsed
            return self._canvas_by_aspect(w, out_w / float(out_h))
        raise ValueError("输出尺寸格式应包含 WxH，例如 7952x5304 或 9:16 (1080x1920)")

    def _parse_size(self, s: str) -> tuple[int, int] | None:
        t = s.strip().lower().replace("×", "x")
        m = re.search(r"(\d+)\s*x\s*(\d+)", t)
        if not m:
            return None
        w = int(m.group(1))
        h = int(m.group(2))
        if w <= 0 or h <= 0:
            return None
        return w, h

    def _output_canvas(self) -> collage.CanvasSpec:
        parsed = self._parse_size(self.output_size_var.get())
        if parsed is not None:
            w, h = parsed
            return collage.CanvasSpec(w, h)
        raise ValueError("输出尺寸格式应包含 WxH，例如 7952x5304 或 9:16 (1080x1920)")

    def _current_key(self) -> tuple:
        return (
            self.input_var.get(),
            bool(self.recursive_var.get()),
            self.preview_res_var.get().strip(),
            self.output_size_var.get().strip(),
            self._snap01(float(self.avg_var.get())),
        )

    def _choose_input(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.input_var.set(folder)
            self._infos = None
            self._last_seed = None
            self._last_key = None
            self._last_image = None
            self._files_total = None
            self._files_used = None
            self.images_var.set("")
            self.status_var.set("")
            self.stats_var.set("")

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg;*.jpeg"), ("PNG", "*.png"), ("All", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.preview_btn.configure(state=state)
        self.refresh_btn.configure(state=state)
        self.save_btn.configure(state=state)
        if self._progress is not None:
            if busy:
                self._progress.start(10)
            else:
                self._progress.stop()

    def _ensure_infos(self) -> list[collage.ImgInfo]:
        if self._infos is not None:
            return self._infos

        folder = Path(self.input_var.get())
        files = collage.iter_image_files(folder, recursive=bool(self.recursive_var.get()))
        if not files:
            raise FileNotFoundError(f"No images found in: {folder}")

        files = sorted(files)

        self._files_total = len(files)
        if len(files) > 500:
            rng = random.Random(0)
            rng.shuffle(files)
            files = files[:500]
        self._files_used = len(files)
        if self._files_total is not None and self._files_used is not None:
            if self._files_total > self._files_used:
                self.images_var.set(f"图片: 发现 {self._files_total} 张; 使用 {self._files_used} 张 (上限500)")
            else:
                self.images_var.set(f"图片: 使用 {self._files_used} 张")

        infos = collage.collect_img_infos(files)
        if len(infos) < 2:
            raise RuntimeError("Need at least 2 readable images")

        self._infos = infos
        return infos

    def _max_cells(self, infos: list[collage.ImgInfo]) -> int:
        return min(500, max(1, len(infos)))

    def _avg_to_params(self) -> tuple[float, float]:
        t = self._snap01(float(self.avg_var.get()))
        slice_balance = 0.15 + 0.85 * t
        slice_min_share = 0.04 + 0.20 * t
        return slice_balance, min(0.49, slice_min_share)

    def _render(self, refresh: bool) -> None:
        try:
            infos = self._ensure_infos()
            canvas = self._preview_canvas()

            seed = random.randint(0, 2**31 - 1) if (refresh or self._last_seed is None) else self._last_seed
            self._last_seed = seed

            slice_balance, slice_min_share = self._avg_to_params()

            preview_w = int(canvas.width)
            base_iters = 360 if preview_w <= 360 else (520 if preview_w <= 540 else 700)
            max_cells = self._max_cells(infos)
            scale = min(1.0, 200.0 / float(max_cells))
            slice_iters = max(120, int(base_iters * scale))

            placed = collage.compute_layout_slice(
                infos,
                canvas=canvas,
                max_cells=max_cells,
                slice_iters=slice_iters,
                slice_tol=0.02,
                slice_balance=slice_balance,
                slice_min_share=slice_min_share,
                seed=seed,
            )

            stats: dict[str, float] = {}
            img = collage.build_collage(
                placed,
                canvas=canvas,
                background=(0, 0, 0),
                fit="cover",
                pad="reflect",
                stats=stats,
                resample=collage.Image.Resampling.BILINEAR,
            )

            areas = [p.w * p.h for p in placed if p.w > 0 and p.h > 0]
            if areas:
                n = len(areas)
                avg_a = sum(areas) / n
                min_a = min(areas)
                max_a = max(areas)
                var = sum((a - avg_a) ** 2 for a in areas) / n
                cv = (var**0.5 / avg_a) if avg_a else 0.0
                ratio = (max_a / min_a) if min_a else 0.0
                tiles_line = f"tiles: n={n}; max/min={ratio:.2f}; cv={cv:.2f}"
            else:
                tiles_line = "tiles: n=0"

            total = int(stats.get("cover_total", 0.0))
            cropped = int(stats.get("cover_cropped", 0.0))
            pct = float(stats.get("cover_crop_area_avg", 0.0)) * 100.0
            cover_line = f"cover: cropped {cropped}/{total}; avg crop {pct:.2f}%"

            self._last_image = img

            preview = img.copy()
            preview.thumbnail((720, 720))
            self._preview_tk = ImageTk.PhotoImage(preview)

            def update_ui() -> None:
                self.preview_label.configure(image=self._preview_tk)
                out_canvas = self._output_canvas()
                self.status_var.set(
                    f"seed={seed}  avg={float(self.avg_label_var.get()):.2f}  preview={canvas.width}x{canvas.height}  output={out_canvas.width}x{out_canvas.height}  imgs={len(placed)}"
                )
                self.stats_var.set(f"{tiles_line}    {cover_line}")
                self._last_key = self._current_key()

            self.after(0, update_ui)
        except Exception as e:
            msg = str(e)
            self.after(0, lambda m=msg: messagebox.showerror("Error", m))
        finally:
            self.after(0, lambda: self._set_busy(False))

    def _run_render(self, refresh: bool) -> None:
        self._set_busy(True)
        self.status_var.set("正在生成预览...")
        self.stats_var.set("")
        th = threading.Thread(target=self._render, args=(refresh,), daemon=True)
        th.start()

    def _on_preview(self) -> None:
        self._last_seed = None
        self._run_render(refresh=True)

    def _on_refresh(self) -> None:
        self._run_render(refresh=True)

    def _on_save(self) -> None:
        if self._last_image is None:
            messagebox.showinfo("提示", "请先生成预览")
            return

        if self._last_seed is None or self._last_key is None or self._last_key != self._current_key():
            messagebox.showinfo("提示", "参数已修改，请先生成预览")
            return

        def save_worker() -> None:
            try:
                self.after(0, lambda: self._set_busy(True))
                self.after(0, lambda: self.status_var.set("正在保存高质量输出..."))

                infos = self._ensure_infos()
                canvas = self._output_canvas()
                slice_balance, slice_min_share = self._avg_to_params()
                seed = int(self._last_seed)

                max_cells = self._max_cells(infos)
                scale = min(1.0, 300.0 / float(max_cells))
                slice_iters = max(600, int(1200 * scale))

                placed = collage.compute_layout_slice(
                    infos,
                    canvas=canvas,
                    max_cells=max_cells,
                    slice_iters=slice_iters,
                    slice_tol=0.02,
                    slice_balance=slice_balance,
                    slice_min_share=slice_min_share,
                    seed=seed,
                )

                img = collage.build_collage(
                    placed,
                    canvas=canvas,
                    background=(0, 0, 0),
                    fit="cover",
                    pad="reflect",
                    stats=None,
                    resample=collage.Image.Resampling.LANCZOS,
                )

                out_path = Path(self.output_var.get())
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ext = out_path.suffix.lower()
                if ext in {".jpg", ".jpeg"}:
                    img.save(out_path, quality=92, subsampling=1, optimize=True)
                else:
                    img.save(out_path)

                self.after(0, lambda: messagebox.showinfo("完成", f"已保存: {out_path}"))
            except Exception as e:
                msg = str(e)
                self.after(0, lambda m=msg: messagebox.showerror("Error", m))
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=save_worker, daemon=True).start()
        return


def main() -> int:
    app = CollageGui()
    app.mainloop()
    return 0
class YearCollageTab(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)

        self._infos: list[collage.ImgInfo] | None = None
        self._last_seed: int | None = None
        self._last_key: tuple | None = None

        self.input_var = tk.StringVar(value=r"D:\STUDY\photo pack\input")
        self.output_var = tk.StringVar(value=str(Path.cwd() / "collage_year.jpg"))
        self.recursive_var = tk.BooleanVar(value=False)

        self.preview_res_var = tk.StringVar(value="720")
        self.output_size_var = tk.StringVar(value="16:9 (1920x1080)")

        self.year_var = tk.StringVar(value="2025")

        self.year_style_var = tk.StringVar(value="渐变")

        self.avg_var = tk.DoubleVar(value=0.80)
        self.avg_label_var = tk.StringVar(value=f"{self.avg_var.get():.2f}")

        self.images_var = tk.StringVar(value="")
        self._files_total: int | None = None
        self._files_used: int | None = None

        self.status_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="")

        self._preview_tk: ImageTk.PhotoImage | None = None
        self._progress: ttk.Progressbar | None = None
        self._last_preview_img = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="nsw")

        right = ttk.Frame(self, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        ttk.Label(left, text="输入文件夹").grid(row=0, column=0, sticky="w")
        in_row = ttk.Frame(left)
        in_row.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        in_row.columnconfigure(0, weight=1)
        ttk.Entry(in_row, textvariable=self.input_var, width=42).grid(row=0, column=0, sticky="ew")
        ttk.Button(in_row, text="选择...", command=self._choose_input).grid(row=0, column=1, padx=(6, 0))
        ttk.Checkbutton(left, text="递归扫描子文件夹", variable=self.recursive_var).grid(row=2, column=0, sticky="w", pady=(0, 10))

        ttk.Label(left, textvariable=self.images_var, foreground="#444").grid(row=3, column=0, sticky="w", pady=(0, 10))

        ttk.Label(left, text="输出文件").grid(row=4, column=0, sticky="w")
        out_row = ttk.Frame(left)
        out_row.grid(row=5, column=0, sticky="ew", pady=(4, 10))
        out_row.columnconfigure(0, weight=1)
        ttk.Entry(out_row, textvariable=self.output_var, width=42).grid(row=0, column=0, sticky="ew")
        ttk.Button(out_row, text="选择...", command=self._choose_output).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(left, text="输出尺寸(16:9，支持手动输入 WxH)").grid(row=6, column=0, sticky="w")
        out_size = ttk.Combobox(
            left,
            textvariable=self.output_size_var,
            values=[
                "16:9 (1280x720)",
                "16:9 (1920x1080)",
                "16:9 (2560x1440)",
                "16:9 (3840x2160)",
                "16:9 (6000x3375)",
            ],
            state="normal",
            width=18,
        )
        out_size.grid(row=7, column=0, sticky="w", pady=(4, 10))

        ttk.Label(left, text="预览宽度(越小越快)").grid(row=8, column=0, sticky="w")
        prev_res = ttk.Combobox(
            left,
            textvariable=self.preview_res_var,
            values=["360", "540", "720", "1080"],
            state="readonly",
            width=10,
        )
        prev_res.grid(row=9, column=0, sticky="w", pady=(4, 10))

        ttk.Label(left, text="文字(数字，例如 2025)").grid(row=10, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.year_var, width=14).grid(row=11, column=0, sticky="w", pady=(4, 10))

        ttk.Label(left, text="文字样式").grid(row=12, column=0, sticky="w")
        year_style = ttk.Combobox(left, textvariable=self.year_style_var, values=["纯色", "渐变"], state="readonly", width=10)
        year_style.grid(row=13, column=0, sticky="w", pady=(4, 10))

        ttk.Label(left, text="平均度(越大越平均，步进0.01)").grid(row=14, column=0, sticky="w")
        avg = ttk.Scale(left, from_=0.0, to=1.0, variable=self.avg_var, orient="horizontal", command=self._on_avg_changed)
        avg.grid(row=15, column=0, sticky="ew")
        ttk.Label(left, textvariable=self.avg_label_var).grid(row=16, column=0, sticky="w", pady=(2, 10))

        btn_row = ttk.Frame(left)
        btn_row.grid(row=17, column=0, sticky="ew", pady=(8, 8))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        self.preview_btn = ttk.Button(btn_row, text="生成预览", command=self._on_preview)
        self.preview_btn.grid(row=0, column=0, sticky="ew")
        self.refresh_btn = ttk.Button(btn_row, text="刷新排布", command=self._on_refresh)
        self.refresh_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.save_btn = ttk.Button(left, text="保存输出(高质量)", command=self._on_save)
        self.save_btn.grid(row=18, column=0, sticky="ew")

        self._progress = ttk.Progressbar(left, mode="indeterminate")
        self._progress.grid(row=19, column=0, sticky="ew", pady=(8, 0))

        ttk.Separator(left, orient="horizontal").grid(row=20, column=0, sticky="ew", pady=(10, 10))
        ttk.Label(left, textvariable=self.status_var, foreground="#444").grid(row=21, column=0, sticky="w")
        ttk.Label(left, textvariable=self.stats_var, foreground="#444").grid(row=22, column=0, sticky="w", pady=(6, 0))

        self.preview_label = ttk.Label(right)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self._on_avg_changed(None)

    def _snap01(self, value: float) -> float:
        v = max(0.0, min(1.0, float(value)))
        return round(v / 0.01) * 0.01

    def _on_avg_changed(self, _evt) -> None:
        snapped = self._snap01(self.avg_var.get())
        if abs(snapped - float(self.avg_var.get())) > 1e-9:
            self.avg_var.set(snapped)
        self.avg_label_var.set(f"{snapped:.2f}")

    def _parse_size(self, s: str) -> tuple[int, int] | None:
        t = s.strip().lower().replace("×", "x")
        m = re.search(r"(\d+)\s*x\s*(\d+)", t)
        if not m:
            return None
        w = int(m.group(1))
        h = int(m.group(2))
        if w <= 0 or h <= 0:
            return None
        return w, h

    def _output_canvas(self) -> collage.CanvasSpec:
        parsed = self._parse_size(self.output_size_var.get())
        if parsed is not None:
            w, h = parsed
            return collage.CanvasSpec(w, h)
        raise ValueError("输出尺寸格式应包含 WxH，例如 6000x4000")

    def _preview_canvas(self) -> collage.CanvasSpec:
        w = int(self.preview_res_var.get())
        out = self._output_canvas()
        h = int(round(w * out.height / float(out.width))) if out.width else int(round(w * 2 / 3))
        return collage.CanvasSpec(w, max(1, h))

    def _current_key(self) -> tuple:
        return (
            self.input_var.get(),
            bool(self.recursive_var.get()),
            self.preview_res_var.get().strip(),
            self.output_size_var.get().strip(),
            self.year_var.get().strip(),
            self.year_style_var.get().strip(),
            self._snap01(float(self.avg_var.get())),
        )

    def _apply_year_overlay(self, img: Image.Image, mask: collage.Image.Image) -> Image.Image:
        if mask is None:
            return img
        m = mask
        if m.mode != "L":
            m = m.convert("L")
        if m.size != img.size:
            m = m.resize(img.size, resample=collage.Image.Resampling.NEAREST)

        style = self.year_style_var.get().strip()
        if style == "纯色":
            layer = Image.new("RGB", img.size, color=(245, 245, 245))
            out = img.copy()
            out.paste(layer, (0, 0), mask=m)
            return out

        # 渐变(竖向)：上亮下稍暗
        top = (252, 252, 252)
        bot = (210, 210, 210)
        grad = Image.new("RGB", img.size)
        px = grad.load()
        h = max(1, img.size[1])
        for y in range(h):
            t = y / float(h - 1) if h > 1 else 0.0
            r = int(round(top[0] * (1 - t) + bot[0] * t))
            g = int(round(top[1] * (1 - t) + bot[1] * t))
            b = int(round(top[2] * (1 - t) + bot[2] * t))
            for x in range(img.size[0]):
                px[x, y] = (r, g, b)
        out = img.copy()
        out.paste(grad, (0, 0), mask=m)
        return out

    def _choose_input(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.input_var.set(folder)
            self._infos = None
            self._last_seed = None
            self._last_key = None
            self._files_total = None
            self._files_used = None
            self.images_var.set("")
            self.status_var.set("")
            self.stats_var.set("")
            self._last_preview_img = None

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg;*.jpeg"), ("PNG", "*.png"), ("All", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.preview_btn.configure(state=state)
        self.refresh_btn.configure(state=state)
        self.save_btn.configure(state=state)
        if self._progress is not None:
            if busy:
                self._progress.start(10)
            else:
                self._progress.stop()

    def _ensure_infos(self) -> list[collage.ImgInfo]:
        if self._infos is not None:
            return self._infos

        folder = Path(self.input_var.get())
        files = collage.iter_image_files(folder, recursive=bool(self.recursive_var.get()))
        if not files:
            raise FileNotFoundError(f"No images found in: {folder}")

        files = sorted(files)
        self._files_total = len(files)
        if len(files) > 500:
            rng = random.Random(0)
            rng.shuffle(files)
            files = files[:500]
        self._files_used = len(files)
        if self._files_total is not None and self._files_used is not None:
            if self._files_total > self._files_used:
                self.images_var.set(f"图片: 发现 {self._files_total} 张; 使用 {self._files_used} 张 (上限500)")
            else:
                self.images_var.set(f"图片: 使用 {self._files_used} 张")

        infos = collage.collect_img_infos(files)
        if len(infos) < 2:
            raise RuntimeError("Need at least 2 readable images")

        self._infos = infos
        return infos

    def _max_cells(self, infos: list[collage.ImgInfo]) -> int:
        return min(500, max(1, len(infos)))

    def _avg_to_params(self) -> tuple[float, float]:
        t = self._snap01(float(self.avg_var.get()))
        slice_balance = 0.15 + 0.85 * t
        slice_min_share = 0.04 + 0.20 * t
        return slice_balance, min(0.49, slice_min_share)

    def _render(self, refresh: bool, high_quality: bool) -> None:
        try:
            infos = self._ensure_infos()
            canvas = self._output_canvas() if high_quality else self._preview_canvas()

            seed = random.randint(0, 2**31 - 1) if (refresh or self._last_seed is None) else self._last_seed
            self._last_seed = seed

            slice_balance, slice_min_share = self._avg_to_params()
            max_cells = self._max_cells(infos)

            preview_w = int(canvas.width)
            base_iters = 360 if preview_w <= 360 else (520 if preview_w <= 540 else 700)
            scale = min(1.0, (300.0 if high_quality else 200.0) / float(max_cells))
            slice_iters = max(600 if high_quality else 120, int((1200 if high_quality else base_iters) * scale))

            year = self.year_var.get().strip()
            if not year or re.search(r"\D", year):
                raise ValueError("文字必须是纯数字，例如 2025")

            placed, forbidden_mask = collage.compute_layout_year_three_band(
                infos,
                canvas=canvas,
                year=year,
                max_cells=max_cells,
                slice_iters=slice_iters,
                slice_tol=0.02,
                slice_balance=slice_balance,
                slice_min_share=slice_min_share,
                seed=seed,
                mid_h_rel=1.0 / 3.0,
                digit_box_w_rel=0.82,
                digit_box_h_rel=0.86,
                digit_gap_rel=0.08,
                digit_stroke_rel=0.14,
                digit_inset_rel=0.08,
                region_block_px=max(8, int(round(min(canvas.width, canvas.height) * 0.028))),
            )

            stats: dict[str, float] | None = {} if not high_quality else None
            img = collage.build_collage(
                placed,
                canvas=canvas,
                background=(0, 0, 0),
                fit="cover",
                pad="reflect",
                stats=stats,
                forbidden_mask=forbidden_mask,
                resample=collage.Image.Resampling.LANCZOS if high_quality else collage.Image.Resampling.BILINEAR,
            )

            img = self._apply_year_overlay(img, forbidden_mask)

            if not high_quality:
                self._last_preview_img = img

                preview = img.copy()
                preview.thumbnail((720, 720))
                self._preview_tk = ImageTk.PhotoImage(preview)

                def update_ui() -> None:
                    self.preview_label.configure(image=self._preview_tk)
                    out_canvas = self._output_canvas()
                    self.status_var.set(
                        f"seed={seed}  avg={float(self.avg_label_var.get()):.2f}  preview={canvas.width}x{canvas.height}  output={out_canvas.width}x{out_canvas.height}  imgs={len(placed)}"
                    )
                    if stats is not None:
                        total = int(stats.get("cover_total", 0.0))
                        cropped = int(stats.get("cover_cropped", 0.0))
                        pct = float(stats.get("cover_crop_area_avg", 0.0)) * 100.0
                        self.stats_var.set(f"cover: cropped {cropped}/{total}; avg crop {pct:.2f}%")
                    else:
                        self.stats_var.set("")
                    self._last_key = self._current_key()

                self.after(0, update_ui)
            else:
                out_path = Path(self.output_var.get())
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ext = out_path.suffix.lower()
                if ext in {".jpg", ".jpeg"}:
                    img.save(out_path, quality=92, subsampling=1, optimize=True)
                else:
                    img.save(out_path)
                self.after(0, lambda: messagebox.showinfo("完成", f"已保存: {out_path}"))

        except Exception as e:
            msg = str(e)
            self.after(0, lambda m=msg: messagebox.showerror("Error", m))
        finally:
            self.after(0, lambda: self._set_busy(False))

    def _run_render(self, refresh: bool, high_quality: bool) -> None:
        self._set_busy(True)
        self.status_var.set("正在生成预览..." if not high_quality else "正在保存高质量输出...")
        self.stats_var.set("")
        th = threading.Thread(target=self._render, args=(refresh, high_quality), daemon=True)
        th.start()

    def _on_preview(self) -> None:
        self._last_seed = None
        self._run_render(refresh=True, high_quality=False)

    def _on_refresh(self) -> None:
        self._run_render(refresh=True, high_quality=False)

    def _on_save(self) -> None:
        if self._last_seed is None or self._last_key is None or self._last_key != self._current_key() or self._last_preview_img is None:
            messagebox.showinfo("提示", "参数已修改，请先生成预览")
            return
        self._run_render(refresh=False, high_quality=True)


if __name__ == "__main__":
    raise SystemExit(main())
