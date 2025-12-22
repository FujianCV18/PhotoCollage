# Photo Collage – AI helper notes

- **Architecture**
  - Core logic in [collage.py](collage.py): image discovery, layout algorithms (slice/BSP/justified), collage compositor, CLI entry.
  - GUI in [collage_gui.py](collage_gui.py): Tkinter app with two tabs (base collage and year-shaped collage) calling `collage` helpers and handling previews/saves.
  - PyInstaller bundling via [PhotoCollage.spec](PhotoCollage.spec); includes GUI as the entry script.

- **Layouts**
  - Slice layout (`compute_layout_slice` in [collage.py](collage.py#L1764-L1807)): builds a balanced binary slice tree trying to match canvas aspect; knobs `slice_iters`, `slice_tol`, `slice_balance`, `slice_min_share`, `max_cells`.
  - BSP layout (`compute_layout_bsp` in [collage.py](collage.py#L1895-L1982)): recursive splits with `min_cell` and `split_jitter`.
  - Justified rows (`compute_layout_justified` in [collage.py](collage.py#L1504-L1588)): target row height with jitter, then scales rows to canvas height.

- **Year collage (GUI tab “年字拼图”)**
  - Uses `compute_layout_year_three_band` to divide canvas into top/mid/bottom bands, place photos in digit holes/gaps/sides, and slice-fill remaining bands; returns `forbidden_mask` to keep digits clear ([collage.py](collage.py#L643-L918)).
  - Mask overlay applied with `_apply_year_overlay` ([collage_gui.py](collage_gui.py#L560-L618)) using solid or vertical gradient fills on digit strokes.

- **Collage rendering**
  - `build_collage` ([collage.py](collage.py#L1987-L2075)) loads images (Pillow), resizes per `fit` (`cover` crops, `contain` scales) and `pad` (`reflect`/`edge`/`background`), optional forbidden mask to avoid painting digits, optional stats for crop ratios.
  - Threaded resizing via `_effective_workers`; default `workers=0` doubles CPU count up to 32.
  - Image IO helpers: `iter_image_files` filters supported extensions; `collect_img_infos` probes EXIF-rotated size.

- **GUI workflows**
  - Preview uses smaller canvas width (`preview_res_var`) and bilinear resampling; save reruns layout at output resolution with more slice iterations and Lanczos.
  - Images capped at 500 per run; status labels surface totals and cover crop stats for previews.
  - Inputs default to `D:\STUDY\photo pack\input`; prompt user to choose existing folders to avoid failures.
  - Saving requires a fresh preview with unchanged parameters (seed/key tracking in GUI state).

- **CLI workflow**
  - Run `python collage.py --input <folder> --layout slice|bsp|justified --preset 9:16 --output collage.jpg ...`.
  - `--fit cover` (default) crops to fill cells; `--fit contain` plus `--pad reflect|edge|background` keeps full images.
  - `--stats` prints crop metrics; `--shuffle` + `--seed` for reproducibility.

- **Conventions & constraints**
  - Keep everything ASCII unless a file already uses other characters (UI strings are Chinese where present).
  - Respect randomness controls: reuse seeds for reproducible previews/saves; slice parameters tuned for balanced tiles, avoid pushing `slice_min_share > 0.49`.
  - Large images allowed by raising `Image.MAX_IMAGE_PIXELS`; still validate positive dimensions before resizing.

- **Build/distribution**
  - GUI bundle via PyInstaller: `pyinstaller PhotoCollage.spec` (uses `EXE` console=False). Pillow is the only declared dependency ([requirements.txt](requirements.txt)).

- **Testing/validation**
  - No test suite; validate by running GUI (`python collage_gui.py`) and CLI sample runs, checking crop stats and digit masks render correctly.

- **When adding features**
  - Reuse layout helpers instead of duplicating slicing logic; prefer passing explicit `CanvasSpec` and seeds.
  - Keep threading-safe operations in `build_collage` helpers (avoid sharing mutable PIL objects across threads).
  - Surface new options both in CLI args and GUI controls when user-facing.
