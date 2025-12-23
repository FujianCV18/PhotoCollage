from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image

import collage


def parse_size(s: str) -> tuple[int, int]:
    s = s.lower().replace("\u00d7", "x").strip()
    if "x" in s:
        w, h = s.split("x", 1)
        return int(w), int(h)
    raise ValueError("--size must be like 1920x1080")


def parse_hex(s: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    t = s.strip().lstrip("#")
    if len(t) != 6:
        return fallback
    try:
        r = int(t[0:2], 16)
        g = int(t[2:4], 16)
        b = int(t[4:6], 16)
        return (r, g, b)
    except Exception:
        return fallback


def make_vertical_gradient(size: tuple[int, int], top: Tuple[int, int, int], bot: Tuple[int, int, int]) -> Image.Image:
    w, h = size
    grad = Image.new("RGB", (w, h))
    px = grad.load()
    h = max(1, h)
    for y in range(h):
        t = y / float(h - 1) if h > 1 else 0.0
        r = int(round(top[0] * (1 - t) + bot[0] * t))
        g = int(round(top[1] * (1 - t) + bot[1] * t))
        b = int(round(top[2] * (1 - t) + bot[2] * t))
        for x in range(w):
            px[x, y] = (r, g, b)
    return grad


def main() -> int:
    ap = argparse.ArgumentParser("Headless generator for 'year' collage")
    ap.add_argument("--input", type=str, default=r"D:\\STUDY\\photo pack\\input")
    ap.add_argument("--output", type=str, default="collage_year.jpg")
    ap.add_argument("--size", type=str, default="1920x1080")
    ap.add_argument("--year", type=str, default="2025")
    ap.add_argument("--style", type=str, choices=["gradient"], default="gradient", help="Fixed gradient style")
    ap.add_argument("--grad-top", dest="grad_top", type=str, default="#ffd166")
    ap.add_argument("--grad-bot", dest="grad_bot", type=str, default="#ef476f")
    args = ap.parse_args()

    in_dir = Path(args.input)
    files = collage.iter_image_files(in_dir, recursive=False)
    if not files:
        raise FileNotFoundError(f"No images found in: {in_dir}")

    infos = collage.collect_img_infos(sorted(files))
    if len(infos) < 2:
        raise RuntimeError("Need at least 2 readable images")

    w, h = parse_size(args.size)
    canvas = collage.CanvasSpec(w, h)

    max_cells = min(500, len(infos))
    placed, forbidden_mask = collage.compute_layout_year_three_band(
        infos,
        canvas=canvas,
        year=str(args.year),
        max_cells=max_cells,
        slice_iters=800,
        slice_tol=0.02,
        slice_balance=0.85,
        slice_min_share=0.18,
        seed=42,
        mid_h_rel=1.0/3.0,
        digit_box_w_rel=0.82,
        digit_box_h_rel=0.86,
        digit_gap_rel=0.08,
        digit_stroke_rel=0.14,
        digit_inset_rel=0.08,
        region_block_px=max(8, int(round(min(w, h) * 0.028))),
    )

    top = parse_hex(args.grad_top, (252, 252, 252))
    bot = parse_hex(args.grad_bot, (210, 210, 210))
    out = collage.build_collage(
        placed,
        canvas=canvas,
        background=top,  # 用顶部色作为底色
        fit="cover",
        pad="reflect",
        stats=None,
        forbidden_mask=forbidden_mask,
        forbidden_fill=(top, bot),
        resample=collage.Image.Resampling.LANCZOS,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        out.save(out_path, quality=92, subsampling=1, optimize=True)
    else:
        out.save(out_path)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
