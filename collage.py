from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# allow large images; keep a very high limit to avoid PIL warning spam
Image.MAX_IMAGE_PIXELS = max(int(getattr(Image, "MAX_IMAGE_PIXELS", 0) or 0), 250_000_000)


def _effective_workers(workers: int) -> int:
    if workers <= 0:
        cpu = os.cpu_count() or 4
        return min(32, max(1, cpu * 2))
    return max(1, int(workers))


@dataclass(frozen=True)
class CanvasSpec:
    width: int
    height: int


@dataclass
class SliceNode:
    aspect: float
    left: "SliceNode | None" = None
    right: "SliceNode | None" = None
    split: str | None = None
    img: "ImgInfo | None" = None
    leaves: int = 1


def parse_canvas(preset: str | None, size: str | None) -> CanvasSpec:
    if size:
        parts = size.lower().replace("x", ":").split(":")
        if len(parts) != 2:
            raise ValueError("--size must be like 1080x1920")
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError("--size must be positive")
        return CanvasSpec(w, h)

    if not preset:
        preset = "9:16"

    preset = preset.strip()

    presets: dict[str, CanvasSpec] = {
        "9:16": CanvasSpec(1080, 1920),
        "2:3": CanvasSpec(1080, 1620),
        "1080x1920": CanvasSpec(1080, 1920),
        "1080x1620": CanvasSpec(1080, 1620),
        "2160x3840": CanvasSpec(2160, 3840),
        "2160x3240": CanvasSpec(2160, 3240),
    }

    if preset in presets:
        return presets[preset]

    if ":" in preset:
        a, b = preset.split(":", 1)
        ra, rb = float(a), float(b)
        if ra <= 0 or rb <= 0:
            raise ValueError("ratio must be positive")
        base_w = 1080
        base_h = int(round(base_w * (rb / ra)))
        return CanvasSpec(base_w, base_h)

    raise ValueError("unknown preset; use 9:16 / 2:3 or --size")


def iter_image_files(folder: Path, recursive: bool) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"input folder not found: {folder}")

    files: List[Path] = []
    if recursive:
        walker: Iterable[Path] = folder.rglob("*")
    else:
        walker = folder.glob("*")

    for p in walker:
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)

    return files


def open_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def safe_resize(
    img: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    w, h = size
    if w <= 0 or h <= 0:
        raise ValueError("resize size must be positive")
    return img.resize((w, h), resample=resample)


def parse_rgb(value: str) -> Tuple[int, int, int]:
    s = value.strip().lower()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError("--background must be RRGGBB or #RRGGBB")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


@dataclass
class ImgInfo:
    path: Path
    w: int
    h: int

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 1.0


def rect_intersects(a: Rect, b: Rect) -> bool:
    return not (
        a.x + a.w <= b.x
        or b.x + b.w <= a.x
        or a.y + a.h <= b.y
        or b.y + b.h <= a.y
    )


def rect_intersection(a: Rect, b: Rect) -> Rect | None:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    return Rect(x0, y0, w, h)


def subtract_rect(base: Rect, cut: Rect) -> list[Rect]:
    inter = rect_intersection(base, cut)
    if inter is None:
        return [base]

    out: list[Rect] = []
    top_h = inter.y - base.y
    if top_h > 0:
        out.append(Rect(base.x, base.y, base.w, top_h))

    bottom_y = inter.y + inter.h
    bottom_h = (base.y + base.h) - bottom_y
    if bottom_h > 0:
        out.append(Rect(base.x, bottom_y, base.w, bottom_h))

    left_w = inter.x - base.x
    if left_w > 0:
        out.append(Rect(base.x, inter.y, left_w, inter.h))

    right_x = inter.x + inter.w
    right_w = (base.x + base.w) - right_x
    if right_w > 0:
        out.append(Rect(right_x, inter.y, right_w, inter.h))

    return [r for r in out if r.w > 0 and r.h > 0]


def subtract_rects(base: Rect, cuts: Sequence[Rect]) -> list[Rect]:
    rects = [base]
    for c in cuts:
        next_rects: list[Rect] = []
        for r in rects:
            next_rects.extend(subtract_rect(r, c))
        rects = next_rects
    return [r for r in rects if r.w > 0 and r.h > 0]


@dataclass
class PlacedImg:
    path: Path
    x: int
    y: int
    w: int
    h: int
    rotate: int = 0  # 0=不旋转, 90=顺时针90度, -90=逆时针90度


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 1.0


def pad_repeat_edges(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    if w == target_w and h == target_h:
        return img

    out = Image.new(img.mode, (target_w, target_h))
    ox = (target_w - w) // 2
    oy = (target_h - h) // 2
    out.paste(img, (ox, oy))

    if ox > 0:
        strip = img.crop((0, 0, 1, h)).resize((ox, h), resample=Image.Resampling.NEAREST)
        out.paste(strip, (0, oy))

    right_pad = target_w - (ox + w)
    if right_pad > 0:
        strip = img.crop((w - 1, 0, w, h)).resize((right_pad, h), resample=Image.Resampling.NEAREST)
        out.paste(strip, (ox + w, oy))

    if oy > 0:
        strip = out.crop((0, oy, target_w, oy + 1)).resize((target_w, oy), resample=Image.Resampling.NEAREST)
        out.paste(strip, (0, 0))

    bottom_pad = target_h - (oy + h)
    if bottom_pad > 0:
        strip = out.crop((0, oy + h - 1, target_w, oy + h)).resize(
            (target_w, bottom_pad), resample=Image.Resampling.NEAREST
        )
        out.paste(strip, (0, oy + h))

    return out


def _digit_polylines(d: str) -> list[list[tuple[float, float]]]:
    m = 0.10
    if d == "0":
        return [[(m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m), (m, m)]]
    if d == "1":
        return [[(0.55, m), (0.55, 1 - m)], [(0.40, 1 - m), (0.70, 1 - m)]]
    if d == "2":
        return [[(m, m), (1 - m, m), (1 - m, 0.50), (m, 0.50), (m, 1 - m), (1 - m, 1 - m)]]
    if d == "3":
        return [[(m, m), (1 - m, m), (1 - m, 0.50), (m, 0.50), (1 - m, 0.50), (1 - m, 1 - m), (m, 1 - m)]]
    if d == "4":
        return [[(m, m), (m, 0.58)], [(m, 0.58), (1 - m, 0.58)], [(1 - m, m), (1 - m, 1 - m)]]
    if d == "5":
        return [[(1 - m, m), (m, m), (m, 0.50), (1 - m, 0.50), (1 - m, 1 - m), (m, 1 - m)]]
    if d == "6":
        return [[(1 - m, m), (m, m), (m, 1 - m), (1 - m, 1 - m), (1 - m, 0.50), (m, 0.50)]]
    if d == "7":
        return [[(m, m), (1 - m, m), (1 - m, 1 - m)]]
    if d == "8":
        return [
            [(m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m), (m, m)],
            [(m, 0.50), (1 - m, 0.50)],
        ]
    if d == "9":
        return [[(1 - m, 0.50), (m, 0.50), (m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m)]]
    raise ValueError(f"unsupported digit: {d}")


def text_digit_boxes(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float,
    box_h_rel: float,
    gap_rel: float,
) -> list[Rect]:
    t = text.strip()
    if not t:
        return []
    if any(ch not in "0123456789" for ch in t):
        raise ValueError("text must be digits only")

    box_w = max(1, int(round(canvas.width * float(box_w_rel))))
    box_h = max(1, int(round(canvas.height * float(box_h_rel))))
    x0 = (canvas.width - box_w) // 2
    y0 = (canvas.height - box_h) // 2

    n = len(t)
    gap = int(round((box_w / max(1, n)) * float(gap_rel)))
    total_gap = gap * (n - 1)
    digit_w = max(1, (box_w - total_gap) // n)
    digit_h = box_h

    rects: list[Rect] = []
    for i in range(n):
        dx = x0 + i * (digit_w + gap)
        rects.append(Rect(dx, y0, digit_w, digit_h))
    return rects


def text_digit_gap_rects(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float,
    box_h_rel: float,
    gap_rel: float,
    full_height: bool = True,
) -> list[Rect]:
    """计算数字之间的间隙区域，用于放置竖图
    
    Args:
        full_height: 如果为True，间隙区域覆盖整个canvas高度；否则只覆盖数字高度
    """
    t = text.strip()
    if not t or len(t) < 2:
        return []
    if any(ch not in "0123456789" for ch in t):
        raise ValueError("text must be digits only")

    box_w = max(1, int(round(canvas.width * float(box_w_rel))))
    box_h = max(1, int(round(canvas.height * float(box_h_rel))))
    x0 = (canvas.width - box_w) // 2
    y0 = (canvas.height - box_h) // 2

    n = len(t)
    gap = int(round((box_w / max(1, n)) * float(gap_rel)))
    total_gap = gap * (n - 1)
    digit_w = max(1, (box_w - total_gap) // n)

    # 计算数字之间的间隙矩形
    gap_rects: list[Rect] = []
    for i in range(n - 1):
        # 第i个数字的右边界
        digit_right = x0 + i * (digit_w + gap) + digit_w
        # 间隙区域 - 覆盖整个高度或只覆盖数字高度
        if full_height:
            gap_rects.append(Rect(digit_right, 0, gap, canvas.height))
        else:
            gap_rects.append(Rect(digit_right, y0, gap, box_h))
    return gap_rects


def text_digit_gap_slot_rects(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float,
    box_h_rel: float,
    gap_rel: float,
    stroke_rel: float = 0.14,
    imgs_per_gap: int = 2,
) -> list[Rect]:
    """计算数字之间的间隙区域中每张图片的矩形位置
    
    每个间隙被分成 imgs_per_gap 个槽位，每个槽位放一张竖图
    间隙区域扩展到紧贴数字笔画边缘
    """
    t = text.strip()
    if not t or len(t) < 2:
        return []
    if any(ch not in "0123456789" for ch in t):
        raise ValueError("text must be digits only")

    box_w = max(1, int(round(canvas.width * float(box_w_rel))))
    box_h = max(1, int(round(canvas.height * float(box_h_rel))))
    x0 = (canvas.width - box_w) // 2

    n = len(t)
    gap = int(round((box_w / max(1, n)) * float(gap_rel)))
    total_gap = gap * (n - 1)
    digit_w = max(1, (box_w - total_gap) // n)
    
    # 计算笔画宽度（数字内部的笔画）
    stroke_w = int(round(digit_w * float(stroke_rel)))

    slot_rects: list[Rect] = []
    imgs_per_gap = max(1, imgs_per_gap)
    slot_h = canvas.height // imgs_per_gap
    
    for i in range(n - 1):
        # 第i个数字的右边界
        digit_right = x0 + i * (digit_w + gap) + digit_w
        # 下一个数字的左边界
        next_digit_left = digit_right + gap
        
        # 扩展间隙区域：从当前数字笔画内边缘到下一个数字笔画内边缘
        # 向左扩展到当前数字的笔画内侧，向右扩展到下一个数字的笔画内侧
        expanded_left = digit_right - stroke_w
        expanded_right = next_digit_left + stroke_w
        expanded_w = expanded_right - expanded_left
        
        # 在这个扩展间隙中创建多个槽位
        for j in range(imgs_per_gap):
            slot_y = j * slot_h
            # 最后一个槽位取剩余高度，避免舍入误差
            if j == imgs_per_gap - 1:
                actual_h = canvas.height - slot_y
            else:
                actual_h = slot_h
            slot_rects.append(Rect(expanded_left, slot_y, expanded_w, actual_h))
    
    return slot_rects


def text_side_gap_rects(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float,
    box_h_rel: float,
    gap_rel: float,
) -> list[Rect]:
    """计算数字区域两侧的空白区域，用于放置竖图"""
    t = text.strip()
    if not t:
        return []

    box_w = max(1, int(round(canvas.width * float(box_w_rel))))
    box_h = max(1, int(round(canvas.height * float(box_h_rel))))
    x0 = (canvas.width - box_w) // 2
    y0 = (canvas.height - box_h) // 2

    side_rects: list[Rect] = []
    # 左侧空白 - 覆盖整个高度
    if x0 > 0:
        side_rects.append(Rect(0, 0, x0, canvas.height))
    # 右侧空白 - 覆盖整个高度
    right_x = x0 + box_w
    if right_x < canvas.width:
        side_rects.append(Rect(right_x, 0, canvas.width - right_x, canvas.height))
    return side_rects


def text_side_gap_slot_rects(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float,
    box_h_rel: float,
    gap_rel: float,
    stroke_rel: float = 0.14,
    imgs_per_side: int = 2,
) -> list[Rect]:
    """计算数字区域两侧的空白区域中每张图片的矩形位置
    
    每侧被分成 imgs_per_side 个槽位，每个槽位放一张竖图
    侧边区域扩展到紧贴数字笔画边缘
    """
    t = text.strip()
    if not t:
        return []

    box_w = max(1, int(round(canvas.width * float(box_w_rel))))
    x0 = (canvas.width - box_w) // 2
    
    # 计算数字宽度和笔画宽度
    n = len(t)
    gap = int(round((box_w / max(1, n)) * float(gap_rel)))
    total_gap = gap * (n - 1)
    digit_w = max(1, (box_w - total_gap) // n)
    stroke_w = int(round(digit_w * float(stroke_rel)))

    slot_rects: list[Rect] = []
    imgs_per_side = max(1, imgs_per_side)
    slot_h = canvas.height // imgs_per_side
    
    # 左侧空白 - 扩展到第一个数字的笔画内侧
    left_w = x0 + stroke_w  # 扩展到第一个数字的笔画内侧
    if left_w > 0:
        for j in range(imgs_per_side):
            slot_y = j * slot_h
            if j == imgs_per_side - 1:
                actual_h = canvas.height - slot_y
            else:
                actual_h = slot_h
            slot_rects.append(Rect(0, slot_y, left_w, actual_h))
    
    # 右侧空白 - 扩展到最后一个数字的笔画内侧
    right_x = x0 + box_w - stroke_w  # 从最后一个数字笔画内侧开始
    if right_x < canvas.width:
        right_w = canvas.width - right_x
        for j in range(imgs_per_side):
            slot_y = j * slot_h
            if j == imgs_per_side - 1:
                actual_h = canvas.height - slot_y
            else:
                actual_h = slot_h
            slot_rects.append(Rect(right_x, slot_y, right_w, actual_h))
    
    return slot_rects


def expand_rect(r: Rect, pad: int, canvas: CanvasSpec) -> Rect:
    x0 = max(0, r.x - pad)
    y0 = max(0, r.y - pad)
    x1 = min(canvas.width, r.x + r.w + pad)
    y1 = min(canvas.height, r.y + r.h + pad)
    return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def union_masks(masks: Sequence[Image.Image], size: tuple[int, int]) -> Image.Image:
    out = Image.new("L", size, 0)
    for m in masks:
        if m.size != size:
            m = m.resize(size, resample=Image.Resampling.NEAREST)
        if m.mode != "L":
            m = m.convert("L")
        out = ImageChops.lighter(out, m)
    return out


def erode_mask_horizontal(mask: Image.Image, pixels: int) -> Image.Image:
    """横向腐蚀mask，只削减左右厚度，保持上下厚度不变
    
    Args:
        mask: L模式的mask图像，白色(255)为笔画区域
        pixels: 腐蚀的像素数（每边削减的量）
    
    Returns:
        腐蚀后的mask
    """
    if pixels <= 0:
        return mask
    if mask.mode != "L":
        mask = mask.convert("L")
    
    w, h = mask.size
    result = mask.copy()
    
    # 使用多次1像素腐蚀来实现更精确的横向腐蚀
    for _ in range(pixels):
        # 创建左移和右移版本
        left_shift = Image.new("L", (w, h), 0)
        right_shift = Image.new("L", (w, h), 0)
        
        # 左移1像素（相当于检查右边是否有像素）
        left_shift.paste(result.crop((1, 0, w, h)), (0, 0))
        # 右移1像素（相当于检查左边是否有像素）
        right_shift.paste(result.crop((0, 0, w - 1, h)), (1, 0))
        
        # 只保留左右都有像素的地方（横向腐蚀）
        result = ImageChops.darker(result, left_shift)
        result = ImageChops.darker(result, right_shift)
    
    return result


def mask_to_rects(mask: Image.Image, block: int, inflate: int = 0) -> list[Rect]:
    if mask.mode != "L":
        mask = mask.convert("L")
    w, h = mask.size
    block = max(1, int(block))
    cols = max(1, (w + block - 1) // block)
    rows = max(1, (h + block - 1) // block)

    grid: list[list[bool]] = [[False] * cols for _ in range(rows)]
    for ry in range(rows):
        y0 = ry * block
        y1 = min(h, y0 + block)
        for cx in range(cols):
            x0 = cx * block
            x1 = min(w, x0 + block)
            if mask.crop((x0, y0, x1, y1)).getbbox() is not None:
                grid[ry][cx] = True

    rects: list[Rect] = []
    active: dict[tuple[int, int], Rect] = {}

    for ry in range(rows):
        row_runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for cx in range(cols):
            if grid[ry][cx]:
                if run_start is None:
                    run_start = cx
            else:
                if run_start is not None:
                    row_runs.append((run_start, cx - 1))
                    run_start = None
        if run_start is not None:
            row_runs.append((run_start, cols - 1))

        next_active: dict[tuple[int, int], Rect] = {}
        for c0, c1 in row_runs:
            key = (c0, c1)
            if key in active:
                prev = active[key]
                next_active[key] = Rect(prev.x, prev.y, prev.w, prev.h + block)
            else:
                x0 = c0 * block
                x1 = min(w, (c1 + 1) * block)
                y0 = ry * block
                y1 = min(h, y0 + block)
                next_active[key] = Rect(x0, y0, x1 - x0, y1 - y0)

        for key, r in active.items():
            if key not in next_active:
                rects.append(r)
        active = next_active

    rects.extend(active.values())

    if inflate > 0:
        rects = [expand_rect(r, pad=int(inflate), canvas=CanvasSpec(w, h)) for r in rects]

    return [r for r in rects if r.w > 0 and r.h > 0]


def year_forbidden_rects(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float,
    box_h_rel: float,
    gap_rel: float,
    stroke_rel: float,
    block_px: int,
    inflate_px: int,
) -> list[Rect]:
    masks = stroke_text_digit_masks(
        text,
        canvas=canvas,
        box_w_rel=box_w_rel,
        box_h_rel=box_h_rel,
        gap_rel=gap_rel,
        stroke_rel=stroke_rel,
    )
    if not masks:
        return []

    u = union_masks(masks, (canvas.width, canvas.height))
    if inflate_px > 0:
        k = int(inflate_px) * 2 + 1
        u = u.filter(ImageFilter.MaxFilter(size=max(3, k)))

    return mask_to_rects(u, block=int(block_px), inflate=0)


def year_forbidden_mask(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float,
    box_h_rel: float,
    gap_rel: float,
    stroke_rel: float,
    inflate_px: int,
) -> Image.Image:
    masks = stroke_text_digit_masks(
        text,
        canvas=canvas,
        box_w_rel=box_w_rel,
        box_h_rel=box_h_rel,
        gap_rel=gap_rel,
        stroke_rel=stroke_rel,
    )
    if not masks:
        return Image.new("L", (canvas.width, canvas.height), 0)

    u = union_masks(masks, (canvas.width, canvas.height))
    if inflate_px > 0:
        k = int(inflate_px) * 2 + 1
        u = u.filter(ImageFilter.MaxFilter(size=max(3, k)))
    return u


def mask_to_free_rects(mask: Image.Image, block: int) -> list[Rect]:
    if mask.mode != "L":
        mask = mask.convert("L")
    w, h = mask.size
    block = max(1, int(block))
    cols = max(1, (w + block - 1) // block)
    rows = max(1, (h + block - 1) // block)

    grid: list[list[bool]] = [[False] * cols for _ in range(rows)]
    for ry in range(rows):
        y0 = ry * block
        y1 = min(h, y0 + block)
        for cx in range(cols):
            x0 = cx * block
            x1 = min(w, x0 + block)
            if mask.crop((x0, y0, x1, y1)).getbbox() is None:
                grid[ry][cx] = True

    rects: list[Rect] = []
    active: dict[tuple[int, int], Rect] = {}

    for ry in range(rows):
        row_runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for cx in range(cols):
            if grid[ry][cx]:
                if run_start is None:
                    run_start = cx
            else:
                if run_start is not None:
                    row_runs.append((run_start, cx - 1))
                    run_start = None
        if run_start is not None:
            row_runs.append((run_start, cols - 1))

        next_active: dict[tuple[int, int], Rect] = {}
        for c0, c1 in row_runs:
            key = (c0, c1)
            if key in active:
                prev = active[key]
                next_active[key] = Rect(prev.x, prev.y, prev.w, prev.h + block)
            else:
                x0 = c0 * block
                x1 = min(w, (c1 + 1) * block)
                y0 = ry * block
                y1 = min(h, y0 + block)
                next_active[key] = Rect(x0, y0, x1 - x0, y1 - y0)

        for key, r in active.items():
            if key not in next_active:
                rects.append(r)
        active = next_active

    rects.extend(active.values())
    return [r for r in rects if r.w > 0 and r.h > 0]


def compute_layout_slice_excluding_mask(
    imgs: Sequence[ImgInfo],
    canvas: CanvasSpec,
    forbidden_mask: Image.Image,
    region_block_px: int,
    max_cells: int,
    slice_iters: int,
    slice_tol: float,
    slice_balance: float,
    slice_min_share: float,
    seed: int | None,
    min_region_side: int = 1,
) -> List[PlacedImg]:
    if not imgs:
        return []

    free = mask_to_free_rects(forbidden_mask, block=int(region_block_px))
    free = [r for r in free if r.w >= int(min_region_side) and r.h >= int(min_region_side)]
    if not free:
        return []

    total_area = sum(r.area for r in free) or 1
    target_cells = max(1, min(int(max_cells), len(imgs)))

    rng = random.Random(seed)
    chosen = list(imgs)
    rng.shuffle(chosen)
    chosen = chosen[:target_cells]

    alloc: list[int] = []
    remaining = len(chosen)
    for i, r in enumerate(free):
        if i == len(free) - 1:
            k = remaining
        else:
            k = max(1, int(round(len(chosen) * (r.area / total_area))))
            k = min(k, remaining - (len(free) - i - 1))
        alloc.append(k)
        remaining -= k

    placed_all: list[PlacedImg] = []
    start = 0
    for i, (r, k) in enumerate(zip(free, alloc)):
        subset = chosen[start : start + k]
        start += k
        if not subset:
            continue

        if k <= 1:
            info = subset[0]
            placed_all.append(PlacedImg(path=info.path, x=r.x, y=r.y, w=r.w, h=r.h))
            continue

        local_seed = None if seed is None else (int(seed) + 1009 * (i + 1))
        local = compute_layout_slice(
            subset,
            canvas=CanvasSpec(r.w, r.h),
            max_cells=k,
            slice_iters=slice_iters,
            slice_tol=slice_tol,
            slice_balance=slice_balance,
            slice_min_share=slice_min_share,
            seed=local_seed,
        )
        for p in local:
            placed_all.append(PlacedImg(path=p.path, x=p.x + r.x, y=p.y + r.y, w=p.w, h=p.h))

    return placed_all


def compute_layout_year_three_band(
    imgs: Sequence[ImgInfo],
    canvas: CanvasSpec,
    year: str,
    max_cells: int,
    slice_iters: int,
    slice_tol: float,
    slice_balance: float,
    slice_min_share: float,
    seed: int | None,
    mid_h_rel: float = 0.56,
    digit_box_w_rel: float = 0.90,
    digit_box_h_rel: float = 0.56,
    digit_gap_rel: float = 0.16,
    digit_stroke_rel: float = 0.24,
    digit_inset_rel: float = 0.08,
    region_block_px: int = 26,
    band_parts: Tuple[int, int, int] = (2, 5, 2),
) -> tuple[List[PlacedImg], Image.Image]:
    """
    改进版布局：
    - 上下区域用照片填充
    - 中间区域只在数字缺口放横图，数字间隙放竖图
    - 数字笔画框架保持纯色/渐变
    """
    if not imgs:
        return [], Image.new("L", (canvas.width, canvas.height), 0)

    w, h = int(canvas.width), int(canvas.height)
    if w <= 0 or h <= 0:
        return [], Image.new("L", (max(1, w), max(1, h)), 0)

    bt, bm, bb = (int(band_parts[0]), int(band_parts[1]), int(band_parts[2]))
    if bt < 0 or bm < 0 or bb < 0 or (bt + bm + bb) <= 0:
        bt, bm, bb = 2, 5, 2
    total_parts = bt + bm + bb
    raw = [h * (bt / float(total_parts)), h * (bm / float(total_parts)), h * (bb / float(total_parts))]
    base = [int(math.floor(v)) for v in raw]
    rem = int(h - sum(base))
    fracs = [raw[i] - base[i] for i in range(3)]
    order = sorted(range(3), key=lambda i: fracs[i], reverse=True)
    for i in range(max(0, rem)):
        base[order[i % 3]] += 1
    top_h, mh, bot_h = base[0], base[1], base[2]
    if top_h + mh + bot_h != h:
        bot_h = max(0, h - top_h - mh)
    band_top = Rect(0, 0, w, top_h)
    band_mid = Rect(0, top_h, w, mh)
    band_bot = Rect(0, top_h + mh, w, bot_h)

    rng = random.Random(seed)
    pool = list(imgs)
    rng.shuffle(pool)
    budget = max(1, min(len(pool), int(max_cells)))
    pool = pool[:budget]

    placed_all: list[PlacedImg] = []

    # 计算中间区域的数字相关矩形
    hole_rects: list[Rect] = []
    gap_rects: list[Rect] = []
    side_rects: list[Rect] = []
    hole_digit_map: list[str] = []  # 记录每个缺口属于哪个数字
    digit_mask_mid = Image.new("L", (w, max(1, mh)), 0)
    
    if band_mid.h > 0:
        mid_canvas = CanvasSpec(w, mh)
        boxes = text_digit_boxes(
            year,
            canvas=mid_canvas,
            box_w_rel=float(digit_box_w_rel),
            box_h_rel=float(digit_box_h_rel),
            gap_rel=float(digit_gap_rel),
        )

        masks = block_text_digit_masks(
            year,
            canvas=mid_canvas,
            box_w_rel=float(digit_box_w_rel),
            box_h_rel=float(digit_box_h_rel),
            gap_rel=float(digit_gap_rel),
            stroke_rel=float(digit_stroke_rel),
            inset_rel=float(digit_inset_rel),
        )
        if masks:
            digit_mask_mid = union_masks(masks, (w, mh))
            digit_mask_mid = digit_mask_mid.filter(
            ImageFilter.MinFilter(3)
)

        # 获取数字缺口区域（放图片），同时记录每个缺口属于哪个数字
        for ch, br in zip(year.strip(), boxes):
            digit_holes = block_digit_hole_rects(
                ch,
                br,
                stroke_rel=float(digit_stroke_rel),
                inset_rel=float(digit_inset_rel),
            )
            hole_rects.extend(digit_holes)
            hole_digit_map.extend([ch] * len(digit_holes))  # 记录属于哪个数字
        
        # 获取数字之间的间隙槽位（每个间隙放2张竖图，覆盖整个高度，扩展到笔画边缘）
        gap_slot_rects = text_digit_gap_slot_rects(
            year,
            canvas=mid_canvas,
            box_w_rel=float(digit_box_w_rel),
            box_h_rel=float(digit_box_h_rel),
            gap_rel=float(digit_gap_rel),
            stroke_rel=float(digit_stroke_rel),
            imgs_per_gap=2,
        )
        
        # 获取数字区域两侧的槽位（每侧放2张竖图，覆盖整个高度，扩展到笔画边缘）
        side_slot_rects = text_side_gap_slot_rects(
            year,
            canvas=mid_canvas,
            box_w_rel=float(digit_box_w_rel),
            box_h_rel=float(digit_box_h_rel),
            gap_rel=float(digit_gap_rel),
            stroke_rel=float(digit_stroke_rel),
            imgs_per_side=2,
        )
    else:
        gap_slot_rects = []
        side_slot_rects = []

    # 计算中间区域需要的图片数量：缺口 + 间隙槽位 + 两侧槽位
    mid_slot_count = len(hole_rects) + len(gap_slot_rects) + len(side_slot_rects)
    mid_slot_count = min(mid_slot_count, budget)
    remaining = max(0, budget - mid_slot_count)

    # 需要的竖图数量 = 间隙槽位 + 两侧槽位
    needed_vertical = len(gap_slot_rects) + len(side_slot_rects)

    # 上下区域按面积分配剩余图片
    a_top = band_top.area
    a_bot = band_bot.area
    a_sum = max(1, a_top + a_bot)

    top_k = int(round(remaining * (a_top / float(a_sum)))) if a_top > 0 else 0
    top_k = max(0, min(top_k, remaining))
    bot_k = max(0, remaining - top_k)

    # 先为中间区域预留图片，再分配上下区域
    # 分离横图和竖图
    horizontal_pool = [img for img in pool if img.aspect >= 1.0]  # 横图
    vertical_pool = [img for img in pool if img.aspect < 1.0]     # 竖图
    
    rng.shuffle(horizontal_pool)
    rng.shuffle(vertical_pool)
    
    # 记录需要旋转的图片路径
    rotate_paths: set[Path] = set()
    
    # 为中间区域分配图片（先分配，避免重复）
    # 1. 为缺口分配图片（0的缺口强制使用竖图）
    hole_assigned: list[ImgInfo] = []
    
    for i, rr in enumerate(hole_rects):
        digit_ch = hole_digit_map[i] if i < len(hole_digit_map) else ''
        
        # 0的缺口强制使用竖图
        if digit_ch == '0':
            if vertical_pool:
                hole_assigned.append(vertical_pool.pop(0))
            elif horizontal_pool:
                # 旋转横图
                img = horizontal_pool.pop(0)
                rotate_paths.add(img.path)
                hole_assigned.append(ImgInfo(path=img.path, w=img.h, h=img.w))
            else:
                hole_assigned.append(None)  # type: ignore
        else:
            # 其他数字使用横图
            if horizontal_pool:
                hole_assigned.append(horizontal_pool.pop(0))
            elif vertical_pool:
                hole_assigned.append(vertical_pool.pop(0))
            else:
                hole_assigned.append(None)  # type: ignore
    
    # 过滤掉None
    hole_assigned = [img for img in hole_assigned if img is not None]
    
    # 2. 为间隙和侧边分配竖图
    vertical_assigned: list[ImgInfo] = []
    for _ in range(needed_vertical):
        if vertical_pool:
            vertical_assigned.append(vertical_pool.pop(0))
        elif horizontal_pool:
            # 需要旋转横图
            img = horizontal_pool.pop(0)
            rotate_paths.add(img.path)
            vertical_assigned.append(ImgInfo(path=img.path, w=img.h, h=img.w))
    
    # 3. 剩余图片用于上下区域
    remaining_pool = horizontal_pool + vertical_pool
    rng.shuffle(remaining_pool)

    # 上区域照片
    top_imgs = remaining_pool[:top_k] if top_k > 0 else []
    
    # 下区域照片
    bot_imgs = remaining_pool[top_k:top_k + bot_k] if bot_k > 0 else []

    # 上区域布局
    if band_top.h > 0 and top_imgs:
        local = compute_layout_slice(
            top_imgs,
            canvas=CanvasSpec(band_top.w, band_top.h),
            max_cells=len(top_imgs),
            slice_iters=slice_iters,
            slice_tol=slice_tol,
            slice_balance=slice_balance,
            slice_min_share=slice_min_share,
            seed=None if seed is None else int(seed) + 101,
        )
        for p in local:
            placed_all.append(PlacedImg(path=p.path, x=p.x + band_top.x, y=p.y + band_top.y, w=p.w, h=p.h))

    # 下区域布局
    if band_bot.h > 0 and bot_imgs:
        local = compute_layout_slice(
            bot_imgs,
            canvas=CanvasSpec(band_bot.w, band_bot.h),
            max_cells=len(bot_imgs),
            slice_iters=slice_iters,
            slice_tol=slice_tol,
            slice_balance=slice_balance,
            slice_min_share=slice_min_share,
            seed=None if seed is None else int(seed) + 303,
        )
        for p in local:
            placed_all.append(PlacedImg(path=p.path, x=p.x + band_bot.x, y=p.y + band_bot.y, w=p.w, h=p.h))

    # 中间区域：数字缺口放图片（使用预先分配的图片）
    if band_mid.h > 0 and hole_rects:
        for i, rr in enumerate(hole_rects):
            if i < len(hole_assigned):
                info = hole_assigned[i]
                rotate = 90 if info.path in rotate_paths else 0
                placed_all.append(
                    PlacedImg(
                        path=info.path,
                        x=rr.x,
                        y=rr.y + band_mid.y,
                        w=rr.w,
                        h=rr.h,
                        rotate=rotate,
                    )
                )

    # 中间区域：数字间隙槽位放竖图（根据图片比例分配高度，避免裁切）
    if band_mid.h > 0 and gap_slot_rects:
        # 按间隙分组处理（每个间隙有多个槽位）
        imgs_per_gap = 2
        n_gaps = len(gap_slot_rects) // imgs_per_gap if imgs_per_gap > 0 else 0
        vert_idx = 0
        
        for gap_i in range(n_gaps):
            # 获取这个间隙的基础信息（用第一个槽位）
            base_slot = gap_slot_rects[gap_i * imgs_per_gap]
            gap_x = base_slot.x
            gap_w = base_slot.w
            gap_total_h = mh  # 整个中间带高度
            
            # 收集这个间隙要放的图片（使用预先分配的竖图）
            gap_imgs = []
            for j in range(imgs_per_gap):
                if vert_idx + j < len(vertical_assigned):
                    gap_imgs.append(vertical_assigned[vert_idx + j])
            
            if not gap_imgs:
                vert_idx += imgs_per_gap
                continue
            
            # 根据图片比例分配高度（保持原始比例，填满宽度后计算高度）
            scaled_heights = []
            for img in gap_imgs:
                # 图片缩放到槽位宽度后的高度
                scale = gap_w / img.w if img.w > 0 else 1
                scaled_h = int(round(img.h * scale))
                scaled_heights.append(max(1, scaled_h))
            
            # 按比例分配总高度
            total_scaled_h = sum(scaled_heights)
            if total_scaled_h > 0:
                allocated_heights = []
                remaining_h = gap_total_h
                for i, sh in enumerate(scaled_heights):
                    if i == len(scaled_heights) - 1:
                        h = remaining_h
                    else:
                        h = int(round(gap_total_h * (sh / total_scaled_h)))
                        h = min(h, remaining_h)
                    allocated_heights.append(max(1, h))
                    remaining_h -= h
            else:
                allocated_heights = [gap_total_h // len(gap_imgs)] * len(gap_imgs)
            
            # 放置图片（添加旋转标记）
            current_y = 0
            for j, img in enumerate(gap_imgs):
                rotate = 90 if img.path in rotate_paths else 0
                placed_all.append(
                    PlacedImg(
                        path=img.path,
                        x=gap_x,
                        y=current_y + band_mid.y,
                        w=gap_w,
                        h=allocated_heights[j],
                        rotate=rotate,
                    )
                )
                current_y += allocated_heights[j]
            
            vert_idx += len(gap_imgs)

    # 中间区域：两侧槽位放竖图（根据图片比例分配高度，避免裁切）
    if band_mid.h > 0 and side_slot_rects:
        imgs_per_side = 2
        n_sides = len(side_slot_rects) // imgs_per_side if imgs_per_side > 0 else 0
        vert_idx = len(gap_slot_rects)  # 继续用竖图（从gap分配后继续）
        
        for side_i in range(n_sides):
            base_slot = side_slot_rects[side_i * imgs_per_side]
            side_x = base_slot.x
            side_w = base_slot.w
            side_total_h = mh
            
            # 收集这个侧边要放的图片（使用预先分配的竖图）
            side_imgs = []
            for j in range(imgs_per_side):
                if vert_idx + j < len(vertical_assigned):
                    side_imgs.append(vertical_assigned[vert_idx + j])
            
            if not side_imgs:
                vert_idx += imgs_per_side
                continue
            
            # 根据图片比例分配高度
            scaled_heights = []
            for img in side_imgs:
                scale = side_w / img.w if img.w > 0 else 1
                scaled_h = int(round(img.h * scale))
                scaled_heights.append(max(1, scaled_h))
            
            total_scaled_h = sum(scaled_heights)
            if total_scaled_h > 0:
                allocated_heights = []
                remaining_h = side_total_h
                for i, sh in enumerate(scaled_heights):
                    if i == len(scaled_heights) - 1:
                        h = remaining_h
                    else:
                        h = int(round(side_total_h * (sh / total_scaled_h)))
                        h = min(h, remaining_h)
                    allocated_heights.append(max(1, h))
                    remaining_h -= h
            else:
                allocated_heights = [side_total_h // len(side_imgs)] * len(side_imgs)
            
            current_y = 0
            for j, img in enumerate(side_imgs):
                rotate = 90 if img.path in rotate_paths else 0
                placed_all.append(
                    PlacedImg(
                        path=img.path,
                        x=side_x,
                        y=current_y + band_mid.y,
                        w=side_w,
                        h=allocated_heights[j],
                        rotate=rotate,
                    )
                )
                current_y += allocated_heights[j]
            
            vert_idx += len(side_imgs)

    forbidden = Image.new("L", (w, h), 0)
    if band_mid.h > 0:
        forbidden.paste(digit_mask_mid, (0, band_mid.y))
    return placed_all, forbidden


def compute_layout_grid_excluding_mask(
    imgs: Sequence[ImgInfo],
    canvas: CanvasSpec,
    forbidden_mask: Image.Image,
    max_cells: int,
    seed: int | None,
) -> List[PlacedImg]:
    if not imgs:
        return []

    if forbidden_mask.mode != "L":
        forbidden_mask = forbidden_mask.convert("L")
    if forbidden_mask.size != (canvas.width, canvas.height):
        forbidden_mask = forbidden_mask.resize((canvas.width, canvas.height), resample=Image.Resampling.NEAREST)

    target = max(1, int(max_cells))
    aspect = canvas.width / canvas.height if canvas.height else 1.0
    cols = max(1, int(round(math.sqrt(target * aspect))))
    rows = max(1, int(math.ceil(target / float(cols))))

    col_ws = distribute_pixels(canvas.width, cols)
    row_hs = distribute_pixels(canvas.height, rows)

    free: list[Rect] = []
    y = 0
    for rh in row_hs:
        x = 0
        for cw in col_ws:
            if forbidden_mask.crop((x, y, x + cw, y + rh)).getbbox() is None:
                free.append(Rect(x, y, cw, rh))
            x += cw
        y += rh

    if not free:
        return []

    rng = random.Random(seed)
    chosen = list(imgs)
    rng.shuffle(chosen)

    placed: list[PlacedImg] = []
    for i, r in enumerate(free):
        info = chosen[i % len(chosen)]
        placed.append(PlacedImg(path=info.path, x=r.x, y=r.y, w=r.w, h=r.h))

    return placed


def solid_text_digit_masks(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float = 0.84,
    box_h_rel: float = 0.42,
    gap_rel: float = 0.18,
    stroke_rel: float = 0.20,
) -> list[Image.Image]:
    t = text.strip()
    if not t:
        return []
    if any(ch not in "0123456789" for ch in t):
        raise ValueError("text must be digits only")

    boxes = text_digit_boxes(t, canvas=canvas, box_w_rel=box_w_rel, box_h_rel=box_h_rel, gap_rel=gap_rel)
    masks: list[Image.Image] = []

    for ch, r in zip(t, boxes):
        m = Image.new("L", (canvas.width, canvas.height), 0)
        draw = ImageDraw.Draw(m)

        th = max(1, int(round(r.h * float(stroke_rel))))
        pad = max(1, th // 2)

        def rect(x0: int, y0: int, x1: int, y1: int) -> None:
            draw.rectangle((x0, y0, x1, y1), fill=255)

        x0, y0, w, h = r.x, r.y, r.w, r.h
        x1, y1 = x0 + w, y0 + h

        inset = max(1, int(round(0.06 * min(w, h))))
        ix0, iy0, ix1, iy1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
        mid_y = (iy0 + iy1) // 2

        if ch == "0":
            if hasattr(draw, "rounded_rectangle"):
                draw.rounded_rectangle((ix0, iy0, ix1, iy1), radius=max(1, th), fill=255)
            else:
                rect(ix0, iy0, ix1, iy1)
        elif ch == "1":
            cx = (ix0 + ix1) // 2
            rect(cx - th // 2, iy0, cx + th // 2 + 1, iy1)
        elif ch == "2":
            rect(ix0, iy0, ix1, iy0 + th)
            rect(ix1 - th, iy0, ix1, mid_y)
            rect(ix0, mid_y - th // 2, ix1, mid_y + th // 2 + 1)
            rect(ix0, mid_y, ix0 + th, iy1)
            rect(ix0, iy1 - th, ix1, iy1)
        elif ch == "3":
            rect(ix0, iy0, ix1, iy0 + th)
            rect(ix0, mid_y - th // 2, ix1, mid_y + th // 2 + 1)
            rect(ix0, iy1 - th, ix1, iy1)
            rect(ix1 - th, iy0, ix1, iy1)
        elif ch == "4":
            rect(ix0, iy0, ix0 + th, mid_y)
            rect(ix0, mid_y - th // 2, ix1, mid_y + th // 2 + 1)
            rect(ix1 - th, iy0, ix1, iy1)
        elif ch == "5":
            rect(ix0, iy0, ix1, iy0 + th)
            rect(ix0, iy0, ix0 + th, mid_y)
            rect(ix0, mid_y - th // 2, ix1, mid_y + th // 2 + 1)
            rect(ix1 - th, mid_y, ix1, iy1)
            rect(ix0, iy1 - th, ix1, iy1)
        elif ch == "6":
            rect(ix0, iy0, ix1, iy0 + th)
            rect(ix0, iy0, ix0 + th, iy1)
            rect(ix0, mid_y - th // 2, ix1, mid_y + th // 2 + 1)
            rect(ix1 - th, mid_y, ix1, iy1)
            rect(ix0, iy1 - th, ix1, iy1)
        elif ch == "7":
            rect(ix0, iy0, ix1, iy0 + th)
            poly = [(ix1, iy0 + th), (ix1 - th, iy0 + th), (ix0 + th, iy1), (ix0, iy1)]
            draw.polygon(poly, fill=255)
        elif ch == "8":
            rect(ix0, iy0, ix1, iy1)
            rect(ix0 + pad, iy0 + pad, ix1 - pad, iy1 - pad)
        elif ch == "9":
            rect(ix0, iy0, ix1, iy0 + th)
            rect(ix0, iy0, ix0 + th, mid_y)
            rect(ix0, mid_y - th // 2, ix1, mid_y + th // 2 + 1)
            rect(ix1 - th, iy0, ix1, iy1)
            rect(ix0, iy1 - th, ix1, iy1)
        else:
            raise ValueError(f"unsupported digit: {ch}")

        masks.append(m)

    return masks


def stroke_text_digit_masks(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float = 0.84,
    box_h_rel: float = 0.42,
    gap_rel: float = 0.18,
    stroke_rel: float = 0.14,
) -> list[Image.Image]:
    t = text.strip()
    if not t:
        return []
    if any(ch not in "0123456789" for ch in t):
        raise ValueError("text must be digits only")

    box_w = max(1, int(round(canvas.width * float(box_w_rel))))
    box_h = max(1, int(round(canvas.height * float(box_h_rel))))
    x0 = (canvas.width - box_w) // 2
    y0 = (canvas.height - box_h) // 2

    n = len(t)
    gap = int(round((box_w / max(1, n)) * float(gap_rel)))
    total_gap = gap * (n - 1)
    digit_w = max(1, (box_w - total_gap) // n)
    digit_h = box_h

    stroke = max(1, int(round(digit_h * float(stroke_rel))))
    r = max(1, stroke // 2)

    masks: list[Image.Image] = []
    for i, ch in enumerate(t):
        m = Image.new("L", (canvas.width, canvas.height), 0)
        draw = ImageDraw.Draw(m)
        dx = x0 + i * (digit_w + gap)

        for poly in _digit_polylines(ch):
            pts = [(dx + int(round(px * digit_w)), y0 + int(round(py * digit_h))) for (px, py) in poly]
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=stroke)
            for (x, y) in pts:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

        masks.append(m)

    return masks


def block_text_digit_masks(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float = 0.90,
    box_h_rel: float = 0.92,
    gap_rel: float = 0.12,
    stroke_rel: float = 0.22,
    inset_rel: float = 0.08,
) -> list[Image.Image]:
    t = text.strip()
    if not t:
        return []
    if any(ch not in "0123456789" for ch in t):
        raise ValueError("text must be digits only")

    boxes = text_digit_boxes(t, canvas=canvas, box_w_rel=box_w_rel, box_h_rel=box_h_rel, gap_rel=gap_rel)
    masks: list[Image.Image] = []

    for ch, r in zip(t, boxes):
        m = Image.new("L", (canvas.width, canvas.height), 0)
        draw = ImageDraw.Draw(m)
        rects = block_digit_stroke_rects(ch, r, stroke_rel=float(stroke_rel), inset_rel=float(inset_rel))
        for rr in rects:
            draw.rectangle((rr.x, rr.y, rr.x + rr.w, rr.y + rr.h), fill=255)
        masks.append(m)

    return masks


def centered_box_mask(
    canvas: CanvasSpec,
    box_w_rel: float,
    box_h_rel: float,
    radius_rel: float = 0.04,
) -> Image.Image:
    box_w = max(1, int(round(canvas.width * float(box_w_rel))))
    box_h = max(1, int(round(canvas.height * float(box_h_rel))))
    x0 = (canvas.width - box_w) // 2
    y0 = (canvas.height - box_h) // 2
    r = max(0, int(round(min(box_w, box_h) * float(radius_rel))))

    m = Image.new("L", (canvas.width, canvas.height), 0)
    draw = ImageDraw.Draw(m)
    if r > 0 and hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=r, fill=255)
    else:
        draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=255)
    return m


def fill_text_masks(img: Image.Image, masks: Sequence[Image.Image], colors: Sequence[Tuple[int, int, int]]) -> Image.Image:
    if not masks:
        return img
    if not colors:
        raise ValueError("colors must be non-empty")

    out = img.copy()
    for i, m in enumerate(masks):
        c = colors[i % len(colors)]
        layer = Image.new("RGB", out.size, color=c)
        out.paste(layer, (0, 0), mask=m)
    return out


def block_digit_stroke_rects(d: str, r: Rect, stroke_rel: float = 0.22, inset_rel: float = 0.08) -> list[Rect]:
    if d not in "0123456789":
        raise ValueError(f"unsupported digit: {d}")

    x0, y0, w, h = int(r.x), int(r.y), int(r.w), int(r.h)
    x1, y1 = x0 + w, y0 + h
    if w <= 0 or h <= 0:
        return []

    inset = max(0, int(round(min(w, h) * float(inset_rel))))
    ix0, iy0, ix1, iy1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
    if ix1 <= ix0 or iy1 <= iy0:
        return []

    t = max(1, int(round((iy1 - iy0) * float(stroke_rel))))
    mid_y = (iy0 + iy1) // 2

    def rect(ax0: int, ay0: int, ax1: int, ay1: int) -> Rect:
        ax0 = max(x0, min(x1, ax0))
        ax1 = max(x0, min(x1, ax1))
        ay0 = max(y0, min(y1, ay0))
        ay1 = max(y0, min(y1, ay1))
        if ax1 <= ax0 or ay1 <= ay0:
            return Rect(0, 0, 0, 0)
        return Rect(ax0, ay0, ax1 - ax0, ay1 - ay0)

    top = rect(ix0, iy0, ix1, iy0 + t)
    mid = rect(ix0, mid_y - t // 2, ix1, mid_y + (t - t // 2))
    bot = rect(ix0, iy1 - t, ix1, iy1)

    left_top = rect(ix0, iy0, ix0 + t, mid_y)
    left_bot = rect(ix0, mid_y, ix0 + t, iy1)
    right_top = rect(ix1 - t, iy0, ix1, mid_y)
    right_bot = rect(ix1 - t, mid_y, ix1, iy1)

    segs: list[Rect] = []

    def add(rr: Rect) -> None:
        if rr.w > 0 and rr.h > 0:
            segs.append(rr)

    if d == "0":
        add(top)
        add(bot)
        add(left_top)
        add(left_bot)
        add(right_top)
        add(right_bot)
    elif d == "1":
        add(right_top)
        add(right_bot)
    elif d == "2":
        add(top)
        add(right_top)
        add(mid)
        add(left_bot)
        add(bot)
    elif d == "3":
        add(top)
        add(right_top)
        add(mid)
        add(right_bot)
        add(bot)
    elif d == "4":
        add(left_top)
        add(mid)
        add(right_top)
        add(right_bot)
    elif d == "5":
        add(top)
        add(left_top)
        add(mid)
        add(right_bot)
        add(bot)
    elif d == "6":
        add(top)
        add(left_top)
        add(left_bot)
        add(mid)
        add(right_bot)
        add(bot)
    elif d == "7":
        add(top)
        add(right_top)
        add(right_bot)
    elif d == "8":
        add(top)
        add(mid)
        add(bot)
        add(left_top)
        add(left_bot)
        add(right_top)
        add(right_bot)
    elif d == "9":
        add(top)
        add(left_top)
        add(mid)
        add(right_top)
        add(right_bot)
        add(bot)

    return segs


def _fit_aspect_in_rect(r: Rect, aspect: float) -> Rect:
    if r.w <= 0 or r.h <= 0:
        return Rect(0, 0, 0, 0)
    a = float(aspect) if float(aspect) > 0 else 1.0
    rw = int(r.w)
    rh = int(r.h)
    if rh <= 0:
        return Rect(r.x, r.y, rw, rh)

    cur = rw / float(rh)
    if cur >= a:
        w = max(1, int(round(rh * a)))
        w = min(w, rw)
        x = r.x + (rw - w) // 2
        return Rect(x, r.y, w, rh)
    h = max(1, int(round(rw / a)))
    h = min(h, rh)
    y = r.y + (rh - h) // 2
    return Rect(r.x, y, rw, h)


def block_digit_hole_rects(d: str, r: Rect, stroke_rel: float = 0.22, inset_rel: float = 0.08) -> list[Rect]:
    if d not in "0123456789":
        raise ValueError(f"unsupported digit: {d}")

    x0, y0, w, h = int(r.x), int(r.y), int(r.w), int(r.h)
    x1, y1 = x0 + w, y0 + h
    if w <= 0 or h <= 0:
        return []

    inset = max(0, int(round(min(w, h) * float(inset_rel))))
    ix0, iy0, ix1, iy1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
    if ix1 <= ix0 or iy1 <= iy0:
        return []

    t = max(1, int(round((iy1 - iy0) * float(stroke_rel))))
    mid_y = (iy0 + iy1) // 2
    gap = max(1, int(round(t * 0.45)))

    def mk(ax0: int, ay0: int, ax1: int, ay1: int) -> Rect:
        rr = Rect(ax0, ay0, ax1 - ax0, ay1 - ay0)
        if rr.w <= 0 or rr.h <= 0:
            return Rect(0, 0, 0, 0)
        return rr

    holes: list[Rect] = []

    if d == "0":
        # 0的内部是一个大的竖向区域
        inner = mk(ix0 + t + gap, iy0 + t + gap, ix1 - t - gap, iy1 - t - gap)
        if inner.w > 0 and inner.h > 0:
            holes.append(inner)  # 不强制比例，保持原区域
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "1":
        # 1的左侧区域
        left_area = mk(ix0 + gap, iy0 + gap, ix1 - t - gap, iy1 - gap)
        if left_area.w > 0 and left_area.h > 0:
            holes.append(left_area)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "2":
        # 2的左上和右下两个缺口
        tl = mk(ix0 + gap, iy0 + t + gap, ix1 - t - gap, mid_y - gap)
        br = mk(ix0 + t + gap, mid_y + gap, ix1 - gap, iy1 - t - gap)
        if tl.w > 0 and tl.h > 0:
            holes.append(tl)
        if br.w > 0 and br.h > 0:
            holes.append(br)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "3":
        # 3的左上和左下两个缺口
        tl = mk(ix0 + gap, iy0 + t + gap, ix1 - t - gap, mid_y - gap)
        bl = mk(ix0 + gap, mid_y + gap, ix1 - t - gap, iy1 - t - gap)
        if tl.w > 0 and tl.h > 0:
            holes.append(tl)
        if bl.w > 0 and bl.h > 0:
            holes.append(bl)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "4":
        # 4的左下和上部区域
        lb = mk(ix0 + gap, mid_y + gap, ix1 - t - gap, iy1 - gap)
        if lb.w > 0 and lb.h > 0:
            holes.append(lb)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "5":
        # 5的右上和左下两个缺口
        ur = mk(ix0 + t + gap, iy0 + t + gap, ix1 - gap, mid_y - gap)
        ll = mk(ix0 + gap, mid_y + gap, ix1 - t - gap, iy1 - t - gap)
        if ur.w > 0 and ur.h > 0:
            holes.append(ur)
        if ll.w > 0 and ll.h > 0:
            holes.append(ll)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "6":
        # 6的右上缺口和内部圆
        ur = mk(ix0 + t + gap, iy0 + t + gap, ix1 - gap, mid_y - gap)
        inner = mk(ix0 + t + gap, mid_y + gap, ix1 - t - gap, iy1 - t - gap)
        if ur.w > 0 and ur.h > 0:
            holes.append(ur)
        if inner.w > 0 and inner.h > 0:
            holes.append(inner)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "7":
        # 7的左侧和下部区域
        left_area = mk(ix0 + gap, iy0 + t + gap, ix1 - t - gap, iy1 - gap)
        if left_area.w > 0 and left_area.h > 0:
            holes.append(left_area)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "8":
        # 8的上下两个内部区域
        top_inner = mk(ix0 + t + gap, iy0 + t + gap, ix1 - t - gap, mid_y - gap)
        bot_inner = mk(ix0 + t + gap, mid_y + gap, ix1 - t - gap, iy1 - t - gap)
        if top_inner.w > 0 and top_inner.h > 0:
            holes.append(top_inner)
        if bot_inner.w > 0 and bot_inner.h > 0:
            holes.append(bot_inner)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    if d == "9":
        # 9的内部圆和左下缺口
        inner = mk(ix0 + t + gap, iy0 + t + gap, ix1 - t - gap, mid_y - gap)
        ll = mk(ix0 + gap, mid_y + gap, ix1 - t - gap, iy1 - t - gap)
        if inner.w > 0 and inner.h > 0:
            holes.append(inner)
        if ll.w > 0 and ll.h > 0:
            holes.append(ll)
        return [rr for rr in holes if rr.w > 0 and rr.h > 0]

    return []


def block_text_digit_tile_rects(
    text: str,
    canvas: CanvasSpec,
    box_w_rel: float = 0.90,
    box_h_rel: float = 0.92,
    gap_rel: float = 0.12,
    stroke_rel: float = 0.22,
    inset_rel: float = 0.08,
) -> list[Rect]:
    t = text.strip()
    if not t:
        return []
    if any(ch not in "0123456789" for ch in t):
        raise ValueError("text must be digits only")

    boxes = text_digit_boxes(t, canvas=canvas, box_w_rel=box_w_rel, box_h_rel=box_h_rel, gap_rel=gap_rel)
    out: list[Rect] = []
    for ch, b in zip(t, boxes):
        out.extend(block_digit_stroke_rects(ch, b, stroke_rel=float(stroke_rel), inset_rel=float(inset_rel)))
    return out


def pad_reflect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    if w == target_w and h == target_h:
        return img

    tiles_x = max(3, int(math.ceil(target_w / w)) + 2)
    tiles_y = max(3, int(math.ceil(target_h / h)) + 2)
    big = Image.new(img.mode, (tiles_x * w, tiles_y * h))

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = img
            if tx % 2 == 1:
                tile = ImageOps.mirror(tile)
            if ty % 2 == 1:
                tile = ImageOps.flip(tile)
            big.paste(tile, (tx * w, ty * h))

    left = max(0, (big.width - target_w) // 2)
    top = max(0, (big.height - target_h) // 2)
    return big.crop((left, top, left + target_w, top + target_h))


def distribute_pixels(total: int, count: int) -> List[int]:
    if count <= 0:
        return []
    base = total // count
    rem = total - base * count
    out = [base] * count
    for i in range(rem):
        out[i] += 1
    return out


def build_rows(
    imgs: Sequence[ImgInfo],
    canvas_w: int,
    target_row_h: int,
    row_h_jitter: float,
    seed: int | None,
) -> List[Tuple[List[ImgInfo], int]]:
    rng = random.Random(seed)

    if not imgs:
        return []

    rows: List[Tuple[List[ImgInfo], int]] = []
    cur: List[ImgInfo] = []
    cur_sum = 0.0

    def new_target_h() -> int:
        if row_h_jitter <= 0:
            return target_row_h
        lo = max(20, int(round(target_row_h * (1.0 - row_h_jitter))))
        hi = max(lo + 1, int(round(target_row_h * (1.0 + row_h_jitter))))
        return rng.randint(lo, hi)

    row_h = new_target_h()

    for info in imgs:
        cur.append(info)
        cur_sum += info.aspect

        est_w = cur_sum * row_h
        if est_w >= canvas_w and len(cur) >= 2:
            rows.append((cur, row_h))
            cur = []
            cur_sum = 0.0
            row_h = new_target_h()

    if cur:
        rows.append((cur, row_h))

    return rows


def compute_layout_justified(
    imgs: Sequence[ImgInfo],
    canvas: CanvasSpec,
    target_row_h: int,
    row_h_jitter: float,
    seed: int | None,
) -> List[PlacedImg]:
    rows = build_rows(imgs, canvas.width, target_row_h, row_h_jitter, seed)
    if not rows:
        return []

    row_heights: List[int] = []
    row_items: List[List[ImgInfo]] = []

    for items, desired_h in rows:
        sum_aspect = sum(i.aspect for i in items) or 1.0
        row_h = canvas.width / sum_aspect
        row_h_int = max(1, int(round(row_h)))
        row_heights.append(row_h_int)
        row_items.append(list(items))

    total_h = sum(row_heights)
    if total_h <= 0:
        raise RuntimeError("invalid layout height")

    scale = canvas.height / total_h
    scaled_heights_float = [h * scale for h in row_heights]

    scaled_heights_int = [max(1, int(math.floor(h))) for h in scaled_heights_float]
    diff = canvas.height - sum(scaled_heights_int)
    if diff != 0:
        remainders = [h - math.floor(h) for h in scaled_heights_float]
        order = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=(diff > 0))
        step = 1 if diff > 0 else -1
        for k in range(abs(diff)):
            scaled_heights_int[order[k % len(order)]] += step

    y = 0
    placed: List[PlacedImg] = []
    for r, items in enumerate(row_items):
        row_h = scaled_heights_int[r]

        widths_float = [i.aspect * row_h for i in items]
        widths_int = [max(1, int(math.floor(w))) for w in widths_float]
        diff_w = canvas.width - sum(widths_int)
        if diff_w != 0:
            remainders = [w - math.floor(w) for w in widths_float]
            order = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=(diff_w > 0))
            step = 1 if diff_w > 0 else -1
            for k in range(abs(diff_w)):
                widths_int[order[k % len(order)]] += step

        x = 0
        for info, w in zip(items, widths_int):
            placed.append(PlacedImg(path=info.path, x=x, y=y, w=w, h=row_h))
            x += w
        y += row_h

    if y != canvas.height:
        placed.sort(key=lambda p: (p.y, p.x))
        last_row_y = max(p.y for p in placed)
        last_row = [p for p in placed if p.y == last_row_y]
        delta = canvas.height - y
        if last_row:
            for i in range(len(last_row)):
                idx = placed.index(last_row[i])
                placed[idx] = PlacedImg(
                    path=placed[idx].path,
                    x=placed[idx].x,
                    y=placed[idx].y,
                    w=placed[idx].w,
                    h=placed[idx].h + delta,
                )

    return placed


def _slice_combine(a: SliceNode, b: SliceNode, split: str) -> SliceNode:
    if split == "v":
        aspect = (a.aspect or 1.0) + (b.aspect or 1.0)
    else:
        ra = a.aspect or 1.0
        rb = b.aspect or 1.0
        aspect = (ra * rb) / (ra + rb)
    return SliceNode(aspect=aspect, left=a, right=b, split=split, leaves=a.leaves + b.leaves)


def build_slice_tree(
    imgs: Sequence[ImgInfo],
    target_aspect: float,
    seed: int | None,
    iters: int,
    tol: float,
    balance_weight: float,
    min_share: float,
) -> SliceNode:
    if not imgs:
        raise ValueError("no images")

    rng = random.Random(seed)
    target_aspect = target_aspect or 1.0

    best: SliceNode | None = None
    best_err = float("inf")

    img_list = list(imgs)
    for _ in range(max(1, int(iters))):
        rng.shuffle(img_list)
        nodes: List[SliceNode] = [SliceNode(aspect=i.aspect, img=i, leaves=1) for i in img_list]

        penalty_sum = 0.0
        combines = 0

        while len(nodes) > 1:
            a_idx = rng.randrange(len(nodes))
            a = nodes.pop(a_idx)

            if len(nodes) == 1:
                b = nodes.pop(0)
            else:
                k = min(6, len(nodes))
                cand_idxs = rng.sample(range(len(nodes)), k=k)
                best_c = cand_idxs[0]
                aa = a.aspect or 1.0
                best_leaf_d = abs(nodes[best_c].leaves - a.leaves)
                best_aspect_d = abs(math.log(((nodes[best_c].aspect or 1.0) / aa)))
                best_score = best_leaf_d + 3.0 * best_aspect_d
                for ci in cand_idxs[1:]:
                    leaf_d = abs(nodes[ci].leaves - a.leaves)
                    aspect_d = abs(math.log(((nodes[ci].aspect or 1.0) / aa)))
                    score = leaf_d + 3.0 * aspect_d
                    if score < best_score:
                        best_score = score
                        best_c = ci
                b = nodes.pop(best_c)

            def option_score(left: SliceNode, right: SliceNode, split: str) -> tuple[float, float, SliceNode]:
                node = _slice_combine(left, right, split)
                aspect_err = abs(math.log((node.aspect or 1.0) / target_aspect))

                la = max(1, int(left.leaves))
                lb = max(1, int(right.leaves))
                desired_left = la / (la + lb)
                desired_right = lb / (la + lb)

                da = left.aspect or 1.0
                db = right.aspect or 1.0
                denom = da + db
                if denom <= 0:
                    share_left = 0.5
                    share_right = 0.5
                elif split == "v":
                    share_left = da / denom
                    share_right = db / denom
                else:
                    share_left = db / denom
                    share_right = da / denom

                ms = max(0.0, min(0.49, float(min_share)))
                skew = 0.0
                if share_left < ms:
                    skew += (ms - share_left)
                elif share_left > (1.0 - ms):
                    skew += (share_left - (1.0 - ms))

                penalty = (abs(share_left - desired_left) + abs(share_right - desired_right) + 2.0 * skew) / 2.0
                score = aspect_err + (max(0.0, float(balance_weight)) * penalty)
                return score, penalty, node

            options = [
                option_score(a, b, "v"),
                option_score(b, a, "v"),
                option_score(a, b, "h"),
                option_score(b, a, "h"),
            ]
            best_opt = min(options, key=lambda t: t[0])
            best_score_opt = best_opt[0]
            tied = [o for o in options if abs(o[0] - best_score_opt) <= 1e-12]
            chosen_score, chosen_penalty, chosen_node = rng.choice(tied)

            penalty_sum += chosen_penalty
            combines += 1
            nodes.append(chosen_node)

        root = nodes[0]
        aspect_err = abs(math.log((root.aspect or 1.0) / target_aspect))
        avg_penalty = (penalty_sum / combines) if combines else 0.0
        err = aspect_err + (max(0.0, float(balance_weight)) * avg_penalty)
        if err < best_err:
            best_err = err
            best = root
            if aspect_err <= tol and float(balance_weight) <= 0.0:
                break

    if best is None:
        raise RuntimeError("failed to build slice tree")
    return best


def layout_slice_tree(root: SliceNode, rect: Rect) -> List[PlacedImg]:
    placed: List[PlacedImg] = []

    def rec(node: SliceNode, r: Rect) -> None:
        if node.img is not None:
            placed.append(PlacedImg(path=node.img.path, x=r.x, y=r.y, w=r.w, h=r.h))
            return
        if node.left is None or node.right is None or node.split not in {"v", "h"}:
            return

        a = node.left
        b = node.right
        da = a.aspect or 1.0
        db = b.aspect or 1.0

        if node.split == "v":
            denom = da + db
            w1 = int(round(r.w * ((da / denom) if denom else 0.5)))
            w1 = max(1, min(r.w - 1, w1))
            ra = Rect(r.x, r.y, w1, r.h)
            rb = Rect(r.x + w1, r.y, r.w - w1, r.h)
        else:
            denom = da + db
            h1 = int(round(r.h * ((db / denom) if denom else 0.5)))
            h1 = max(1, min(r.h - 1, h1))
            ra = Rect(r.x, r.y, r.w, h1)
            rb = Rect(r.x, r.y + h1, r.w, r.h - h1)

        rec(a, ra)
        rec(b, rb)

    rec(root, rect)
    return placed


def compute_layout_slice(
    imgs: Sequence[ImgInfo],
    canvas: CanvasSpec,
    max_cells: int,
    slice_iters: int,
    slice_tol: float,
    slice_balance: float,
    slice_min_share: float,
    seed: int | None,
) -> List[PlacedImg]:
    if not imgs:
        return []

    rng = random.Random(seed)
    chosen = list(imgs)
    rng.shuffle(chosen)
    chosen = chosen[: max(1, min(len(chosen), int(max_cells)))]

    target_aspect = canvas.width / canvas.height if canvas.height else 1.0
    root = build_slice_tree(
        chosen,
        target_aspect=target_aspect,
        seed=seed,
        iters=slice_iters,
        tol=slice_tol,
        balance_weight=max(0.0, min(1.0, float(slice_balance))),
        min_share=max(0.0, min(0.49, float(slice_min_share))),
    )
    return layout_slice_tree(root, Rect(0, 0, canvas.width, canvas.height))


def compute_layout_slice_excluding(
    imgs: Sequence[ImgInfo],
    canvas: CanvasSpec,
    forbidden: Sequence[Rect],
    max_cells: int,
    slice_iters: int,
    slice_tol: float,
    slice_balance: float,
    slice_min_share: float,
    seed: int | None,
    min_region_side: int = 80,
) -> List[PlacedImg]:
    base = Rect(0, 0, canvas.width, canvas.height)
    free = subtract_rects(base, list(forbidden))
    free = [r for r in free if r.w >= min_region_side and r.h >= min_region_side]
    if not free:
        return []

    total_area = sum(r.area for r in free) or 1
    target_cells = max(1, min(int(max_cells), len(imgs)))

    rng = random.Random(seed)
    chosen = list(imgs)
    rng.shuffle(chosen)
    chosen = chosen[:target_cells]

    alloc: list[int] = []
    remaining = len(chosen)
    for i, r in enumerate(free):
        if i == len(free) - 1:
            k = remaining
        else:
            k = max(1, int(round(len(chosen) * (r.area / total_area))))
            k = min(k, remaining - (len(free) - i - 1))
        alloc.append(k)
        remaining -= k

    placed_all: list[PlacedImg] = []
    start = 0
    for i, (r, k) in enumerate(zip(free, alloc)):
        subset = chosen[start : start + k]
        start += k
        if not subset:
            continue

        local_seed = None if seed is None else (int(seed) + 1009 * (i + 1))
        local = compute_layout_slice(
            subset,
            canvas=CanvasSpec(r.w, r.h),
            max_cells=k,
            slice_iters=slice_iters,
            slice_tol=slice_tol,
            slice_balance=slice_balance,
            slice_min_share=slice_min_share,
            seed=local_seed,
        )
        for p in local:
            placed_all.append(PlacedImg(path=p.path, x=p.x + r.x, y=p.y + r.y, w=p.w, h=p.h))

    return placed_all


def compute_layout_bsp(
    imgs: Sequence[ImgInfo],
    canvas: CanvasSpec,
    max_cells: int,
    min_cell: int,
    split_jitter: float,
    seed: int | None,
) -> List[PlacedImg]:
    if not imgs:
        return []

    rng = random.Random(seed)
    target = min(len(imgs), max_cells)

    rects: List[Rect] = [Rect(0, 0, canvas.width, canvas.height)]

    def can_split(r: Rect) -> bool:
        return r.w >= 2 * min_cell or r.h >= 2 * min_cell

    while len(rects) < target:
        rects.sort(key=lambda r: r.area, reverse=True)
        idx = next((i for i, r in enumerate(rects) if can_split(r)), None)
        if idx is None:
            break
        r = rects.pop(idx)

        if r.aspect > 1.35:
            split_vert = True
        elif r.aspect < 0.75:
            split_vert = False
        else:
            split_vert = rng.random() < 0.5

        lo = 0.5 - (0.15 + split_jitter)
        hi = 0.5 + (0.15 + split_jitter)
        lo = max(0.25, lo)
        hi = min(0.75, hi)
        ratio = rng.uniform(lo, hi)

        if split_vert and r.w >= 2 * min_cell:
            w1 = int(round(r.w * ratio))
            w1 = max(min_cell, min(r.w - min_cell, w1))
            a = Rect(r.x, r.y, w1, r.h)
            b = Rect(r.x + w1, r.y, r.w - w1, r.h)
        elif (not split_vert) and r.h >= 2 * min_cell:
            h1 = int(round(r.h * ratio))
            h1 = max(min_cell, min(r.h - min_cell, h1))
            a = Rect(r.x, r.y, r.w, h1)
            b = Rect(r.x, r.y + h1, r.w, r.h - h1)
        elif r.w >= 2 * min_cell:
            w1 = max(min_cell, r.w // 2)
            a = Rect(r.x, r.y, w1, r.h)
            b = Rect(r.x + w1, r.y, r.w - w1, r.h)
        elif r.h >= 2 * min_cell:
            h1 = max(min_cell, r.h // 2)
            a = Rect(r.x, r.y, r.w, h1)
            b = Rect(r.x, r.y + h1, r.w, r.h - h1)
        else:
            rects.append(r)
            break

        rects.append(a)
        rects.append(b)

    rects = rects[:target]
    rng.shuffle(rects)

    remaining = list(imgs)
    rng.shuffle(remaining)

    def pick_for_rect(rect: Rect, k: int = 12) -> ImgInfo:
        if len(remaining) <= k:
            best_idx = 0
            best_score = float("inf")
            for i, info in enumerate(remaining):
                score = abs(math.log((info.aspect or 1.0) / (rect.aspect or 1.0)))
                if score < best_score:
                    best_score = score
                    best_idx = i
            return remaining.pop(best_idx)

        best_idx = None
        best_score = float("inf")
        for _ in range(k):
            i = rng.randrange(0, len(remaining))
            info = remaining[i]
            score = abs(math.log((info.aspect or 1.0) / (rect.aspect or 1.0)))
            if score < best_score:
                best_score = score
                best_idx = i
        assert best_idx is not None
        return remaining.pop(best_idx)

    placed: List[PlacedImg] = []
    for r in rects:
        if not remaining:
            break
        info = pick_for_rect(r)
        placed.append(PlacedImg(path=info.path, x=r.x, y=r.y, w=r.w, h=r.h))

    return placed


def build_collage(
    placed: Sequence[PlacedImg],
    canvas: CanvasSpec,
    background: Tuple[int, int, int],
    fit: str,
    pad: str,
    stats: dict[str, float] | None = None,
    forbidden_mask: Image.Image | None = None,
    forbidden_fill: Tuple[Tuple[int, int, int], Tuple[int, int, int]] | None = None,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
    workers: int = 0,
) -> Image.Image:
    out = Image.new("RGB", (canvas.width, canvas.height), color=background)

    fm: Image.Image | None = None
    if forbidden_mask is not None:
        fm = forbidden_mask
        if fm.mode != "L":
            fm = fm.convert("L")
        if fm.size != (canvas.width, canvas.height):
            fm = fm.resize((canvas.width, canvas.height), resample=Image.Resampling.NEAREST)
        if fm.getbbox() is None:
            fm = None

    cover_total = 0
    cover_cropped = 0
    cover_crop_area_sum = 0.0

    def prepare_one(p: PlacedImg):
        try:
            img = open_image(p.path)
            
            # 如果需要旋转，先旋转图片
            if p.rotate == 90:
                img = img.transpose(Image.Transpose.ROTATE_270)  # 顺时针90度
            elif p.rotate == -90:
                img = img.transpose(Image.Transpose.ROTATE_90)   # 逆时针90度
            
            ow, oh = img.size
            if ow <= 0 or oh <= 0:
                return None

            if fit == "cover":
                scale = max(p.w / ow, p.h / oh)
                new_w = max(p.w, int(math.ceil(ow * scale)))
                new_h = max(p.h, int(math.ceil(oh * scale)))
                resized = safe_resize(img, (new_w, new_h), resample=resample)

                add_total = 1
                add_cropped = 0
                add_crop_sum = 0.0
                if new_w != p.w or new_h != p.h:
                    add_cropped = 1
                    add_crop_sum = (new_w * new_h - p.w * p.h) / float(new_w * new_h)

                left = max(0, (new_w - p.w) // 2)
                top = max(0, (new_h - p.h) // 2)
                cropped = resized.crop((left, top, left + p.w, top + p.h))
                return (cropped, p.x, p.y, add_total, add_cropped, add_crop_sum)

            scale = min(p.w / ow, p.h / oh)
            new_w = max(1, int(math.floor(ow * scale)))
            new_h = max(1, int(math.floor(oh * scale)))
            if new_w > p.w:
                new_w = p.w
            if new_h > p.h:
                new_h = p.h
            resized = safe_resize(img, (new_w, new_h), resample=resample)

            if pad == "reflect":
                filled = pad_reflect(resized, p.w, p.h)
                return (filled, p.x, p.y, 0, 0, 0.0)
            if pad == "edge":
                filled = pad_repeat_edges(resized, p.w, p.h)
                return (filled, p.x, p.y, 0, 0, 0.0)

            paste_x = p.x + (p.w - new_w) // 2
            paste_y = p.y + (p.h - new_h) // 2
            return (resized, paste_x, paste_y, 0, 0, 0.0)
        except Exception:
            return None

    n_workers = _effective_workers(workers)
    if n_workers <= 1 or len(placed) <= 2:
        for p in placed:
            res = prepare_one(p)
            if res is None:
                continue
            img, x, y, add_total, add_cropped, add_crop_sum = res
            if fm is not None:
                cut = fm.crop((x, y, x + img.size[0], y + img.size[1]))
                if cut.getbbox() is None:
                    out.paste(img, (x, y))
                else:
                    out.paste(img, (x, y), mask=ImageChops.invert(cut))
            else:
                out.paste(img, (x, y))
            cover_total += add_total
            cover_cropped += add_cropped
            cover_crop_area_sum += add_crop_sum
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(prepare_one, p) for p in placed]
            for fut in as_completed(futs):
                res = fut.result()
                if res is None:
                    continue
                img, x, y, add_total, add_cropped, add_crop_sum = res
                if fm is not None:
                    cut = fm.crop((x, y, x + img.size[0], y + img.size[1]))
                    if cut.getbbox() is None:
                        out.paste(img, (x, y))
                    else:
                        out.paste(img, (x, y), mask=ImageChops.invert(cut))
                else:
                    out.paste(img, (x, y))
                cover_total += add_total
                cover_cropped += add_cropped
                cover_crop_area_sum += add_crop_sum

    # 在末端统一为禁用区域上色（可渐变）；避免多个调用点重复逻辑
    if fm is not None and forbidden_fill is not None:
        top, bot = forbidden_fill

        def _make_vertical_gradient(size: tuple[int, int], c_top: tuple[int, int, int], c_bot: tuple[int, int, int], gamma: float = 2.0) -> Image.Image:
            grad = Image.new("RGB", size)
            px = grad.load()
            h = max(1, size[1])
            for y in range(h):
                t = y / float(h - 1) if h > 1 else 0.0
                # 使用 gamma 调整，让上下差异更明显
                if gamma > 0:
                    t = max(0.0, min(1.0, t)) ** float(gamma)
                r = int(round(c_top[0] * (1 - t) + c_bot[0] * t))
                g = int(round(c_top[1] * (1 - t) + c_bot[1] * t))
                b = int(round(c_top[2] * (1 - t) + c_bot[2] * t))
                for x in range(size[0]):
                    px[x, y] = (r, g, b)
            return grad

        mask = fm
        if mask.mode != "L":
            mask = mask.convert("L")
        if mask.size != out.size:
            mask = mask.resize(out.size, resample=Image.Resampling.NEAREST)

        bbox = mask.getbbox()
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            patch_w = max(1, x1 - x0)
            patch_h = max(1, y1 - y0)
            overlay_patch = _make_vertical_gradient((patch_w, patch_h), top, bot, gamma=2.0)
            mask_patch = mask.crop(bbox)
            out.paste(overlay_patch, (x0, y0), mask=mask_patch)

    if stats is not None and fit == "cover":
        stats["cover_total"] = float(cover_total)
        stats["cover_cropped"] = float(cover_cropped)
        stats["cover_crop_area_avg"] = (cover_crop_area_sum / cover_total) if cover_total else 0.0

    return out


def collect_img_infos(paths: Sequence[Path], workers: int = 0) -> List[ImgInfo]:
    def probe(p: Path) -> ImgInfo | None:
        try:
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
                w, h = img.size
        except Exception:
            return None
        if w <= 1 or h <= 1:
            return None
        return ImgInfo(path=p, w=w, h=h)

    n_workers = _effective_workers(workers)
    if n_workers <= 1 or len(paths) <= 8:
        infos: List[ImgInfo] = []
        for p in paths:
            info = probe(p)
            if info is not None:
                infos.append(info)
        return infos

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        infos = [info for info in ex.map(probe, paths) if info is not None]
    return infos


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a photo collage that exactly fills a canvas. Choose layout and fit strategy (contain/cover) via CLI options."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=r"D:\\STUDY\\photo pack\\input",
        help="Input folder containing photos. Change this default or pass --input.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively")

    parser.add_argument(
        "--preset",
        type=str,
        default="9:16",
        help="Canvas preset: 9:16 / 2:3 (or 1080x1920 etc). Ignored if --size is provided.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Explicit canvas size like 1080x1920.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="collage.jpg",
        help="Output file path (jpg or png)",
    )

    parser.add_argument(
        "--layout",
        type=str,
        default="slice",
        choices=["slice", "bsp", "justified"],
        help="Layout algorithm. slice tries to match tile aspect to images to minimize crop; bsp creates irregular rectangles; justified is row-based.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=180,
        help="Max number of placed images (slice/bsp).",
    )

    parser.add_argument(
        "--slice-iters",
        type=int,
        default=1200,
        help="How many attempts to build a slice layout (higher may reduce crop).",
    )
    parser.add_argument(
        "--slice-tol",
        type=float,
        default=0.02,
        help="Stop early if slice layout aspect error is below this tolerance.",
    )
    parser.add_argument(
        "--slice-balance",
        type=float,
        default=0.80,
        help="[0..1] Increase to make tile areas more even (may increase cropping).",
    )

    parser.add_argument(
        "--slice-min-share",
        type=float,
        default=0.18,
        help="[0..0.49] Discourage extreme split ratios; higher makes tile sizes more even.",
    )
    parser.add_argument(
        "--min-cell",
        type=int,
        default=120,
        help="Minimum cell side in pixels (bsp only).",
    )
    parser.add_argument(
        "--split-jitter",
        type=float,
        default=0.10,
        help="BSP split randomness in [0..1].",
    )

    parser.add_argument(
        "--fit",
        type=str,
        default="cover",
        choices=["cover", "contain"],
        help="cover: scale proportionally then center-crop to fill each cell (full fill). contain: no cropping; use --pad to fill empty areas.",
    )

    parser.add_argument(
        "--pad",
        type=str,
        default="reflect",
        choices=["reflect", "edge", "background"],
        help="How to fill empty areas inside each cell when --fit=contain.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="000000",
        help="Background color in RRGGBB or #RRGGBB (used when --pad=background)",
    )

    parser.add_argument(
        "--row-height",
        type=int,
        default=220,
        help="Target row height in pixels before global scaling.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.25,
        help="Row height randomness in [0..1]. 0 means regular rows; 0.25 recommended.",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order for a more irregular look",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for reproducible results)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print cover cropping statistics to stdout",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Thread workers for image IO/resize. 0 means auto.",
    )

    args = parser.parse_args()

    canvas = parse_canvas(args.preset, args.size)

    folder = Path(args.input)
    files = iter_image_files(folder, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No images found in: {folder}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(files)

    infos = collect_img_infos(files, workers=args.workers)
    if len(infos) < 2:
        raise SystemExit("Need at least 2 readable images")

    if args.layout == "slice":
        placed = compute_layout_slice(
            infos,
            canvas=canvas,
            max_cells=max(1, int(args.max_cells)),
            slice_iters=max(1, int(args.slice_iters)),
            slice_tol=max(0.0, float(args.slice_tol)),
            slice_balance=max(0.0, min(1.0, float(args.slice_balance))),
            slice_min_share=max(0.0, min(0.49, float(args.slice_min_share))),
            seed=args.seed,
        )
    elif args.layout == "bsp":
        placed = compute_layout_bsp(
            infos,
            canvas=canvas,
            max_cells=max(1, int(args.max_cells)),
            min_cell=max(20, int(args.min_cell)),
            split_jitter=max(0.0, float(args.split_jitter)),
            seed=args.seed,
        )
    else:
        placed = compute_layout_justified(
            infos,
            canvas=canvas,
            target_row_h=args.row_height,
            row_h_jitter=max(0.0, float(args.jitter)),
            seed=args.seed,
        )

    background = parse_rgb(args.background)

    stat_map: dict[str, float] | None = {} if args.stats else None
    img = build_collage(
        placed,
        canvas=canvas,
        background=background,
        fit=args.fit,
        pad=args.pad,
        stats=stat_map,
        workers=args.workers,
    )

    if args.stats and placed:
        areas = [p.w * p.h for p in placed if p.w > 0 and p.h > 0]
        if areas:
            n = len(areas)
            avg_a = sum(areas) / n
            min_a = min(areas)
            max_a = max(areas)
            var = sum((a - avg_a) ** 2 for a in areas) / n
            cv = (math.sqrt(var) / avg_a) if avg_a else 0.0
            ratio = (max_a / min_a) if min_a else 0.0
            print(f"tiles: n={n}; max/min area={ratio:.2f}; area cv={cv:.2f}")

    if stat_map is not None and args.fit == "cover":
        total = int(stat_map.get("cover_total", 0.0))
        cropped = int(stat_map.get("cover_cropped", 0.0))
        avg = float(stat_map.get("cover_crop_area_avg", 0.0))
        pct = avg * 100.0
        print(f"cover: cropped {cropped}/{total} images; avg cropped area {pct:.2f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        img.save(out_path, quality=92, subsampling=1, optimize=True)
    else:
        img.save(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
