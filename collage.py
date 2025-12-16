from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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


@dataclass
class PlacedImg:
    path: Path
    x: int
    y: int
    w: int
    h: int


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
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    out = Image.new("RGB", (canvas.width, canvas.height), color=background)

    cover_total = 0
    cover_cropped = 0
    cover_crop_area_sum = 0.0

    for p in placed:
        img = open_image(p.path)
        ow, oh = img.size
        if ow <= 0 or oh <= 0:
            continue

        if fit == "cover":
            scale = max(p.w / ow, p.h / oh)
            new_w = max(p.w, int(math.ceil(ow * scale)))
            new_h = max(p.h, int(math.ceil(oh * scale)))
            resized = safe_resize(img, (new_w, new_h), resample=resample)

            cover_total += 1
            if new_w != p.w or new_h != p.h:
                cover_cropped += 1
                cover_crop_area_sum += (new_w * new_h - p.w * p.h) / float(new_w * new_h)

            left = max(0, (new_w - p.w) // 2)
            top = max(0, (new_h - p.h) // 2)
            cropped = resized.crop((left, top, left + p.w, top + p.h))
            out.paste(cropped, (p.x, p.y))
        else:
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
                out.paste(filled, (p.x, p.y))
            elif pad == "edge":
                filled = pad_repeat_edges(resized, p.w, p.h)
                out.paste(filled, (p.x, p.y))
            else:
                paste_x = p.x + (p.w - new_w) // 2
                paste_y = p.y + (p.h - new_h) // 2
                out.paste(resized, (paste_x, paste_y))

    if stats is not None and fit == "cover":
        stats["cover_total"] = float(cover_total)
        stats["cover_cropped"] = float(cover_cropped)
        stats["cover_crop_area_avg"] = (cover_crop_area_sum / cover_total) if cover_total else 0.0

    return out


def collect_img_infos(paths: Sequence[Path]) -> List[ImgInfo]:
    infos: List[ImgInfo] = []
    for p in paths:
        try:
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
                w, h = img.size
        except Exception:
            continue
        if w <= 1 or h <= 1:
            continue
        infos.append(ImgInfo(path=p, w=w, h=h))
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

    args = parser.parse_args()

    canvas = parse_canvas(args.preset, args.size)

    folder = Path(args.input)
    files = iter_image_files(folder, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No images found in: {folder}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(files)

    infos = collect_img_infos(files)
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
