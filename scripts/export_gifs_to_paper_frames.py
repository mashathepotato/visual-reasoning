#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ExportResult:
    gif_path: Path
    n_frames_total: int
    frame_indices: list[int]
    out_path: Path
    written: bool


def _unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _select_evenly_spaced_indices(
    start: int,
    end_inclusive: int,
    *,
    n_select: int,
) -> list[int]:
    if n_select <= 0:
        return []
    if end_inclusive < start:
        return []
    n_available = end_inclusive - start + 1
    if n_available <= n_select:
        return list(range(start, end_inclusive + 1))

    indices = np.linspace(start, end_inclusive, n_select)
    indices = np.rint(indices).astype(int).tolist()
    indices = _unique_in_order([int(i) for i in indices])
    if indices and indices[0] != start:
        indices.insert(0, start)
    if indices and indices[-1] != end_inclusive:
        indices.append(end_inclusive)
    return indices


def _parse_rgb(rgb: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in rgb.split(",")]
    if len(parts) != 3:
        raise ValueError("expected format like '255,255,255'")
    rgb_ints: list[int] = []
    for part in parts:
        val = int(part)
        if val < 0 or val > 255:
            raise ValueError("RGB values must be in [0, 255]")
        rgb_ints.append(val)
    return rgb_ints[0], rgb_ints[1], rgb_ints[2]


def _resample_from_name(name: str) -> int:
    normalized = name.strip().lower()
    if normalized == "nearest":
        return Image.Resampling.NEAREST
    if normalized == "bilinear":
        return Image.Resampling.BILINEAR
    if normalized == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError(f"Unsupported resample method: {name}")


def _normalize_frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        rgb = np.repeat(frame[:, :, None], 3, axis=2)
        return rgb.astype(np.uint8, copy=False)
    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    if frame.shape[2] == 3:
        return frame.astype(np.uint8, copy=False)
    if frame.shape[2] == 4:
        return frame[:, :, :3].astype(np.uint8, copy=False)
    if frame.shape[2] == 1:
        return np.repeat(frame, 3, axis=2).astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported channel count: {frame.shape}")


def _split_left_right_if_side_by_side(
    frame_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = frame_rgb.shape[0], frame_rgb.shape[1]
    if width != 2 * height:
        return None
    left = frame_rgb[:, :height, :]
    right = frame_rgb[:, height : 2 * height, :]
    return left, right


def _load_font(font_size: int, font_path: Path | None) -> ImageFont.ImageFont:
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=font_size)
        except Exception as e:
            raise SystemExit(f"Failed to load font at {font_path}: {e}") from e

    try:
        # Pillow typically ships with DejaVu fonts.
        import PIL  # type: ignore

        pil_dir = Path(PIL.__file__).resolve().parent
        candidate = pil_dir / "fonts" / "DejaVuSans.ttf"
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=font_size)
    except Exception:
        pass

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except Exception:
        return ImageFont.load_default()


def _make_labeled_rollout_row_figure(
    images: list[Image.Image],
    labels: list[str],
    *,
    scale: int,
    pad: int,
    margin: int,
    label_pad: int,
    background_rgb: tuple[int, int, int],
    resample: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    if not images:
        raise ValueError("images is empty")
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if pad < 0:
        raise ValueError("pad must be >= 0")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if label_pad < 0:
        raise ValueError("label_pad must be >= 0")

    resized_images: list[Image.Image] = []
    for img in images:
        img_rgb = img.convert("RGB")
        if scale != 1:
            width, height = img_rgb.size
            img_rgb = img_rgb.resize((width * scale, height * scale), resample=resample)
        resized_images.append(img_rgb)

    cell_width, cell_height = resized_images[0].size
    for img in resized_images[1:]:
        if img.size != (cell_width, cell_height):
            raise ValueError(
                "All images must have the same size after optional scaling. "
                f"Expected {(cell_width, cell_height)}, got {img.size}."
            )

    dummy = Image.new("RGB", (1, 1), background_rgb)
    dummy_draw = ImageDraw.Draw(dummy)
    max_text_height = 0
    for label in labels:
        bbox = dummy_draw.textbbox((0, 0), label, font=font)
        max_text_height = max(max_text_height, bbox[3] - bbox[1])
    label_height = max_text_height + 2 * label_pad

    n = len(resized_images)
    canvas_width = margin * 2 + n * cell_width + max(0, n - 1) * pad
    canvas_height = margin * 2 + label_height + cell_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), background_rgb)
    draw = ImageDraw.Draw(canvas)

    for idx, (img, label) in enumerate(zip(resized_images, labels)):
        x = margin + idx * (cell_width + pad)
        y_img = margin + label_height
        canvas.paste(img, (x, y_img))

        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x_text = x + (cell_width - text_width) // 2
        y_text = margin + (label_height - text_height) // 2
        draw.text((x_text, y_text), label, fill=(0, 0, 0), font=font)

    return canvas


def export_gif_rollout_figure(
    gif_path: Path,
    *,
    gifs_dir: Path,
    out_dir: Path,
    n_steps: int,
    scale: int,
    pad: int,
    margin: int,
    label_pad: int,
    background_rgb: tuple[int, int, int],
    resample: int,
    font: ImageFont.ImageFont,
    split_side_by_side: bool,
    overwrite: bool,
) -> ExportResult:
    rel = gif_path.resolve().relative_to(gifs_dir.resolve())
    subtag = rel.parent.as_posix().replace("/", "_")
    prefix = f"{subtag}__{gif_path.stem}" if subtag else gif_path.stem

    frames = imageio.mimread(gif_path)
    n_frames_total = len(frames)
    if n_frames_total <= 0:
        raise ValueError(f"{gif_path} has no frames")

    first_rgb = _normalize_frame_to_rgb(frames[0])
    split = _split_left_right_if_side_by_side(first_rgb) if split_side_by_side else None
    has_target_panel = split is not None

    if has_target_panel:
        left0, right0 = split  # type: ignore[misc]
        input_img = Image.fromarray(left0)
        target_img = Image.fromarray(right0)

        step_indices = _select_evenly_spaced_indices(
            1,
            n_frames_total - 1,
            n_select=n_steps,
        )
        step_imgs: list[Image.Image] = []
        for frame_idx in step_indices:
            fr_rgb = _normalize_frame_to_rgb(frames[frame_idx])
            left_right = _split_left_right_if_side_by_side(fr_rgb)
            if left_right is None:
                raise ValueError(
                    f"{gif_path}: expected side-by-side frame at index {frame_idx}"
                )
            step_imgs.append(Image.fromarray(left_right[0]))

        images = [input_img] + step_imgs + [target_img]
        labels = ["Input"] + [f"Step {i}" for i in range(1, len(step_imgs) + 1)] + [
            "Target"
        ]
        frame_indices = [0] + step_indices
    else:
        input_img = Image.fromarray(first_rgb)
        last_rgb = (
            _normalize_frame_to_rgb(frames[-1]) if n_frames_total > 1 else first_rgb
        )
        target_img = Image.fromarray(last_rgb)

        step_indices = _select_evenly_spaced_indices(
            1,
            n_frames_total - 2,
            n_select=n_steps,
        )
        step_imgs = [
            Image.fromarray(_normalize_frame_to_rgb(frames[i])) for i in step_indices
        ]
        images = [input_img] + step_imgs + [target_img]
        labels = ["Input"] + [f"Step {i}" for i in range(1, len(step_imgs) + 1)] + [
            "Target"
        ]
        frame_indices = (
            [0]
            + step_indices
            + ([n_frames_total - 1] if n_frames_total > 1 else [0])
        )

    out_path = out_dir / f"{prefix}.png"
    if out_path.exists() and not overwrite:
        return ExportResult(
            gif_path=gif_path,
            n_frames_total=n_frames_total,
            frame_indices=frame_indices,
            out_path=out_path,
            written=False,
        )

    figure = _make_labeled_rollout_row_figure(
        images,
        labels,
        scale=scale,
        pad=pad,
        margin=margin,
        label_pad=label_pad,
        background_rgb=background_rgb,
        resample=resample,
        font=font,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(out_path, format="PNG", optimize=True)

    return ExportResult(
        gif_path=gif_path,
        n_frames_total=n_frames_total,
        frame_indices=frame_indices,
        out_path=out_path,
        written=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export GIF rollouts into one labeled PNG per GIF (Input, Step 1..N, Target)."
        )
    )
    parser.add_argument("--gifs-dir", type=Path, default=Path("gifs"))
    parser.add_argument("--out-dir", type=Path, default=Path("examples/paper_figures"))
    parser.add_argument(
        "--frames-per-figure",
        type=int,
        default=None,
        help="(Deprecated) Total tiles = input + steps + target. Overrides --n-steps.",
    )
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--pad", type=int, default=12)
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument("--label-pad", type=int, default=6)
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--background", type=str, default="255,255,255")
    parser.add_argument(
        "--no-split-side-by-side",
        action="store_true",
        help="Disable auto-splitting wide frames (e.g., [state|target]) into separate Input/Target tiles.",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="nearest",
        choices=["nearest", "bilinear", "bicubic"],
        help="Resampling method used when --scale > 1.",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    gifs_dir: Path = args.gifs_dir
    out_dir: Path = args.out_dir
    frames_per_figure: int | None = args.frames_per_figure
    n_steps: int = args.n_steps
    scale: int = args.scale
    pad: int = args.pad
    margin: int = args.margin
    label_pad: int = args.label_pad
    font_size: int = args.font_size
    font_path: Path | None = args.font_path
    background_rgb = _parse_rgb(args.background)
    resample = _resample_from_name(args.resample)
    font = _load_font(font_size=font_size, font_path=font_path)
    split_side_by_side = not args.no_split_side_by_side
    overwrite: bool = args.overwrite

    if frames_per_figure is not None:
        if frames_per_figure < 2:
            raise SystemExit("--frames-per-figure must be >= 2")
        n_steps = max(frames_per_figure - 2, 0)
    if n_steps < 0:
        raise SystemExit("--n-steps must be >= 0")
    if scale <= 0:
        raise SystemExit("--scale must be > 0")
    if pad < 0:
        raise SystemExit("--pad must be >= 0")
    if margin < 0:
        raise SystemExit("--margin must be >= 0")
    if label_pad < 0:
        raise SystemExit("--label-pad must be >= 0")
    if font_size <= 0:
        raise SystemExit("--font-size must be > 0")

    gif_paths = sorted(gifs_dir.rglob("*.gif"))
    if not gif_paths:
        raise SystemExit(f"No GIFs found under: {gifs_dir}")

    results: list[ExportResult] = []
    for gif_path in gif_paths:
        res = export_gif_rollout_figure(
            gif_path,
            gifs_dir=gifs_dir,
            out_dir=out_dir,
            n_steps=n_steps,
            scale=scale,
            pad=pad,
            margin=margin,
            label_pad=label_pad,
            background_rgb=background_rgb,
            resample=resample,
            font=font,
            split_side_by_side=split_side_by_side,
            overwrite=overwrite,
        )
        results.append(res)
        print(
            f"{gif_path}: total_frames={res.n_frames_total} "
            f"shown={len(res.frame_indices)} written={res.written} -> {res.out_path}"
        )

    total_written = sum(1 for r in results if r.written)
    print(f"Done. Wrote {total_written} rollout figures to {out_dir}")


if __name__ == "__main__":
    main()
