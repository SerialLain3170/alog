from pathlib import Path

import cv2
import numpy as np

from pipeline.compositor.color_harmonize import harmonize_color
from pipeline.config import settings
from pipeline.scene_parser.depth import estimate_depth


def _depth_to_magma(depth_0to1: np.ndarray) -> np.ndarray:
    """Colorizes a 0(near)-1(far) depth map with the same magma-style palette
    PISCO's own precomputed depth videos use — confirmed by cross-referencing a
    known near/far cue (an awning projecting toward camera, so nearer than the
    wall behind it) in their example data against this exact colormap."""
    vis = (np.clip(depth_0to1, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)


def prepare_pisco_inputs(
    background_video_path: str,
    start_frame: int,
    char_rgba: np.ndarray,
    anchor_positions: list[tuple[float, float]],
    anchor_frame_indices: list[int],
    work_dir: Path,
) -> Path:
    """Builds one PISCO input package (clean.mp4, clean_depth.mp4, mask/,
    video_masked/, video_depth_masked/) for a settings.pisco_num_frames-length
    chunk starting at `start_frame` of the background video. `anchor_positions`
    gives the character's normalized (x, y) feet position for every frame in the
    chunk (same convention as compositor.anchor.track_anchor); only frames listed
    in `anchor_frame_indices` get real sparse content written — PISCO propagates
    the character across the rest of the chunk itself."""
    num_frames = settings.pisco_num_frames
    w, h = settings.pisco_width, settings.pisco_height
    char_h = int(h * settings.character_scale)

    for sub in ("mask", "video_masked", "video_depth_masked"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(background_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {background_video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    clean_writer = cv2.VideoWriter(str(work_dir / "clean.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    depth_writer = cv2.VideoWriter(str(work_dir / "clean_depth.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))

    for i in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Background video ran out of frames at chunk offset {i}")
        frame_resized = cv2.resize(frame, (w, h))
        clean_writer.write(frame_resized)

        depth = estimate_depth(frame_resized)
        depth_writer.write(_depth_to_magma(depth))

        if i in anchor_frame_indices:
            ax_n, ay_n = anchor_positions[i]
            anchor_px = (int(ax_n * w), int(ay_n * h))
            _write_sparse_anchor(work_dir, i, frame_resized, depth, char_rgba, anchor_px, char_h)

    clean_writer.release()
    depth_writer.release()
    cap.release()
    return work_dir


def _write_sparse_anchor(
    work_dir: Path,
    frame_idx: int,
    bg_frame: np.ndarray,
    bg_depth: np.ndarray,
    char_rgba: np.ndarray,
    anchor_px: tuple[int, int],
    char_h: int,
) -> None:
    """Writes the sparse mask/video_masked/video_depth_masked PNGs for one anchor
    frame — the character's own content isolated on a black canvas (not blended
    with the background; PISCO does that itself)."""
    h, w = bg_frame.shape[:2]
    ch, cw = char_rgba.shape[:2]
    scale = char_h / ch
    new_w, new_h = max(int(cw * scale), 1), max(int(ch * scale), 1)
    resized = cv2.resize(char_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ax, ay = anchor_px
    x0, y0 = ax - new_w // 2, ay - new_h  # anchor = character's feet

    src_x0, src_y0 = max(-x0, 0), max(-y0, 0)
    dst_x0, dst_y0 = max(x0, 0), max(y0, 0)
    dst_x1, dst_y1 = min(x0 + new_w, w), min(y0 + new_h, h)
    src_x1, src_y1 = src_x0 + max(dst_x1 - dst_x0, 0), src_y0 + max(dst_y1 - dst_y0, 0)

    mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    masked_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    depth_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        crop = resized[src_y0:src_y1, src_x0:src_x1]
        alpha = crop[:, :, 3]
        char_bgr = crop[:, :, :3]

        margin_x, margin_y = (dst_x1 - dst_x0) // 2, (dst_y1 - dst_y0) // 2
        bx0, by0 = max(dst_x0 - margin_x, 0), max(dst_y0 - margin_y, 0)
        bx1, by1 = min(dst_x1 + margin_x, w), min(dst_y1 + margin_y, h)
        char_bgr = harmonize_color(char_bgr, alpha, bg_frame[by0:by1, bx0:bx1], settings.color_harmonize_strength)

        binary_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        mask_canvas[dst_y0:dst_y1, dst_x0:dst_x1] = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        masked_canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.where(alpha[..., None] > 0, char_bgr, 0)

        # Uniform placement depth: the character sits at the anchor point's own
        # local background depth (same convention as the compositor's occlusion
        # heuristic) — not a per-pixel body depth, which a flat 2D cutout doesn't
        # meaningfully have.
        anchor_depth_val = float(bg_depth[min(ay, h - 1), min(ax, w - 1)])
        depth_patch = np.full((dst_y1 - dst_y0, dst_x1 - dst_x0), anchor_depth_val, dtype=np.float32)
        depth_patch_colored = _depth_to_magma(depth_patch)
        depth_canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.where(alpha[..., None] > 0, depth_patch_colored, 0)

    cv2.imwrite(str(work_dir / "mask" / f"{frame_idx:05d}.png"), mask_canvas)
    cv2.imwrite(str(work_dir / "video_masked" / f"{frame_idx:05d}.png"), masked_canvas)
    cv2.imwrite(str(work_dir / "video_depth_masked" / f"{frame_idx:05d}.png"), depth_canvas)
