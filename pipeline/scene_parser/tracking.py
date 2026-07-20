import functools
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

from pipeline.config import settings
from pipeline.schemas import BBox


def _ensure_checkpoint() -> Path:
    ckpt_dir = settings.model_cache_dir / "sam2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / Path(settings.sam2_checkpoint_url).name
    if not ckpt_path.exists():
        resp = requests.get(settings.sam2_checkpoint_url, stream=True, timeout=300)
        resp.raise_for_status()
        with ckpt_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return ckpt_path


@functools.lru_cache(maxsize=1)
def _predictor():
    from sam2.build_sam import build_sam2_video_predictor

    ckpt_path = _ensure_checkpoint()
    return build_sam2_video_predictor(settings.sam2_config_name, str(ckpt_path))


def extract_frames_to_dir(video_path: str, out_dir: Path) -> tuple[int, int, int]:
    """Extracts every frame of `video_path` to `out_dir` as 00000.jpg, 00001.jpg, ...
    (the layout SAM2's video predictor expects). Returns (width, height, frame_count)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = height = 0
    idx = 0
    ok, frame = cap.read()
    while ok:
        height, width = frame.shape[:2]
        cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
        ok, frame = cap.read()
    cap.release()
    return width, height, idx


def segment_entity(
    frames_dir: Path, seed_frame_idx: int, seed_bbox: BBox, frame_size: tuple[int, int]
) -> dict[int, np.ndarray]:
    """Runs SAM2 video segmentation across a directory of extracted JPEG frames (see
    `extract_frames_to_dir`), seeded by a normalized bounding box on `seed_frame_idx`.
    Returns {frame_idx: boolean mask} for every frame the object is visible on. Shared
    by entity tracking (bbox derived from the mask) and character matting (the mask
    itself becomes the alpha channel)."""
    predictor = _predictor()
    w, h = frame_size

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path=str(frames_dir))

        # This SAM2 commit (pinned by Wan2.2's requirements_animate.txt) predates
        # add_new_points_or_box — seed with point prompts derived from the bbox
        # instead: center plus a small ring of inset points, all positive, so a
        # single bad point (e.g. landing on dark clothing near the background
        # color) can't fragment the mask.
        cx, cy = seed_bbox.cx * w, seed_bbox.cy * h
        dx, dy = seed_bbox.w * w * 0.3, seed_bbox.h * h * 0.3
        points = np.array(
            [
                [cx, cy],
                [cx - dx, cy - dy],
                [cx + dx, cy - dy],
                [cx - dx, cy + dy],
                [cx + dx, cy + dy],
            ],
            dtype=np.float32,
        )
        labels = np.array([1, 1, 1, 1, 1], dtype=np.int32)

        predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=seed_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )

        masks: dict[int, np.ndarray] = {}
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            masks[frame_idx] = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
    return masks


def track_entity(
    frames_dir: Path, seed_frame_idx: int, seed_bbox: BBox, frame_size: tuple[int, int]
) -> dict[int, BBox]:
    """Tracks a single entity, returning {frame_idx: BBox} for every frame it's
    visible on (bounding box of `segment_entity`'s mask)."""
    w, h = frame_size
    tracks: dict[int, BBox] = {}
    for frame_idx, mask in segment_entity(frames_dir, seed_frame_idx, seed_bbox, frame_size).items():
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        tracks[frame_idx] = BBox(
            cx=((x1 + x2) / 2) / w,
            cy=((y1 + y2) / 2) / h,
            w=(x2 - x1) / w,
            h=(y2 - y1) / h,
        )
    return tracks
