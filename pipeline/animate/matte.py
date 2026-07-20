from pathlib import Path

import cv2

from pipeline.scene_parser.tracking import extract_frames_to_dir, segment_entity
from pipeline.schemas import BBox


def matte_character(raw_video_path: str, work_dir: Path) -> Path:
    """Runs SAM2 on the raw Wan-Animate ANIMATION-mode output — the character is the
    single dominant, centered subject there by construction — and writes RGBA PNG
    frames with the character isolated on transparency. Returns the output frames
    directory."""
    frames_dir = work_dir / "raw_frames"
    width, height, _ = extract_frames_to_dir(raw_video_path, frames_dir)

    # Character fills most of the frame height, roughly centered horizontally.
    seed_bbox = BBox(cx=0.5, cy=0.55, w=0.5, h=0.85)
    masks = segment_entity(frames_dir, seed_frame_idx=0, seed_bbox=seed_bbox, frame_size=(width, height))

    out_dir = work_dir / "matte_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, mask in masks.items():
        frame = cv2.imread(str(frames_dir / f"{frame_idx:05d}.jpg"))
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (mask.astype("uint8")) * 255
        cv2.imwrite(str(out_dir / f"{frame_idx:05d}.png"), rgba)

    return out_dir
