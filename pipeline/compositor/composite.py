from pathlib import Path

import cv2
import numpy as np

from pipeline.animate.shadow import render_shadow
from pipeline.compositor.color_harmonize import harmonize_color
from pipeline.config import settings
from pipeline.schemas import DialogueCue
from pipeline.scene_parser.depth import estimate_depth


def _composite_frame(
    bg_frame: np.ndarray,
    char_rgba: np.ndarray,
    anchor_px: tuple[int, int],
    char_h: int,
    depth_map: np.ndarray | None,
    harmonize_strength: float = 0.0,
) -> np.ndarray:
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

    full_alpha = np.zeros((h, w), dtype=np.float32)
    full_rgb = np.zeros((h, w, 3), dtype=np.float32)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        char_bgr = resized[src_y0:src_y1, src_x0:src_x1, :3]
        char_alpha_region = resized[src_y0:src_y1, src_x0:src_x1, 3]

        if harmonize_strength > 0:
            # Sample local lighting/color context from around the placement region
            # (not the whole frame) — a character standing in a sunset patch vs. a
            # shadowed patch should harmonize toward different colors.
            margin_x, margin_y = (dst_x1 - dst_x0) // 2, (dst_y1 - dst_y0) // 2
            bx0, by0 = max(dst_x0 - margin_x, 0), max(dst_y0 - margin_y, 0)
            bx1, by1 = min(dst_x1 + margin_x, w), min(dst_y1 + margin_y, h)
            bg_crop = bg_frame[by0:by1, bx0:bx1]
            char_bgr = harmonize_color(char_bgr, char_alpha_region, bg_crop, harmonize_strength)

        full_alpha[dst_y0:dst_y1, dst_x0:dst_x1] = char_alpha_region / 255.0
        full_rgb[dst_y0:dst_y1, dst_x0:dst_x1] = char_bgr

    shadow_map = render_shadow(full_alpha > 0, bg_frame)
    canvas = bg_frame.astype(np.float32) * (1.0 - shadow_map[..., None])

    if depth_map is not None:
        # Occlusion approximation: the character is placed "at" the anchor point's
        # background depth. Any background pixel nearer than that should be drawn
        # over the character instead of under it — a 2D depth-map heuristic, not
        # true 3D occlusion.
        anchor_depth = float(depth_map[min(ay, h - 1), min(ax, w - 1)])
        occluded = depth_map < (anchor_depth - 0.03)
        full_alpha = np.where(occluded, 0.0, full_alpha)

    canvas = full_rgb * full_alpha[..., None] + canvas * (1.0 - full_alpha[..., None])
    return canvas.astype(np.uint8)


def composite_silent(
    background_video_path: str,
    matte_frames_dir: Path,
    anchor_positions: list[tuple[float, float]],
    out_path: Path,
    use_depth_occlusion: bool = True,
    harmonize_strength: float | None = None,
) -> Path:
    """Alpha-blends the matted character + synthetic shadow onto the background video
    at the tracked anchor position, per frame. No audio — see `mux_final` for the
    audio/subtitle pass."""
    cap = cv2.VideoCapture(background_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {background_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    matte_paths = sorted(matte_frames_dir.glob("*.png"))
    if not matte_paths:
        raise RuntimeError(f"No matte frames in {matte_frames_dir}")

    char_h = int(h * settings.character_scale)
    strength = settings.color_harmonize_strength if harmonize_strength is None else harmonize_strength

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    ok, bg_frame = cap.read()
    while ok:
        char_rgba = cv2.imread(str(matte_paths[min(frame_idx, len(matte_paths) - 1)]), cv2.IMREAD_UNCHANGED)

        ax_norm, ay_norm = anchor_positions[min(frame_idx, len(anchor_positions) - 1)]
        anchor_px = (int(ax_norm * w), int(ay_norm * h))

        depth_map = estimate_depth(bg_frame) if use_depth_occlusion else None
        out_frame = _composite_frame(bg_frame, char_rgba, anchor_px, char_h, depth_map, strength)
        writer.write(out_frame)

        frame_idx += 1
        ok, bg_frame = cap.read()

    writer.release()
    cap.release()
    return out_path


def mux_final(
    silent_video_path: Path, background_video_path: str, dialogue_cues: list[DialogueCue], out_path: Path
) -> Path:
    """Attaches the original background audio, layers in any synthesized dialogue
    voice clips, and burns in subtitle text for cues that have no synthesized voice
    (no DASHSCOPE_API_KEY, or that specific TTS call failed) rather than dropping
    the line silently."""
    from moviepy import AudioFileClip, CompositeAudioClip, CompositeVideoClip, TextClip, VideoFileClip

    silent = VideoFileClip(str(silent_video_path))
    bg_clip = VideoFileClip(background_video_path)

    audio_tracks = []
    if bg_clip.audio is not None:
        audio_tracks.append(bg_clip.audio.subclipped(0, min(bg_clip.duration, silent.duration)))

    subtitle_clips = []
    for cue in dialogue_cues:
        if cue.audio_path:
            audio_tracks.append(AudioFileClip(cue.audio_path).with_start(cue.t_start))
        else:
            text_clip = (
                TextClip(
                    font=settings.subtitle_font_path,
                    text=cue.text,
                    font_size=42,
                    color="white",
                    stroke_color="black",
                    stroke_width=2,
                )
                .with_position(("center", 0.88), relative=True)
                .with_start(cue.t_start)
                .with_duration(max(cue.t_end - cue.t_start, 0.5))
            )
            subtitle_clips.append(text_clip)

    video = CompositeVideoClip([silent, *subtitle_clips]) if subtitle_clips else silent
    if audio_tracks:
        video = video.with_audio(CompositeAudioClip(audio_tracks))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(str(out_path), fps=silent.fps, codec="libx264", audio_codec="aac", logger=None)
    return out_path
