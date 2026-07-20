from pathlib import Path

from moviepy import VideoFileClip, concatenate_videoclips, vfx

from pipeline.motion.library import get_clip_for_tag
from pipeline.schemas import AssembledMotion, BehaviorPlan, MotionSegment


def _fit_duration(clip: VideoFileClip, target_duration: float) -> VideoFileClip:
    if clip.duration >= target_duration:
        return clip.subclipped(0, target_duration)
    return clip.with_effects([vfx.Loop(duration=target_duration)])


def assemble_driving_video(behavior_plan: BehaviorPlan, work_dir: Path) -> AssembledMotion:
    """Selects/retimes/concatenates motion-library clips per the behavior plan into
    one continuous driving video for Wan2.2-Animate."""
    clips: list[VideoFileClip] = []
    segments: list[MotionSegment] = []
    cursor = 0.0

    for primitive in behavior_plan.primitives:
        duration = primitive.t_end - primitive.t_start
        if duration <= 0:
            continue

        entry = get_clip_for_tag(primitive.motion_tag)
        source_clip = VideoFileClip(entry["path"])
        fitted = _fit_duration(source_clip, duration)
        if primitive.mirror:
            fitted = fitted.with_effects([vfx.MirrorX()])

        clips.append(fitted)
        segments.append(
            MotionSegment(
                primitive_id=primitive.id,
                clip_path=entry["path"],
                t_start=cursor,
                t_end=cursor + duration,
                source_start=0.0,
                source_end=min(duration, source_clip.duration),
                mirrored=primitive.mirror,
            )
        )
        cursor += duration

    if not clips:
        raise RuntimeError("Behavior plan produced no motion segments to assemble")

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "driving_video.mp4"
    final = concatenate_videoclips(clips)
    final.write_videofile(str(out_path), fps=24, codec="libx264", audio=False, logger=None)

    return AssembledMotion(driving_video_path=str(out_path), segments=segments, total_duration=cursor)
