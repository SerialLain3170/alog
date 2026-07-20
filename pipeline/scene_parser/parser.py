from pathlib import Path

from pipeline.config import settings
from pipeline.scene_parser import audio, frames, tracking, vision_events
from pipeline.schemas import BBox, SceneEntity, SceneTimeline, TrackPoint


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.cx - a.w / 2, a.cy - a.h / 2, a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1, bx2, by2 = b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _latest_bbox(entity: SceneEntity) -> BBox:
    return entity.track[-1].bbox


def _already_tracked(label: str, bbox: BBox, entities: list[SceneEntity]) -> bool:
    """A new keyframe detection is treated as the same physical entity as an existing
    track (not re-tracked) if it shares a category and overlaps its most recent known
    position — otherwise every keyframe would spawn a duplicate of the same object."""
    for ent in entities:
        if ent.category == label and _iou(bbox, _latest_bbox(ent)) > 0.3:
            return True
    return False


def parse_scene(video_path: str, work_dir: Path) -> SceneTimeline:
    fps, duration, _ = frames.video_meta(video_path)
    camera_motion = frames.compute_camera_motion(video_path)
    keyframes = frames.sample_keyframes(video_path, settings.keyframe_interval_s)

    frames_dir = work_dir / "frames"
    width, height, _ = tracking.extract_frames_to_dir(video_path, frames_dir)

    entities: list[SceneEntity] = []
    entity_idx = 0
    for t, frame in keyframes:
        detection = vision_events.detect_keyframe_entities(frame)
        seed_frame_idx = int(round(t * fps))

        candidates = [e for e in detection["entities"] if e["salience"] >= settings.entity_salience_floor]
        candidates.sort(key=lambda e: e["salience"], reverse=True)

        for e in candidates:
            if len(entities) >= settings.max_tracked_entities:
                break

            seed_bbox = BBox(cx=e["cx"], cy=e["cy"], w=e["w"], h=e["h"])
            if _already_tracked(e["label"], seed_bbox, entities):
                continue

            entity_id = f"entity_{entity_idx}"
            entity_idx += 1
            try:
                raw_track = tracking.track_entity(frames_dir, seed_frame_idx, seed_bbox, (width, height))
            except Exception:
                raw_track = {seed_frame_idx: seed_bbox}

            track_points = [TrackPoint(t=idx / fps, bbox=box) for idx, box in sorted(raw_track.items())]
            if not track_points:
                track_points = [TrackPoint(t=t, bbox=seed_bbox)]

            entities.append(
                SceneEntity(
                    id=entity_id,
                    category=e["label"],
                    description=e["description"],
                    first_seen=track_points[0].t,
                    last_seen=track_points[-1].t,
                    salience=e["salience"],
                    track=track_points,
                )
            )

    audio_events, speech_segments = audio.analyze_audio(video_path, work_dir)
    _correlate_audio_direction(audio_events, entities)

    return SceneTimeline(
        video_path=video_path,
        duration=duration,
        fps=fps,
        entities=entities,
        audio_events=audio_events,
        speech_segments=speech_segments,
        camera_motion=camera_motion,
    )


def _correlate_audio_direction(audio_events, entities: list[SceneEntity]) -> None:
    """Heuristic sound-direction proxy, not real audio localization (source audio is
    mono/stereo): if a salient entity first appears within a short window of an audio
    event's onset, treat the event as sourced from that entity."""
    window_s = 0.6
    for ev in audio_events:
        best, best_dt = None, window_s
        for ent in entities:
            dt = abs(ent.first_seen - ev.onset)
            if dt < best_dt:
                best, best_dt = ent, dt
        if best is not None:
            ev.source_entity_id = best.id
