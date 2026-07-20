from pydantic import BaseModel


class BBox(BaseModel):
    """Normalized (0-1) box in frame coordinates, origin top-left."""

    cx: float
    cy: float
    w: float
    h: float


class TrackPoint(BaseModel):
    t: float
    bbox: BBox


class SceneEntity(BaseModel):
    id: str
    category: str
    description: str
    first_seen: float
    last_seen: float
    salience: float
    track: list[TrackPoint] = []


class AudioEvent(BaseModel):
    id: str
    label: str
    onset: float
    duration: float
    intensity: float
    source_entity_id: str | None = None


class SpeechSegment(BaseModel):
    start: float
    end: float
    text: str


class CameraMotion(BaseModel):
    t: float
    dx: float
    dy: float
    motion_magnitude: float


class SceneTimeline(BaseModel):
    video_path: str
    duration: float
    fps: float
    entities: list[SceneEntity] = []
    audio_events: list[AudioEvent] = []
    speech_segments: list[SpeechSegment] = []
    camera_motion: list[CameraMotion] = []


class Persona(BaseModel):
    name: str
    character_image_path: str
    curiosity: float = 0.6
    timidity: float = 0.3
    talkativeness: float = 0.6
    voice_style: str = "cheerful, casual"
    speech_language: str = "ja"


class Goal(BaseModel):
    title: str
    theme: str
    tone: str = "casual vlog"
    max_duration_s: float | None = None


class WorldStateFrame(BaseModel):
    t: float
    visible_entity_ids: list[str] = []
    gaze_target_id: str | None = None
    unexpected_event: bool = False
    interest: float = 0.0
    fear: float = 0.0
    narration_opportunity: str | None = None


class CharacterWorldState(BaseModel):
    frames: list[WorldStateFrame] = []


class SemanticAction(BaseModel):
    id: str
    t_start: float
    t_end: float
    action_type: str
    target_entity_id: str | None = None
    dialogue: str | None = None


class SemanticPlan(BaseModel):
    actions: list[SemanticAction] = []


class BehaviorPrimitive(BaseModel):
    id: str
    t_start: float
    t_end: float
    motion_tag: str
    gaze_target_id: str | None = None
    expression: str = "neutral"
    dialogue: str | None = None
    mirror: bool = False


class BehaviorPlan(BaseModel):
    primitives: list[BehaviorPrimitive] = []


class MotionSegment(BaseModel):
    primitive_id: str
    clip_path: str
    t_start: float
    t_end: float
    source_start: float
    source_end: float
    mirrored: bool = False


class AssembledMotion(BaseModel):
    driving_video_path: str
    segments: list[MotionSegment] = []
    total_duration: float = 0.0


class DialogueCue(BaseModel):
    t_start: float
    t_end: float
    text: str
    audio_path: str | None = None
