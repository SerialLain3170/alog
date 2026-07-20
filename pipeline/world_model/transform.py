from pipeline.config import settings
from pipeline.schemas import CharacterWorldState, Persona, SceneEntity, SceneTimeline, WorldStateFrame

_NEW_ENTITY_WINDOW_S = 1.0
_LOUD_AUDIO_WINDOW_S = 0.5
_LOUD_AUDIO_INTENSITY = 0.5
_NARRATION_SALIENCE_FLOOR = 0.5
_NARRATION_TALKATIVENESS_FLOOR = 0.35


def _visible_entities(entities: list[SceneEntity], t: float) -> list[SceneEntity]:
    return [e for e in entities if e.first_seen <= t <= e.last_seen]


def _pick_gaze_target(visible: list[SceneEntity], t: float, persona: Persona) -> SceneEntity | None:
    if not visible:
        return None

    def score(e: SceneEntity) -> float:
        novelty = 1.0 if (t - e.first_seen) < _NEW_ENTITY_WINDOW_S else 0.0
        return e.salience * (1.0 + persona.curiosity) + novelty * persona.curiosity

    return max(visible, key=score)


def build_world_state(timeline: SceneTimeline, persona: Persona) -> CharacterWorldState:
    frames: list[WorldStateFrame] = []
    t = 0.0
    while t <= timeline.duration:
        visible = _visible_entities(timeline.entities, t)
        gaze_target = _pick_gaze_target(visible, t, persona)

        just_appeared = any((t - e.first_seen) < _NEW_ENTITY_WINDOW_S for e in visible)
        loud_events = [
            ev
            for ev in timeline.audio_events
            if abs(ev.onset - t) < _LOUD_AUDIO_WINDOW_S and ev.intensity >= _LOUD_AUDIO_INTENSITY
        ]
        unexpected = just_appeared or bool(loud_events)

        interest = persona.curiosity * (gaze_target.salience if gaze_target else 0.0)
        fear = persona.timidity * max((ev.intensity for ev in loud_events), default=0.0)

        narration_opportunity = None
        if (
            gaze_target is not None
            and gaze_target.salience >= _NARRATION_SALIENCE_FLOOR
            and persona.talkativeness >= _NARRATION_TALKATIVENESS_FLOOR
        ):
            narration_opportunity = f"{gaze_target.category}: {gaze_target.description}"

        frames.append(
            WorldStateFrame(
                t=t,
                visible_entity_ids=[e.id for e in visible],
                gaze_target_id=gaze_target.id if gaze_target else None,
                unexpected_event=unexpected,
                interest=interest,
                fear=fear,
                narration_opportunity=narration_opportunity,
            )
        )
        t += settings.world_state_interval_s

    return CharacterWorldState(frames=frames)
