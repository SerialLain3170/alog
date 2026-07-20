from pipeline.schemas import BehaviorPlan, BehaviorPrimitive, SceneEntity, SemanticPlan

ACTION_TO_MOTION_TAG = {
    "idle": "idle_stand",
    "look_at": "turn_head",
    "walk_to": "walk_forward",
    "comment": "idle_stand",
    "pick_up": "pick_up_object",
    "sit": "sit_down",
    "wave": "wave",
    "startle": "startle_flinch",
}

ACTION_TO_EXPRESSION = {
    "idle": "neutral",
    "look_at": "curious",
    "walk_to": "neutral",
    "comment": "talking",
    "pick_up": "focused",
    "sit": "relaxed",
    "wave": "happy",
    "startle": "surprised",
}


def _should_mirror(target: SceneEntity | None) -> bool:
    """Motion library clips default to facing/moving screen-right. Mirror the clip
    when the target entity sits in the left half of frame so the character at least
    faces the right way, even though we don't have true 3D gaze control."""
    return target is not None and target.track and target.track[0].bbox.cx < 0.5


def build_behavior_plan(semantic_plan: SemanticPlan, entities_by_id: dict[str, SceneEntity]) -> BehaviorPlan:
    actions = sorted(semantic_plan.actions, key=lambda a: a.t_start)

    primitives: list[BehaviorPrimitive] = []
    for i, action in enumerate(actions):
        t_end = action.t_end
        if i + 1 < len(actions):
            t_end = min(t_end, actions[i + 1].t_start)
        if t_end <= action.t_start:
            continue

        target = entities_by_id.get(action.target_entity_id) if action.target_entity_id else None
        primitives.append(
            BehaviorPrimitive(
                id=action.id,
                t_start=action.t_start,
                t_end=t_end,
                motion_tag=ACTION_TO_MOTION_TAG.get(action.action_type, "idle_stand"),
                gaze_target_id=action.target_entity_id,
                expression=ACTION_TO_EXPRESSION.get(action.action_type, "neutral"),
                dialogue=action.dialogue,
                mirror=_should_mirror(target),
            )
        )

    return BehaviorPlan(primitives=primitives)
