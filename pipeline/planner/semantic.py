from anthropic import Anthropic

from pipeline.config import require_anthropic_key, settings
from pipeline.schemas import CharacterWorldState, Goal, Persona, SceneTimeline, SemanticAction, SemanticPlan

_ACTION_TYPES = ["idle", "look_at", "walk_to", "comment", "pick_up", "sit", "wave", "startle"]

_SYSTEM_PROMPT = (
    "You are the behavior-planning module for a 2D anime character being composited "
    "into real vlog footage as an autonomous on-screen presence. Given what the "
    "character can see/hear over time (a world-state timeline), the character's "
    "persona, and the creator's goal for this vlog, decide what the character "
    "actually does: an ordered sequence of semantic actions with start/end "
    "timestamps that fit within the video's real duration and do not overlap in "
    "time.\n\n"
    f"Available action_type values: {', '.join(_ACTION_TYPES)}.\n"
    "- look_at / walk_to / pick_up / sit / wave / startle should reference a "
    "target_entity_id from the provided entity list when relevant (idle/comment "
    "may leave it null).\n"
    "- comment actions must include short, natural dialogue in the persona's "
    "speech language, matching their talkativeness and personality.\n"
    "- Keep the pacing believable for the persona (a timid character reacts less "
    "often and more subtly than a curious one).\n"
    "- Do not invent entities or timestamps outside what's given."
)

_TOOL = {
    "name": "report_plan",
    "description": "Report the ordered semantic action plan for the character.",
    "input_schema": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "t_start": {"type": "number"},
                        "t_end": {"type": "number"},
                        "action_type": {"type": "string", "enum": _ACTION_TYPES},
                        "target_entity_id": {"type": ["string", "null"]},
                        "dialogue": {"type": ["string", "null"]},
                    },
                    "required": ["t_start", "t_end", "action_type", "target_entity_id", "dialogue"],
                },
            }
        },
        "required": ["actions"],
    },
}


def _build_context(
    timeline: SceneTimeline, world_state: CharacterWorldState, persona: Persona, goal: Goal
) -> str:
    entity_lines = "\n".join(
        f"- {e.id} ({e.category}): {e.description}, visible {e.first_seen:.1f}s-{e.last_seen:.1f}s, "
        f"salience={e.salience:.2f}"
        for e in timeline.entities
    ) or "(none detected)"

    audio_lines = "\n".join(
        f"- t={ev.onset:.1f}s label={ev.label} intensity={ev.intensity:.2f}"
        for ev in timeline.audio_events
        if ev.intensity >= 0.3
    ) or "(none)"

    world_lines = "\n".join(
        f"- t={f.t:.1f}s gaze={f.gaze_target_id} interest={f.interest:.2f} fear={f.fear:.2f} "
        f"unexpected={f.unexpected_event} narration_hint={f.narration_opportunity}"
        for f in world_state.frames
    )

    return (
        f"Video duration: {timeline.duration:.1f}s\n\n"
        f"Persona: {persona.name} (curiosity={persona.curiosity}, timidity={persona.timidity}, "
        f"talkativeness={persona.talkativeness}, speech_language={persona.speech_language})\n\n"
        f"Vlog goal: {goal.title} — {goal.theme} (tone: {goal.tone})\n\n"
        f"Entities:\n{entity_lines}\n\n"
        f"Notable audio events:\n{audio_lines}\n\n"
        f"Character world-state over time:\n{world_lines}\n"
    )


def plan_semantic_actions(
    timeline: SceneTimeline, world_state: CharacterWorldState, persona: Persona, goal: Goal
) -> SemanticPlan:
    client = Anthropic(api_key=require_anthropic_key())
    context = _build_context(timeline, world_state, persona, goal)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_plan"},
        messages=[{"role": "user", "content": context}],
    )

    for block in response.content:
        if block.type == "tool_use":
            actions = [SemanticAction(id=f"action_{i}", **a) for i, a in enumerate(block.input["actions"])]
            return SemanticPlan(actions=actions)
    raise RuntimeError("Claude did not return a tool_use block")
