import sys
from pathlib import Path

from pipeline.config import settings
from pipeline.schemas import BehaviorPlan, DialogueCue, Persona


def extract_dialogue_cues(behavior_plan: BehaviorPlan) -> list[DialogueCue]:
    return [
        DialogueCue(t_start=p.t_start, t_end=p.t_end, text=p.dialogue)
        for p in behavior_plan.primitives
        if p.dialogue
    ]


def synthesize_voice(cues: list[DialogueCue], persona: Persona, work_dir: Path) -> list[DialogueCue]:
    """Fills in `audio_path` for each cue via DashScope TTS (CosyVoice) if
    DASHSCOPE_API_KEY is set. Otherwise — or if a given cue's synthesis call fails —
    leaves audio_path=None; the compositor then burns the text in as a subtitle
    instead of dropping the line silently."""
    if not settings.dashscope_api_key or not cues:
        return cues

    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer

    dashscope.api_key = settings.dashscope_api_key
    out_dir = work_dir / "dialogue_audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = []
    for i, cue in enumerate(cues):
        try:
            synthesizer = SpeechSynthesizer(model="cosyvoice-v2", voice=persona.voice_style)
            audio_bytes = synthesizer.call(cue.text)
            audio_path = out_dir / f"cue_{i}.mp3"
            audio_path.write_bytes(audio_bytes)
            result.append(cue.model_copy(update={"audio_path": str(audio_path)}))
        except Exception as exc:
            print(f"dialogue TTS failed for cue {i} ({cue.text!r}): {exc}", file=sys.stderr)
            result.append(cue)
    return result
