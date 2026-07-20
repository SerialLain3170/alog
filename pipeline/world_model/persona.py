from pipeline.schemas import Persona

PRESETS: dict[str, dict] = {
    "curious_vlogger": dict(
        name="Curious Vlogger",
        curiosity=0.8,
        timidity=0.2,
        talkativeness=0.7,
        voice_style="cheerful, energetic",
    ),
    "timid_observer": dict(
        name="Timid Observer",
        curiosity=0.4,
        timidity=0.7,
        talkativeness=0.3,
        voice_style="soft, hesitant",
    ),
    "deadpan_narrator": dict(
        name="Deadpan Narrator",
        curiosity=0.5,
        timidity=0.15,
        talkativeness=0.5,
        voice_style="flat, dry, understated",
    ),
}


def get_preset(name: str, character_image_path: str, speech_language: str = "ja") -> Persona:
    if name not in PRESETS:
        raise ValueError(f"Unknown persona preset '{name}'. Available: {list(PRESETS)}")
    return Persona(
        character_image_path=character_image_path,
        speech_language=speech_language,
        **PRESETS[name],
    )
