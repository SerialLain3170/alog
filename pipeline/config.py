import os
from pathlib import Path

from pydantic_settings import BaseSettings


class PipelineSettings(BaseSettings):
    harmonize_data_dir: Path = Path("/data/shasegawa/harmonize")

    anthropic_api_key: str = ""
    dashscope_api_key: str = ""

    claude_model: str = "claude-sonnet-5"

    # Scene parsing tunables.
    keyframe_interval_s: float = 2.0
    entity_salience_floor: float = 0.35
    max_tracked_entities: int = 8
    world_state_interval_s: float = 1.0
    character_scale: float = 0.35
    color_harmonize_strength: float = 0.35
    subtitle_font_path: str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    audio_tag_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    whisper_model_size: str = "small"
    # Pinned to the SAM2 commit Wan2.2's own requirements_animate.txt installs
    # (pre-2.1 configs) — checkpoint/config names must match that vintage.
    sam2_checkpoint_url: str = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    )
    sam2_config_name: str = "sam2_hiera_s.yaml"

    # PISCO (github.com/taco-group/PISCO) lives in its own isolated venv — it pins
    # torch==2.6.0 while the rest of this project's environment (Wan2.2, SAM2,
    # flash-attn) is built against torch 2.8.0. Installing PISCO's deps into the
    # main venv would risk breaking that already-working stack, so we shell out to
    # a separate interpreter instead (same pattern as pipeline/animate/wan_pipeline.py
    # shelling out to the vendored Wan2.2 repo).
    pisco_dir: Path = Path("/data/shasegawa/harmonize/vendor/PISCO")
    pisco_python: Path = Path("/data/shasegawa/harmonize/vendor/PISCO/.venv/bin/python")
    pisco_model_id: str = "xiangbog/PISCO-1.3B"
    pisco_width: int = 832
    pisco_height: int = 480
    pisco_num_frames: int = 49

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def model_cache_dir(self) -> Path:
        path = self.harmonize_data_dir / "model_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def motion_library_dir(self) -> Path:
        path = self.harmonize_data_dir / "motion_library"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def uploads_dir(self) -> Path:
        path = self.harmonize_data_dir / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def outputs_dir(self) -> Path:
        path = self.harmonize_data_dir / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def jobs_dir(self) -> Path:
        path = self.harmonize_data_dir / "jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = PipelineSettings()

# All ML model downloads (Depth-Anything, AST audio tagger, ...) must land under
# /data, never in a user home dir cache, since these caches can grow to several GB
# and the repo/host disk quota outside /data is small. setdefault (not direct
# assignment) so an operator's own pre-existing HF_HOME/TORCH_HOME — e.g. a shared
# cache already reused across other projects on this host — wins if set; we only
# supply a /data-backed default when nothing else has claimed it.
os.environ.setdefault("HF_HOME", str(settings.model_cache_dir / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(settings.model_cache_dir / "torch"))


def require_anthropic_key() -> str:
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. The scene parser (keyframe captioning) "
            "and behavior planner (semantic action + dialogue generation) both "
            "require Claude API access — set it in .env before running the pipeline."
        )
    return settings.anthropic_api_key
