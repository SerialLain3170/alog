from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"

    wan_repo_dir: Path = Path("/data/shasegawa/vendor/Wan2.2")
    wan_ckpt_dir: Path = Path("/data/shasegawa/vendor/Wan2.2-Animate-14B")

    data_dir: Path = Path("/data/shasegawa")
    uploads_dir: Path = Path("/data/shasegawa/uploads")
    outputs_dir: Path = Path("/data/shasegawa/outputs")

    # animate-14B (the only task this repo drives) only supports these two sizes
    # per wan/configs/__init__.py's SUPPORTED_SIZES — not the 832*480 the
    # t2v/i2v example commands use.
    wan_size: str = "1280*720"
    wan_frame_num: int = 81
    wan_use_relighting_lora: bool = True

    worker_gpus: str = "0"

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def worker_gpu_list(self) -> list[int]:
        return [int(g) for g in self.worker_gpus.split(",") if g.strip()]


settings = Settings()
