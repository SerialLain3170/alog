import os
import subprocess
from pathlib import Path

import numpy as np

from pipeline.config import settings
from pipeline.pisco.prepare import prepare_pisco_inputs


def run_pisco_chunk(
    background_video_path: str,
    start_frame: int,
    char_rgba: np.ndarray,
    anchor_positions: list[tuple[float, float]],
    anchor_frame_indices: list[int],
    work_dir: Path,
    gpu_id: int = 0,
) -> Path:
    """Prepares one PISCO input package (in this process) and runs PISCO inference
    on it (in PISCO's own isolated venv, via subprocess — see pipeline/config.py's
    pisco_python for why it's isolated). Returns the output video path."""
    prepare_pisco_inputs(
        background_video_path, start_frame, char_rgba, anchor_positions, anchor_frame_indices, work_dir
    )

    output_path = work_dir / "pisco_output.mp4"
    cmd = [
        str(settings.pisco_python),
        str(settings.pisco_dir / "inference" / "pretrained" / "infer_harmonize_bridge.py"),
        "--work-dir",
        str(work_dir),
        "--output",
        str(output_path),
        "--model-id",
        settings.pisco_model_id,
        "--width",
        str(settings.pisco_width),
        "--height",
        str(settings.pisco_height),
        "--num-frames",
        str(settings.pisco_num_frames),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result = subprocess.run(cmd, cwd=str(settings.pisco_dir), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"PISCO inference failed ({result.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return output_path
