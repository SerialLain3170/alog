import functools

import numpy as np
import torch
from PIL import Image

from pipeline.config import settings


@functools.lru_cache(maxsize=1)
def _depth_pipeline():
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="depth-estimation",
        model=settings.depth_model_id,
        device=device,
    )


def estimate_depth(frame_bgr: np.ndarray) -> np.ndarray:
    """Returns a per-pixel depth map, same (h, w) as the input frame, normalized to
    0 (near) - 1 (far). Used by the compositor for ground-plane scale and occlusion."""
    image = Image.fromarray(frame_bgr[:, :, ::-1])
    result = _depth_pipeline()(image)
    raw = np.array(result["depth"], dtype=np.float32)
    raw -= raw.min()
    max_val = raw.max()
    if max_val > 0:
        raw /= max_val
    # Depth-Anything's own output convention is inverse/relative depth — HIGHER value
    # means NEARER, confirmed by inspecting a real frame where the closest object (a
    # foreground wall) had the highest raw value, not the lowest. Invert so this
    # function's own documented convention (0=near, 1=far) actually holds.
    return 1.0 - raw
