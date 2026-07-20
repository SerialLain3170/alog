import base64

import cv2
import numpy as np
from anthropic import Anthropic

from pipeline.config import require_anthropic_key, settings

_SYSTEM_PROMPT = (
    "You are the perception module for an anime vlog character being composited "
    "into real background footage. Given a single video frame, identify the "
    "entities in it that a curious on-screen character would plausibly notice or "
    "react to (animals, moving objects, signage, interesting props, people, "
    "notable scenery). Ignore generic background (sky, pavement, generic trees) "
    "unless something about it is unusual."
)

_TOOL = {
    "name": "report_entities",
    "description": "Report salient entities visible in this frame.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "short category, e.g. 'bird flock', 'signage', 'dropped object'",
                        },
                        "description": {
                            "type": "string",
                            "description": "one sentence: what it is / what's notable about it",
                        },
                        "cx": {"type": "number", "description": "normalized 0-1 horizontal center"},
                        "cy": {"type": "number", "description": "normalized 0-1 vertical center"},
                        "w": {"type": "number", "description": "normalized 0-1 width"},
                        "h": {"type": "number", "description": "normalized 0-1 height"},
                        "salience": {
                            "type": "number",
                            "description": "0-1, how much this would grab a curious vlogger's attention",
                        },
                    },
                    "required": ["label", "description", "cx", "cy", "w", "h", "salience"],
                },
            },
            "mood": {"type": "string", "description": "overall scene mood/lighting, one short phrase"},
        },
        "required": ["entities", "mood"],
    },
}


def _encode_frame(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def detect_keyframe_entities(frame_bgr: np.ndarray) -> dict:
    """Returns {"entities": [{label, description, cx, cy, w, h, salience}], "mood": str}
    for a single frame via Claude vision."""
    client = Anthropic(api_key=require_anthropic_key())
    image_b64 = _encode_frame(frame_bgr)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_entities"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                    },
                    {"type": "text", "text": "Report the salient entities in this frame."},
                ],
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Claude did not return a tool_use block")
