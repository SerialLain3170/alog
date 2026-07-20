from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "PENDING"
    SCENE_PARSER = "SCENE_PARSER"
    WORLD_MODEL = "WORLD_MODEL"
    SEMANTIC_PLAN = "SEMANTIC_PLAN"
    BEHAVIOR_PLAN = "BEHAVIOR_PLAN"
    DIALOGUE = "DIALOGUE"
    MOTION_ASSEMBLY = "MOTION_ASSEMBLY"
    WAN_ANIMATE = "WAN_ANIMATE"
    MATTING = "MATTING"
    COMPOSITING = "COMPOSITING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class JobSubmitResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    detail: str | None = None
