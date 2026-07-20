from pathlib import Path

from pipeline.animate import matte, wan_pipeline
from pipeline.compositor import anchor, composite
from pipeline.config import settings
from pipeline.motion import assembler
from pipeline.planner import behavior, dialogue, semantic
from pipeline.scene_parser import parser
from pipeline.schemas import Goal, Persona
from pipeline.world_model import transform
from worker.gpu_config import GpuPlan


def run_pipeline(
    video_path: str,
    character_image_path: str,
    persona: Persona,
    goal: Goal,
    job_id: str,
    gpu_plan: GpuPlan,
    anchor_seed: tuple[float, float] = (0.5, 0.85),
    progress=lambda stage: None,
) -> Path:
    """Runs the full scene-parse -> world-model -> plan -> animate -> composite
    pipeline for one job. Every stage's output is persisted as JSON under
    jobs/<job_id>/ so a run can be inspected/debugged/resumed by hand without
    re-running the expensive stages."""
    job_dir = settings.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    progress("scene_parser")
    timeline = parser.parse_scene(video_path, job_dir / "scene_parser")
    (job_dir / "scene_timeline.json").write_text(timeline.model_dump_json(indent=2))

    progress("world_model")
    world_state = transform.build_world_state(timeline, persona)
    (job_dir / "world_state.json").write_text(world_state.model_dump_json(indent=2))

    progress("semantic_plan")
    semantic_plan = semantic.plan_semantic_actions(timeline, world_state, persona, goal)
    (job_dir / "semantic_plan.json").write_text(semantic_plan.model_dump_json(indent=2))

    progress("behavior_plan")
    entities_by_id = {e.id: e for e in timeline.entities}
    behavior_plan = behavior.build_behavior_plan(semantic_plan, entities_by_id)
    (job_dir / "behavior_plan.json").write_text(behavior_plan.model_dump_json(indent=2))

    progress("dialogue")
    cues = dialogue.extract_dialogue_cues(behavior_plan)
    cues = dialogue.synthesize_voice(cues, persona, job_dir)

    progress("motion_assembly")
    motion = assembler.assemble_driving_video(behavior_plan, job_dir / "motion")
    (job_dir / "assembled_motion.json").write_text(motion.model_dump_json(indent=2))

    progress("wan_animate")
    process_results_dir = job_dir / "wan_process_results"
    raw_animate_path = job_dir / "wan_raw_output.mp4"
    wan_pipeline.preprocess(Path(character_image_path), Path(motion.driving_video_path), process_results_dir)
    wan_pipeline.generate(process_results_dir, raw_animate_path, gpu_plan)

    progress("matting")
    matte_dir = matte.matte_character(str(raw_animate_path), job_dir / "matte")

    progress("compositing")
    anchor_positions = anchor.track_anchor(video_path, anchor_seed)
    silent_path = job_dir / "composite_silent.mp4"
    composite.composite_silent(video_path, matte_dir, anchor_positions, silent_path)

    output_path = settings.outputs_dir / f"{job_id}.mp4"
    composite.mux_final(silent_path, video_path, cues, output_path)

    return output_path
