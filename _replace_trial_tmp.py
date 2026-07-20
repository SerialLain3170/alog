from pathlib import Path

from pipeline.animate import wan_pipeline
from worker.gpu_config import GpuPlan
from app.config import settings

settings.wan_size = "1280*720"

gpu_plan = GpuPlan(gpu_ids=[0, 1, 2, 3], cuda_visible_devices="0,1,2,3", use_fsdp=True)
work_dir = Path("/data/shasegawa/harmonize/jobs/_replace_test")
process_dir = work_dir / "process_results_short"
output_path = work_dir / "replace_output_short.mp4"

character = Path("HatsuneMikuSekai.png")
driving = work_dir / "driving_clip_short.mp4"

print("=== preprocess (replace mode, 1280x720, short 1s clip) ===", flush=True)
wan_pipeline.preprocess(character, driving, process_dir, replace_flag=True)
print("preprocess OK", flush=True)

print("=== generate (replace mode, 1280x720, FSDP 4 GPUs, with relighting lora) ===", flush=True)
wan_pipeline.generate(process_dir, output_path, gpu_plan, replace_flag=True, use_relighting_lora=True)
print("generate OK:", output_path, flush=True)
