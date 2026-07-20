"""
Thin subprocess wrapper around the official Wan-Video/Wan2.2 scripts for
Animate-14B. Supports both workflows:
- ANIMATION mode (replace_flag=False): character reference image + any driving
  video of a real person -> a video of the character mimicking that person's
  motion, isolated on its own background. Used by our own compositor pipeline
  (animate/matte.py, animate/shadow.py, compositor/) to build a background-agnostic
  character layer.
- REPLACEMENT mode (replace_flag=True): the driving video already contains a real
  person; that person is replaced in-place by the reference character, with the
  original background/lighting preserved (--use_relighting_lora).

Shells out to the repo's own CLI scripts rather than importing an internal Python
API, because the CLI flags are the documented/stable surface.
"""

import os
import subprocess
from pathlib import Path

from app.config import settings as wan_settings
from worker.gpu_config import GpuPlan


def _resolution_area() -> tuple[str, str]:
    w, h = wan_settings.wan_size.split("*")
    return w, h


def preprocess(
    character_image: Path, driving_video: Path, process_results_dir: Path, replace_flag: bool = False
) -> None:
    process_results_dir.mkdir(parents=True, exist_ok=True)
    script = wan_settings.wan_repo_dir / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
    w, h = _resolution_area()
    cmd = [
        "python",
        str(script),
        "--ckpt_path",
        str(wan_settings.wan_ckpt_dir / "process_checkpoint"),
        "--video_path",
        str(driving_video),
        "--refer_path",
        str(character_image),
        "--save_path",
        str(process_results_dir),
        "--resolution_area",
        w,
        h,
    ]
    if replace_flag:
        # Matches Wan2.2's own README replace-preprocessing example exactly —
        # these inpainting-search params are specific to replacement mode.
        cmd += ["--iterations", "3", "--k", "7", "--w_len", "1", "--h_len", "1", "--replace_flag"]
    else:
        cmd += ["--retarget_flag"]
    _run(cmd, env={})


_ENTRYPOINT = Path(__file__).parent / "_generate_entrypoint.py"


def generate(
    process_results_dir: Path,
    output_path: Path,
    gpu_plan: GpuPlan,
    replace_flag: bool = False,
    use_relighting_lora: bool = True,
) -> None:
    generate_py = wan_settings.wan_repo_dir / "generate.py"

    # --offload_model True is explicit rather than left to generate.py's own default
    # (False whenever world_size > 1) — confirmed by direct testing that even 2x
    # L40S (48GB each) OOMs at 1280x720 during the actual sampling loop without it,
    # not just during model loading.
    base_args = [
        "--task",
        "animate-14B",
        "--ckpt_dir",
        str(wan_settings.wan_ckpt_dir),
        "--src_root_path",
        str(process_results_dir),
        "--refert_num",
        "1",
        "--size",
        wan_settings.wan_size,
        "--frame_num",
        str(wan_settings.wan_frame_num),
        "--save_file",
        str(output_path),
        "--offload_model",
        "True",
    ]
    if replace_flag:
        # Replacement mode: preserve the driving video's own background/lighting.
        base_args.append("--replace_flag")
        if use_relighting_lora:
            # Blends the reference character into the original lighting via a LoRA
            # — adds VRAM on top of an already tight budget at 1280x720 on 2x48GB
            # GPUs (confirmed OOMing at VAE decode with it on); pass False to trade
            # that automatic lighting match away for a completed render.
            base_args.append("--use_relighting_lora")

    # Run through _generate_entrypoint.py (imports cv2 before generate.py's own
    # torch/torchvision imports get a chance to) rather than generate.py directly —
    # see that file's docstring for why.
    args = [str(generate_py), *base_args]

    if gpu_plan.use_fsdp:
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpu_plan.world_size}",
            str(_ENTRYPOINT),
            *args,
            "--dit_fsdp",
            "--t5_fsdp",
            f"--ulysses_size={gpu_plan.world_size}",
        ]
    else:
        cmd = ["python", str(_ENTRYPOINT), *args]

    env = {
        "CUDA_VISIBLE_DEVICES": gpu_plan.cuda_visible_devices,
        # PyTorch's suggested fix for the CUDA-OOM-despite-large-reserved-pool
        # pattern seen here at 1280x720 — lets fragmented reserved memory actually
        # get used instead of forcing a fresh allocation.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if gpu_plan.use_fsdp:
        # Confirmed by direct testing on this host: NCCL's default P2P transport
        # hangs indefinitely (600s collective-op timeout) between these GPUs despite
        # `nvidia-smi topo -m` showing a normal PCIe (NODE) link — a host/driver
        # quirk, not something fixable from our side beyond disabling P2P.
        env["NCCL_P2P_DISABLE"] = "1"

    _run(cmd, env=env)


def _run(cmd: list[str], env: dict[str, str]) -> None:
    full_env = {**os.environ, **env}
    result = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
