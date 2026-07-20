from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class GpuPlan:
    gpu_ids: list[int]
    cuda_visible_devices: str
    use_fsdp: bool

    @property
    def world_size(self) -> int:
        return len(self.gpu_ids)


def resolve_gpu_plan() -> GpuPlan:
    """
    Reads WORKER_GPUS for this process. One GPU -> run generate.py directly
    on it. Multiple GPUs -> shard a single job across them via FSDP + Ulysses
    sequence parallelism (torchrun), since the Wan2.2 A14B models don't fit
    comfortably on one L40S (48GB) without offloading/quantization.
    """
    gpu_ids = settings.worker_gpu_list
    if not gpu_ids:
        raise ValueError("WORKER_GPUS must list at least one GPU index")

    return GpuPlan(
        gpu_ids=gpu_ids,
        cuda_visible_devices=",".join(str(g) for g in gpu_ids),
        use_fsdp=len(gpu_ids) > 1,
    )
