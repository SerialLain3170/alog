import json
import shutil
import sys
from pathlib import Path

from pipeline.config import settings
from pipeline.scene_parser.frames import video_meta

# Wan2.2's own bundled inference example — a "Last Week Tonight" broadcast clip
# used upstream purely to demo the CLI. It is NOT ours to redistribute or bundle
# as a default asset (third-party broadcast footage, off-center subject, PIP
# overlay graphics baked into frame). `bootstrap_for_smoke_test` references it
# in place, never copies it, and is never called automatically.
_SMOKE_TEST_CLIP = Path("/data/shasegawa/vendor/Wan2.2/examples/wan_animate/animate/video.mp4")
FALLBACK_TAG = "fallback"

MOTION_TAGS = [
    "idle_stand",
    "walk_forward",
    "turn_head",
    "point_at",
    "pick_up_object",
    "sit_down",
    "wave",
    "startle_flinch",
]


def _manifest_path() -> Path:
    return settings.motion_library_dir / "manifest.json"


def _load_manifest() -> dict[str, list[dict]]:
    path = _manifest_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_manifest(manifest: dict[str, list[dict]]) -> None:
    _manifest_path().write_text(json.dumps(manifest, indent=2))


def bootstrap_for_smoke_test() -> None:
    """Registers Wan2.2's bundled inference-demo clip under the `fallback` tag purely
    so the pipeline can be exercised end-to-end before any real motion footage has
    been recorded. This is third-party broadcast footage the Wan2.2 project ships
    for its own CLI examples — not licensed for redistribution, so this references
    it at its existing vendor path rather than copying it, and must be invoked
    explicitly (`harmonize motion bootstrap`), never automatically. Replace it with
    real clips via `add_clip` before producing anything meant to be kept or shared."""
    if not _SMOKE_TEST_CLIP.exists():
        raise RuntimeError(f"Smoke-test clip not found at {_SMOKE_TEST_CLIP}")

    print(
        "WARNING: registering Wan2.2's bundled third-party demo clip as a motion "
        "fallback. This is for local smoke-testing only — replace it with real "
        "footage via `harmonize motion add` before producing anything for real use.",
        file=sys.stderr,
    )

    manifest = _load_manifest()
    fps, duration, _ = video_meta(str(_SMOKE_TEST_CLIP))
    manifest[FALLBACK_TAG] = [{"path": str(_SMOKE_TEST_CLIP), "duration": duration, "fps": fps}]
    _save_manifest(manifest)


def add_clip(tag: str, clip_path: str) -> None:
    manifest = _load_manifest()
    dest_dir = settings.motion_library_dir / tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(clip_path).name
    shutil.copy(clip_path, dest)

    fps, duration, _ = video_meta(str(dest))
    manifest.setdefault(tag, []).append({"path": str(dest), "duration": duration, "fps": fps})
    _save_manifest(manifest)


def list_clips(tag: str | None = None) -> dict[str, list[dict]]:
    manifest = _load_manifest()
    return {tag: manifest.get(tag, [])} if tag is not None else manifest


def get_clip_for_tag(tag: str) -> dict:
    """Returns one clip entry for `tag`, falling back to the bootstrap clip if the
    user hasn't added real footage for this specific tag yet."""
    manifest = _load_manifest()
    clips = manifest.get(tag) or manifest.get(FALLBACK_TAG)
    if not clips:
        raise RuntimeError(
            f"No motion clip available for tag '{tag}' and no fallback registered — "
            "run `harmonize motion add <tag> <clip.mp4>` for real footage, or "
            "`harmonize motion bootstrap` for a local smoke-test placeholder."
        )
    return clips[0]
