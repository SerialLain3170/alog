#!/usr/bin/env python3
"""
Standalone CLI for the scene-reactive anime vlog character pipeline: parses a real
background video + its audio, plans what a persona-driven character would do in
that scene, animates the character (Wan2.2-Animate-14B, animation mode), and
composites it back onto the original footage.

Usage:
    python harmonize_cli.py run \\
        --video path/to/background.mp4 \\
        --character path/to/character.png \\
        --persona curious_vlogger \\
        --title "Night Walk" --theme "a quiet night stroll" \\
        --output path/to/result.mp4 \\
        --gpus 0

    python harmonize_cli.py motion add turn_head path/to/clip.mp4
    python harmonize_cli.py motion list
    python harmonize_cli.py motion bootstrap
"""

import argparse
import sys
import uuid
from pathlib import Path

from pipeline.motion import library
from pipeline.schemas import Goal
from pipeline.world_model import persona as persona_presets
from worker.gpu_config import GpuPlan


def _cmd_run(args: argparse.Namespace) -> None:
    # Imported lazily: this pulls in torch/transformers/sam2/anthropic/moviepy,
    # which every other subcommand (motion add/list/bootstrap) has no need to pay
    # startup cost for.
    from pipeline.orchestrator import run_pipeline

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")
    if not args.character.exists():
        sys.exit(f"character image not found: {args.character}")

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        sys.exit("--gpus must list at least one GPU index")
    gpu_plan = GpuPlan(
        gpu_ids=gpu_ids, cuda_visible_devices=",".join(str(g) for g in gpu_ids), use_fsdp=len(gpu_ids) > 1
    )

    persona = persona_presets.get_preset(args.persona, str(args.character), speech_language=args.language)
    goal = Goal(title=args.title, theme=args.theme, tone=args.tone)
    job_id = args.job_id or f"cli-{uuid.uuid4()}"

    def progress(stage: str) -> None:
        print(f"[{job_id}] {stage}...")

    output_path = run_pipeline(
        video_path=str(args.video),
        character_image_path=str(args.character),
        persona=persona,
        goal=goal,
        job_id=job_id,
        gpu_plan=gpu_plan,
        anchor_seed=(args.anchor_x, args.anchor_y),
        progress=progress,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path).replace(args.output)

    print(f"Done: {output_path}")


def _cmd_motion_add(args: argparse.Namespace) -> None:
    if not args.clip.exists():
        sys.exit(f"clip not found: {args.clip}")
    library.add_clip(args.tag, str(args.clip))
    print(f"Added {args.clip} under tag '{args.tag}'")


def _cmd_motion_list(args: argparse.Namespace) -> None:
    manifest = library.list_clips(args.tag)
    if not manifest:
        print("(no motion clips registered)")
        return
    for tag, clips in manifest.items():
        print(f"{tag}:")
        for clip in clips:
            print(f"  - {clip['path']} ({clip['duration']:.1f}s)")


def _cmd_motion_bootstrap(_args: argparse.Namespace) -> None:
    library.bootstrap_for_smoke_test()
    print("Registered local smoke-test fallback clip (not for real use — see warning above).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline on one video")
    run_parser.add_argument("--video", required=True, type=Path, help="Real background video")
    run_parser.add_argument("--character", required=True, type=Path, help="Anime character reference image")
    run_parser.add_argument(
        "--persona", default="curious_vlogger", choices=list(persona_presets.PRESETS), help="Persona preset"
    )
    run_parser.add_argument("--language", default="ja", help="Dialogue language (default: ja)")
    run_parser.add_argument("--title", default="Vlog", help="Vlog title/goal")
    run_parser.add_argument("--theme", default="exploring a real place", help="Vlog theme/focus")
    run_parser.add_argument("--tone", default="casual vlog", help="Vlog tone")
    run_parser.add_argument(
        "--anchor-x", type=float, default=0.5, help="Normalized x of the character's ground anchor on frame 0"
    )
    run_parser.add_argument(
        "--anchor-y", type=float, default=0.85, help="Normalized y of the character's ground anchor on frame 0"
    )
    run_parser.add_argument("--gpus", default="0", help="Comma-separated GPU indices for Wan2.2-Animate generation")
    run_parser.add_argument(
        "--output", type=Path, default=None, help="Also copy the final mp4 here (default: only under outputs dir)"
    )
    run_parser.add_argument("--job-id", default=None, help="Reuse a specific job id (default: random)")
    run_parser.set_defaults(func=_cmd_run)

    motion_parser = subparsers.add_parser("motion", help="Manage the motion clip library")
    motion_sub = motion_parser.add_subparsers(dest="motion_command", required=True)

    add_parser = motion_sub.add_parser("add", help="Register a real driving clip for a behavior tag")
    add_parser.add_argument("tag", help=f"Behavior tag, e.g. one of: {', '.join(library.MOTION_TAGS)}")
    add_parser.add_argument("clip", type=Path, help="Path to the driving video clip")
    add_parser.set_defaults(func=_cmd_motion_add)

    list_parser = motion_sub.add_parser("list", help="List registered motion clips")
    list_parser.add_argument("--tag", default=None, help="Only list this tag")
    list_parser.set_defaults(func=_cmd_motion_list)

    bootstrap_parser = motion_sub.add_parser(
        "bootstrap", help="Register a local smoke-test fallback clip (not for real use)"
    )
    bootstrap_parser.set_defaults(func=_cmd_motion_bootstrap)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
