# harmonize

A pipeline that watches a real background video (+ its audio), decides how a
persona-driven 2D anime character would react to it (gaze, emotion, movement,
dialogue), animates that character, and composites it back onto the original
footage as a plausible layer (ground-anchored, occlusion-aware, with a
synthetic contact shadow) — not a straight face/body swap.

## Architecture

```
scene_parser  ->  world_model  ->  planner  ->  motion  ->  animate  ->  compositor
(what's in      (what the       (what the      (assemble    (Wan2.2-    (matte +
 the video/     character       character      a driving     Animate-14B  shadow +
 audio)         notices)        decides to     video from    ANIMATION    occlude
                                 do)            clips)        mode)        onto bg)
```

Concrete tools behind each stage (see `pipeline/`):

| Stage | Implementation |
|---|---|
| Scene parsing | OpenCV optical flow/motion salience, `transformers` Depth-Anything, Claude vision (keyframe entity captioning), SAM2 (entity tracking), `librosa` + `transformers` AST (AudioSet tagging) + Whisper (speech) |
| World model | Deterministic persona-weighted salience/interest/fear scoring, gaze-target selection — no model |
| Planning | Claude API structured-output call (semantic actions + dialogue), rule-based behavior/timing mapping |
| Motion | A small curated library of real driving clips (walk/idle/turn-head/pick-up/sit/wave/startle) selected + retimed + concatenated per the behavior plan |
| Animation | Wan2.2-Animate-14B, **animation mode** (character mimics a driving video's motion — not replacement mode, which needs an actor already filmed in the target scene) |
| Compositing | SAM2 matting to RGBA, procedural contact shadow, optical-flow ground-anchor tracking, depth-based occlusion, `moviepy` audio/subtitle mux |

**Known limitations**, stated rather than hidden: physical object interaction
and occlusion are 2D/mask/depth heuristics, not true affordance planning or 3D
contact physics. Sound "direction" is inferred by correlating audio onsets
with visually salient entities appearing at the same time (mono/stereo audio
has no real spatial info). Motion variety is bounded by whatever clips are in
`motion_library/` — see below.

## Storage layout

All large/generated artifacts live under `/data/shasegawa/harmonize/`, never
in the git checkout:

```
/data/shasegawa/harmonize/
  model_cache/       HF/torch caches (Depth-Anything, AST, Whisper), SAM2 checkpoint
  motion_library/     driving clips + manifest.json, by behavior tag
  uploads/            per-job uploaded character image + background video
  outputs/             final composited mp4s, one per job id
  jobs/<job_id>/        intermediate JSON + frames for one run (scene_timeline.json,
                        world_state.json, semantic_plan.json, behavior_plan.json, ...)
```

The existing Wan2.2 code/weights (`/data/shasegawa/vendor/Wan2.2`,
`data/vendor/Wan2.2-Animate-14B`) are reused unchanged from the original
scaffold — not moved, not re-downloaded.

## Setup

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY (required); DASHSCOPE_API_KEY is optional (TTS)

./scripts/clone_wan_repo.sh      # if not already done — clones Wan-Video/Wan2.2
./scripts/download_weights.sh    # if not already done — downloads Animate-14B checkpoint (~72GB)

pip install -r requirements-worker.txt
pip install -r /data/shasegawa/vendor/Wan2.2/requirements.txt
pip install -r /data/shasegawa/vendor/Wan2.2/requirements_animate.txt   # SAM2
pip install -r /data/shasegawa/vendor/Wan2.2/requirements_s2v.txt       # whisper, librosa
```

### Motion library

The pipeline needs real driving clips (a person performing each behavior) to
assemble Wan-Animate's driving video from. Before a real run:

```bash
python harmonize_cli.py motion add walk_forward path/to/clip.mp4
python harmonize_cli.py motion add turn_head path/to/clip.mp4
python harmonize_cli.py motion list
```

Behavior tags: `idle_stand`, `walk_forward`, `turn_head`, `point_at`,
`pick_up_object`, `sit_down`, `wave`, `startle_flinch`.

For a quick end-to-end smoke test before recording real footage:

```bash
python harmonize_cli.py motion bootstrap
```

This registers Wan2.2's own bundled inference-demo clip as a fallback. That
clip is third-party broadcast footage the Wan2.2 project ships purely for its
own CLI examples — fine for confirming the pipeline runs, not for producing
anything real. Replace it with `motion add` before that.

## CLI (no web app / no Redis needed)

```bash
python harmonize_cli.py run \
    --video path/to/background.mp4 \
    --character path/to/character.png \
    --persona curious_vlogger \
    --title "Night Walk" --theme "a quiet night stroll" \
    --gpus 0                       # or "0,1,2,3" to shard the Wan-Animate step via FSDP
```

Persona presets: `curious_vlogger`, `timid_observer`, `deadpan_narrator` (see
`pipeline/world_model/persona.py`). Intermediate stage output lands under
`/data/shasegawa/harmonize/jobs/<job_id>/` for inspection — read
`scene_timeline.json` and `behavior_plan.json` first if a run looks off.

## Web app

```bash
docker compose build
docker compose up
```

API on `http://localhost:8000`:

- `POST /jobs` — multipart form: `character_image`, `video`, and optional
  `persona`, `title`, `theme`, `tone`, `language`, `anchor_x`, `anchor_y`
- `GET /jobs/{job_id}` — status cycles through `PENDING` → `SCENE_PARSER` →
  `WORLD_MODEL` → `SEMANTIC_PLAN` → `BEHAVIOR_PLAN` → `DIALOGUE` →
  `MOTION_ASSEMBLY` → `WAN_ANIMATE` → `MATTING` → `COMPOSITING` → `SUCCESS`/`FAILURE`
- `GET /jobs/{job_id}/result` — downloads the output MP4 once `SUCCESS`

## GPU strategy for a 4x L40S box

Official docs quote ≥80GB VRAM for naive single-GPU bf16 inference of the
Animate-14B model — that won't fit one 48GB L40S as-is. `worker/gpu_config.py`
supports either one job sharded across multiple GPUs via FSDP
(`WORKER_GPUS=0,1,2,3`) or one worker process pinned per GPU
(`docker-compose.yml`'s `worker-gpu0..3` services) — benchmark both against
your actual hardware before committing to one.
