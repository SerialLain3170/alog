# TAKEOVER.md — migrating `harmonize` to the H200 server

Written 2026-07-20 on the current 4x L40S host, for whoever (human or agent) picks this
project up on the new H200 server. Read this before touching anything — several
workarounds below exist only because of the old host's tight 46GB/GPU VRAM ceiling and
should be re-tested, not blindly ported, on H200 (141GB/GPU).

## 1. What this project is

`harmonize` inserts/replaces a character (currently: Hatsune Miku, `HatsuneMikuSekai.png`)
into real video. The goal has narrowed over the course of this project from "reactive
anime vlog character composited into a landscape" (the original design doc, see
`.claude/plans/glittery-hopping-wigderson.md` if it's still around) down to a much more
concrete task: **replace the real person in an existing vlog video with the reference
character, consistently, frame to frame**, ideally rendered as flat 2D anime rather than
a photoreal/CG blend.

Three approaches have been tried, in this order:
1. **PISCO** (sparse-keyframe-to-dense propagation/harmonization) — works, but is an
   insertion/harmonization tool, not an autonomous character animator. Good quality,
   not the current focus.
2. **Wan2.2-Animate-14B, replacement mode** — works reliably for short clips
   (~1s/30 frames) at 1280x720. This is the current best *working* result:
   `/data/shasegawa/harmonize/jobs/_replace_test/replace_output_short.mp4`. Downside:
   renders the character as photoreal/CG-blended, not flat anime.
3. **SCAIL-2** (zai-org/SCAIL-2) — chosen because it avoids pose/skeleton intermediates
   and has native long-sequence chunking. Setup is fully done (weights downloaded,
   checkpoint converted, two real bugs in the vendored code patched — see §5). **Blocked
   purely on GPU availability**: it OOMs on 2-GPU FSDP (same failure shape Wan-Animate
   had), and only 2 of the 4 L40S GPUs were free at time of writing. This is almost
   certainly a non-issue on H200 — see §7.

## 2. Repo & storage layout — READ THIS BEFORE COPYING FILES

Large files must never go into git or a home directory. Two different absolute paths
are in play and they are **not** interchangeable — check `app/config.py` /
`pipeline/config.py` defaults and `.env` before assuming either one:

- `/data/shasegawa/vendor/Wan2.2` (code, 19MB) and
  `/data/shasegawa/vendor/Wan2.2-Animate-14B` (weights, 68GB) — the *original*
  Wan2.2 location, referenced by `app/config.py`'s `wan_repo_dir`/`wan_ckpt_dir` and
  `.env`'s `WAN_REPO_DIR`/`WAN_CKPT_DIR`. This convention predates the
  `HARMONIZE_DATA_DIR` one below.
- `/data/shasegawa/harmonize/{model_cache,motion_library,uploads,outputs,jobs,vendor}` —
  everything added after that point: PISCO (114GB), SCAIL-2 (146GB), SAM2/Whisper
  caches, per-job intermediates and outputs. Referenced by `pipeline/config.py`'s
  `harmonize_data_dir` and `.env`'s `HARMONIZE_DATA_DIR`.

**Known cleanup opportunity, not yet acted on**: `data/vendor/Wan2.2-Animate-14B` inside
the git working tree (53GB) is a full duplicate of
`/data/shasegawa/vendor/Wan2.2-Animate-14B`, left over from before the storage
convention was corrected. It's gitignored (`data/vendor/*` in `.gitignore`) so it was
never committed, but don't bother copying it during migration — nothing reads from it
(`app/config.py` points at `/data/shasegawa/...`, not the in-repo `data/` dir).

**The git repo itself has zero commits** (`git log` → "No commits yet"). Everything is
currently untracked. There is no remote. Decide on migration whether to `git init`-and-
commit fresh on the new host, or tar up `.git` as-is (it has no history to lose either
way) — this doesn't need any special handling, just don't assume there's commit history
to preserve.

## 3. What actually needs to move

From the git working tree (`/home/shasegawa/program/harmonize`): everything except
`.venv/`, `__pycache__/`, `data/vendor/Wan2.2-Animate-14B` (see cleanup note above). Keep
`HatsuneMikuSekai.png` and `iceland.mp4` — they're the standing test assets referenced
throughout the code and by hand in ad-hoc test scripts.

From `/data/shasegawa` (or wherever the new host's equivalent shared-storage mount is):
- `vendor/Wan2.2` (19MB) + `vendor/Wan2.2-Animate-14B` (68GB)
- `harmonize/vendor/PISCO` (114GB) + `harmonize/vendor/SCAIL-2` (146GB, includes the
  converted 65GB safetensors — re-running `convert.py` is possible but slow, prefer
  copying the already-converted file at
  `harmonize/vendor/SCAIL-2/converted/SCAIL-2.safetensors`)
- `harmonize/model_cache` (SAM2 + Whisper caches, ~1.6GB — cheap to just re-download
  instead of copying if that's easier)
- `harmonize/jobs/_replace_test/` and `harmonize/jobs/_scail_test/` if you want the
  existing test artifacts (driving clips, masks, the one successful Wan-Animate output)
  rather than regenerating them

Total is roughly 330GB+ of vendored model weights. Do not `git add` any of this.

## 4. Python environments — THREE separate venvs, intentionally

All were created with `uv`, not stdlib `venv` (no `pip` binary inside — use
`uv pip install --python <venv>/bin/python ...` or `uv pip list --python ...`).

| venv | torch | Why isolated |
|---|---|---|
| `/home/shasegawa/program/harmonize/.venv` (main) | 2.8.0+cu128 | Runs everything except PISCO/SCAIL-2 subprocesses: FastAPI/Celery app, `pipeline/` (scene parser, SAM2, Depth-Anything, planner), and shells out to Wan2.2's `generate.py` via `pipeline/animate/wan_pipeline.py`. Also has flash-attn 2.8.3.post1, peft==0.17.0 (pinned — see §5), transformers==4.51.3, opencv-python-headless==4.10.0.84, numpy==1.26.4. |
| `/data/shasegawa/harmonize/vendor/PISCO/.venv` | 2.6.0+cu124 | PISCO/DiffSynth-Studio pins an older torch than the rest of the stack; installing its deps into the main venv would break the Wan2.2 chain. |
| `/data/shasegawa/harmonize/vendor/SCAIL-2/.venv` | 2.8.0+cu128 | Same torch as main venv (deliberately matched — flash-attn wheel compatibility, see §5), but SCAIL-2's own transformers==5.14.1/diffusers==0.39.0/decord==0.6.0 stack is kept separate to avoid destabilizing the main venv. |

`requirements-worker.txt` is **stale/aspirational** — it pins `numpy==2.1.*` but the
live main venv actually runs `numpy==1.26.4` (downgraded by hand while resolving a real
conflict with the Wan2.2 stack). Don't trust the requirements files at face value on the
new host; if you need to rebuild an environment, prefer literally copying the working
venv, or `uv pip freeze` the live one first and use that as the real spec.

flash-attn is installed everywhere from prebuilt wheels off `Dao-AILab/flash-attention`
GitHub releases, matched exactly to torch version + CUDA + cxx11abi + Python 3.12 (e.g.
`flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`). **On H200 you
will need new wheels** — same matching process, just against whatever torch/CUDA the new
host's driver supports. Don't assume the same wheel URLs work.

## 5. Patches applied to vendored (non-git-tracked) code — MUST re-verify after copy

These are hand-edits to files that live outside this git repo, in the vendored
model dirs. If you re-clone/re-download instead of copying the existing checkout, you
must re-apply these:

1. **`/data/shasegawa/vendor/Wan2.2/wan/configs/__init__.py`** — added
   `'480*832', '832*480'` to `animate-14B`'s `SUPPORTED_SIZES` tuple, to test a lower
   resolution under VRAM pressure. **Result: this resolution produces complete garbage
   output for animate-14B specifically** (confirmed by direct test, not a hypothesis) —
   the patch is there but should NOT actually be used; 1280x720/720x1280 are the only
   viable sizes for this task regardless of GPU. On H200, if you want a lower-res option
   for speed, this is not the way to get it — you'd need to check whether Wan-Animate's
   own team ever validates lower resolutions upstream, don't just widen the tuple again.

2. **`/data/shasegawa/harmonize/vendor/SCAIL-2/convert.py`** — `torch.load()` needed
   `weights_only=False` added explicitly. torch>=2.6 defaults `weights_only=True`,
   which rejects a numpy global inside the official `zai-org/SCAIL-2` checkpoint. Safe
   because the file is from that official HF repo, not an untrusted source.

3. **`/data/shasegawa/harmonize/vendor/SCAIL-2/generate.py`** — uncommented a
   `dist.init_process_group(...)` call (around what's now line ~376) that upstream left
   commented out. As shipped, that line only ever executes indirectly via xfuser's
   `init_distributed_environment` when `--ulysses_size`/`--ring_size` > 1 — but xfuser
   isn't even in `requirements.txt`, so plain `--dit_fsdp --t5_fsdp` weight-sharding
   (no sequence parallelism) had no working multi-GPU path at all before this fix. This
   looks like a genuine upstream bug/gap in the `wan-scail2` branch, not something
   specific to our setup — worth allowing for as a general fact about this fork if you
   pull upstream changes later; re-check whether upstream has fixed this before
   re-copying that file.

`peft==0.17.0` in the main venv is also a deliberate pin, not a default — 0.19.1 broke
on an `ALL_PARALLEL_STYLES` import when loading the Wan-Animate relighting LoRA against
transformers==4.51.3, and 0.15.2 broke diffusers' own floor. Don't let anything bump it
without re-testing the relighting LoRA load path.

## 6. `.env` — required vars (values are host-specific, not reproduced here)

See `.env.example` for the full annotated list. Load-bearing ones:
- `WAN_REPO_DIR`, `WAN_CKPT_DIR` — see §2, must point at the Wan2.2 location.
- `HARMONIZE_DATA_DIR` — see §2, must point at the harmonize-prefixed location.
- `WAN_SIZE` — must be `1280*720` or `720*1280` (animate-14B's actual supported sizes;
  see the SUPPORTED_SIZES patch caveat in §5).
- `WORKER_GPUS` — comma-separated GPU indices; if more than one, `wan_pipeline.py`
  launches via `torchrun --dit_fsdp --t5_fsdp --ulysses_size=<count>` with
  `NCCL_P2P_DISABLE=1` (see §7 for why, and reconsider on H200's likely different
  interconnect).
- `ANTHROPIC_API_KEY` — hard-required; `pipeline/config.py`'s `require_anthropic_key()`
  raises a clear error rather than silently degrading if unset.
- `DASHSCOPE_API_KEY` — optional (TTS); pipeline runs fine without it, subtitles-only.

`.env` itself is gitignored and was not migrated by this doc — copy it by hand
(it currently exists at the repo root on this host) or reconstruct from `.env.example` +
your own API keys.

## 7. GPU/VRAM context — re-test everything here on H200, don't just copy settings

This host is 4x NVIDIA L40S, 46GB VRAM each, driver 595.71.05, CUDA 13.2. Every
multi-GPU/FSDP workaround in this codebase exists because 14B-class models (Wan-Animate,
SCAIL-2) do not fit on a single 46GB GPU, and even 2x46GB FSDP-sharded still OOMs during
the DiT forward pass (confirmed twice, once for each model, both failing with ~43-44GB
resident right before the crash). The fix both times was simply "use 4 GPUs instead of
2" — not a smarter memory technique.

H200 has 141GB per GPU. It is very plausible that:
- A single H200 GPU can hold what needed 4x L40S (14B bf16 weights are roughly
  28-30GB; even with activations/T5/VAE/CLIP resident it may fit in 141GB without FSDP
  at all).
- `--dit_fsdp --t5_fsdp --ulysses_size=N` and the whole `GpuPlan`/multi-GPU machinery in
  `worker/gpu_config.py` may become unnecessary complexity for single-job runs, though
  it's still useful for running multiple jobs concurrently across GPUs.
- The `NCCL_P2P_DISABLE=1` workaround (added because NCCL P2P hung indefinitely between
  two L40S GPUs on this host despite normal PCIe topology) may or may not reproduce on
  H200's interconnect (NVLink vs PCIe changes the failure surface entirely) — test
  without it first rather than assuming it's still needed.
- The 480x832/832x480 "garbage output" result for animate-14B (§5) was about the
  model/resolution combination, not VRAM — don't expect more VRAM to fix that one.

Recommend: once migrated, re-run the SCAIL-2 replacement job on a single H200 GPU first
(no FSDP, no torchrun) before reaching for multi-GPU at all. If it fits, that simplifies
everything downstream.

## 8. Known-good and known-broken results, for comparison after migration

- `jobs/_replace_test/replace_output_short.mp4` — Wan-Animate replacement, 30 frames,
  1280x720, 4-GPU FSDP + relighting LoRA. Stable, background preserved, but
  photoreal/CG-blended rather than flat anime. **The current best working output.**
- `jobs/_replace_test/replace_output_720p_4gpu.mp4` — same setup but 120 frames;
  degrades into noise after ~90-100 frames. Root-caused as sequence-length error
  accumulation, NOT a resolution issue (a separate, unrelated 480x832 test produced
  immediate total garbage from frame 1, a different failure mode — don't conflate the
  two if this comes up again). SCAIL-2's `--segment_len`/`--segment_overlap` chunking is
  the hoped-for fix for this specific problem; not yet verified since SCAIL-2 hasn't
  successfully completed a run yet (§1).
- SCAIL-2: fully set up, blocked purely on GPU count as of this writing. Prompt used for
  the pending test is in `/tmp/scail_prompt.txt` on this host (not migrated — it's a
  scratch file, rewrite it or pull from this doc's conversation history if needed) —
  describes the reference character eating noodles in a food-stall setting, matching
  the specific test driving clip at `jobs/_replace_test/driving_clip_short.mp4`.

## 9. Session-local automation that will NOT survive migration

At the time of writing, this Claude Code session has a background `nvidia-smi` poller
(via the `Monitor` tool) waiting for 4 free GPUs to retry the SCAIL-2 job, plus a
scheduled wakeup as a fallback. **These are artifacts of this specific chat session on
this specific host** — they do not persist across a server migration or a new
session/agent. Whoever picks this up on H200 needs to just directly check GPU
availability and re-launch; don't go looking for these pollers on the new host.

## 10. Other things worth knowing

- The motion-library bootstrap clip (`pipeline/motion/library.py`) references Wan2.2's
  own bundled example video in-place rather than copying it, because that file was
  discovered to actually be a copyrighted "Last Week Tonight" broadcast clip, not
  generic demo footage. It's gated behind explicit opt-in
  (`harmonize motion bootstrap`) with a printed warning every use. Don't remove that
  gate or start treating that clip as generic sample data.
- Depth-Anything's pipeline output is *inverse* depth (higher value = nearer). This is
  inverted in `pipeline/scene_parser/depth.py`'s `estimate_depth()` — if depth-based
  occlusion/scale logic ever looks backwards again, check that inversion is still there
  before re-deriving the bug from scratch.
- SAM2 is pinned to a pre-2.1 commit/checkpoint (matching Wan2.2's own
  `requirements_animate.txt`), which changes some API surface from what SAM2's current
  docs describe (`add_new_points`, not `add_new_points_or_box`; `sam2_hiera_s.yaml`
  config name, not `sam2.1_hiera_s.yaml`). If you upgrade SAM2 independently on the new
  host, expect to hit both of these again.
