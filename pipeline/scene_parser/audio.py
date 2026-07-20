import functools
from pathlib import Path

import librosa
import numpy as np
import torch

from pipeline.config import settings
from pipeline.schemas import AudioEvent, SpeechSegment


class NoAudioTrackError(RuntimeError):
    pass


def extract_audio(video_path: str, out_wav: Path, sr: int = 16000) -> Path:
    """Extracts the mono audio track from `video_path` to a wav file."""
    from moviepy import VideoFileClip

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise NoAudioTrackError(f"Video has no audio track: {video_path}")
    clip.audio.write_audiofile(str(out_wav), fps=sr, nbytes=2, codec="pcm_s16le", logger=None)
    clip.close()
    return out_wav


def detect_onsets(y: np.ndarray, sr: int) -> list[AudioEvent]:
    """librosa onset detection with per-onset RMS-based intensity."""
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    rms_max = float(rms.max()) if len(rms) else 0.0

    events = []
    for i, t in enumerate(onset_times):
        idx = min(int(np.searchsorted(rms_times, t)), len(rms) - 1)
        intensity = float(rms[idx] / (rms_max + 1e-6))
        duration = float(onset_times[i + 1] - t) if i + 1 < len(onset_times) else 1.0
        events.append(
            AudioEvent(
                id=f"onset_{i}",
                label="onset",
                onset=float(t),
                duration=min(duration, 2.0),
                intensity=intensity,
            )
        )
    return events


@functools.lru_cache(maxsize=1)
def _audio_tagger():
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(task="audio-classification", model=settings.audio_tag_model_id, device=device)


def tag_audio_windows(
    y: np.ndarray, sr: int, window_s: float = 2.0, top_k: int = 3, min_score: float = 0.15
) -> list[AudioEvent]:
    """AudioSet tagging over sliding windows (bird calls, wind, waves, etc), keeping
    the top-k labels per window above `min_score`."""
    tagger = _audio_tagger()
    window_samples = int(window_s * sr)
    events = []
    idx = 0
    for start in range(0, len(y), window_samples):
        chunk = y[start : start + window_samples]
        if len(chunk) < sr * 0.5:
            continue
        results = tagger({"array": chunk, "sampling_rate": sr}, top_k=top_k)
        t = start / sr
        for r in results:
            if r["score"] < min_score:
                continue
            events.append(
                AudioEvent(
                    id=f"tag_{idx}", label=r["label"], onset=t, duration=window_s, intensity=float(r["score"])
                )
            )
            idx += 1
    return events


@functools.lru_cache(maxsize=1)
def _whisper_model():
    import whisper

    return whisper.load_model(
        settings.whisper_model_size, download_root=str(settings.model_cache_dir / "whisper")
    )


def transcribe_speech(wav_path: Path) -> list[SpeechSegment]:
    model = _whisper_model()
    result = model.transcribe(str(wav_path), fp16=torch.cuda.is_available())
    return [
        SpeechSegment(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip())
        for seg in result.get("segments", [])
        if seg["text"].strip()
    ]


def analyze_audio(video_path: str, work_dir: Path) -> tuple[list[AudioEvent], list[SpeechSegment]]:
    """Full audio analysis for one video: onset events, AudioSet tags, and any speech.
    Returns ([], []) if the video has no audio track — not every background clip has one."""
    try:
        wav_path = extract_audio(video_path, work_dir / "audio.wav")
    except NoAudioTrackError:
        return [], []

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    events = detect_onsets(y, sr) + tag_audio_windows(y, sr)
    speech = transcribe_speech(wav_path)
    return events, speech
