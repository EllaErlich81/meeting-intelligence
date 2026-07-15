"""Core Setup & Ingestion stage.

Validates the input .mp4 with `ffprobe` and extracts its audio track to a
standardized 16kHz, mono, 16-bit WAV file with `ffmpeg`. This is the single
entry point every downstream stage (speech and vision) depends on, so it
fails loudly and early on malformed input rather than letting a bad file
propagate silently into VAD/ASR/OCR.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path

from .models import IngestionResult

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


class IngestionError(RuntimeError):
    """Raised when the input file is missing, unreadable, or has no audio track."""


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise IngestionError(
            f"Required binary '{name}' was not found on PATH. Install ffmpeg "
            f"(e.g. `brew install ffmpeg` or `apt-get install ffmpeg`)."
        )


def probe_video(video_path: str | Path) -> dict:
    """Run ffprobe and return the parsed JSON stream/format metadata.

    Raises IngestionError if the file does not exist, is not readable by
    ffprobe, or contains no audio stream.
    """
    video_path = Path(video_path)
    _require_binary("ffprobe")

    if not video_path.is_file():
        raise IngestionError(f"Input video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise IngestionError(f"ffprobe failed on {video_path}: {proc.stderr.strip()}")

    try:
        probe = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise IngestionError(f"ffprobe returned invalid JSON for {video_path}") from exc

    streams = probe.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_audio:
        raise IngestionError(f"No audio stream found in {video_path}")

    return probe


def extract_audio(
    video_path: str | Path,
    output_dir: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> Path:
    """Extract the audio track to a 16-bit PCM WAV file via ffmpeg."""
    _require_binary("ffmpeg")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise IngestionError(f"ffmpeg audio extraction failed: {proc.stderr.strip()}")

    return wav_path


def run_ingestion(video_path: str | Path, output_dir: str | Path) -> IngestionResult:
    """Validate the input video and extract its audio track.

    This is the stage entry point exposed to the CLI (`stage ingest`) and
    to the pipeline orchestrator.
    """
    probe = probe_video(video_path)
    duration_sec = float(probe.get("format", {}).get("duration", 0.0))

    wav_path = extract_audio(video_path, output_dir)

    logger.info("Ingested %s -> %s (%.2fs)", video_path, wav_path, duration_sec)
    return IngestionResult(
        video_path=str(video_path),
        wav_path=str(wav_path),
        duration_sec=duration_sec,
        sample_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
    )


async def run_ingestion_async(video_path: str | Path, output_dir: str | Path) -> IngestionResult:
    """Async wrapper so the pipeline orchestrator can run ingestion alongside other I/O."""
    return await asyncio.to_thread(run_ingestion, video_path, output_dir)
