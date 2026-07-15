"""Shared fixtures for the Meeting Intelligence test suite.

Tests avoid the heavy ML dependencies (torch, pyannote.audio, faster-whisper,
paddleocr, openai, google-genai) entirely: every stage module exposes
its model/pipeline/client construction behind an injectable `*_loader` /
`*_factory` parameter, and these fixtures + fakes exercise that seam instead
of the real network/GPU-backed models. ffmpeg/ffprobe are real system
binaries and ARE exercised for real in the ingestion/scene-detection tests,
guarded by a `requires_ffmpeg` skip marker.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed")


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A tiny 2-second .mp4: 1s solid red + 1s solid blue, with a silent mono audio track."""
    video_path = tmp_path / "sample.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=64x64:d=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=64x64:d=1",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-r",
        "5",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path


@pytest.fixture
def sample_video_no_audio(tmp_path: Path) -> Path:
    """A tiny .mp4 with a video stream but no audio stream, for ingestion failure tests."""
    video_path = tmp_path / "no_audio.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=64x64:d=1",
        "-r",
        "5",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """A 1-second, 16kHz mono WAV of low-amplitude noise."""
    wav_path = tmp_path / "sample.wav"
    sample_rate = 16_000
    rng = np.random.default_rng(seed=0)
    audio = (rng.standard_normal(sample_rate) * 0.01).astype(np.float32)
    sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")
    return wav_path


def wav_properties(wav_path: Path) -> dict:
    with wave.open(str(wav_path), "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "sample_rate": wf.getframerate(),
        }
