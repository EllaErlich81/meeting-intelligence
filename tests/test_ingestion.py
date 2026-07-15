from __future__ import annotations

from pathlib import Path

import pytest

from conftest import requires_ffmpeg, wav_properties
from meeting_intelligence.ingestion import IngestionError, probe_video, run_ingestion


@requires_ffmpeg
def test_run_ingestion_produces_16khz_mono_wav(tmp_path: Path, sample_video: Path):
    output_dir = tmp_path / "out"
    result = run_ingestion(sample_video, output_dir)

    assert Path(result.wav_path).is_file()
    assert result.sample_rate == 16_000
    assert result.channels == 1
    assert result.duration_sec > 0

    props = wav_properties(Path(result.wav_path))
    assert props["sample_rate"] == 16_000
    assert props["channels"] == 1
    assert props["sample_width"] == 2  # 16-bit PCM


@requires_ffmpeg
def test_probe_video_missing_file_raises(tmp_path: Path):
    with pytest.raises(IngestionError):
        probe_video(tmp_path / "does_not_exist.mp4")


@requires_ffmpeg
def test_probe_video_without_audio_stream_raises(sample_video_no_audio: Path):
    with pytest.raises(IngestionError, match="No audio stream"):
        probe_video(sample_video_no_audio)


def test_missing_ffmpeg_binary_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(IngestionError, match="ffprobe"):
        probe_video(tmp_path / "anything.mp4")
