"""Part A / Segmentation: Voice Activity Detection via Silero VAD.

Uses Silero VAD's default configuration (as required by the brief) to
partition the full-length WAV into clean speech chunks, which are then fed
individually into the ASR stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .models import VadSegment

logger = logging.getLogger(__name__)

ModelBundle = tuple[Any, dict[str, Callable]]


def load_silero_vad() -> ModelBundle:
    """Load Silero VAD via torch.hub with default configuration.

    Isolated into its own function (rather than inlined in `run_vad`) so
    tests can substitute a fake loader and avoid a network call / torch
    dependency entirely.
    """
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    get_speech_timestamps, _save_audio, read_audio, *_rest = utils
    return model, {"get_speech_timestamps": get_speech_timestamps, "read_audio": read_audio}


def run_vad(
    wav_path: str | Path,
    sample_rate: int = 16_000,
    model_loader: Callable[[], ModelBundle] = load_silero_vad,
) -> list[VadSegment]:
    """Partition `wav_path` into speech-active chunks using Silero VAD defaults."""
    model, utils = model_loader()
    wav = utils["read_audio"](str(wav_path), sampling_rate=sample_rate)
    timestamps = utils["get_speech_timestamps"](wav, model, sampling_rate=sample_rate)

    segments = []
    for ts in timestamps:
        start_s = ts["start"] / sample_rate
        end_s = ts["end"] / sample_rate
        # Encode the segment's own boundaries into its id as zero-padded
        # centiseconds (e.g. 1.85s -> "000185"), not a plain running
        # counter, so the id is self-describing, sorts lexicographically in
        # time order, and stays free of "." characters.
        segment_id = f"seg_{round(start_s * 100):06d}-{round(end_s * 100):06d}"
        # start_s/duration are rounded to milliseconds by VadSegment's
        # RoundedFloat fields, clearing binary floating-point noise (e.g.
        # 0.7319999999999998) from sample-count division.
        segments.append(VadSegment(segment_id=segment_id, start_s=start_s, duration=end_s - start_s))
    logger.info("Silero VAD found %d speech segment(s) in %s", len(segments), wav_path)
    if not segments:
        logger.warning("VAD found no speech in %s; downstream stages will receive no audio", wav_path)
    return segments
