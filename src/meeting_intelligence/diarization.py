"""Part A / Diarization: speaker turns via pyannote.audio.

Per the brief's "graceful degradation" requirement: if pyannote fails to
load or to diarize the file, the stage falls back to treating the
recording as a single speaker (SPEAKER_00) spanning its full duration,
rather than raising and breaking the pipeline.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Callable

from .models import SpeakerTurn

logger = logging.getLogger(__name__)

FALLBACK_SPEAKER_ID = "Speaker_00"


def load_diarization_pipeline(hf_token: str | None):
    """Load the pretrained pyannote speaker-diarization pipeline.

    Isolated so tests can inject a fake pipeline without a Hugging Face
    token or network access.
    """
    from pyannote.audio import Pipeline

    if not hf_token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is required to download pyannote/speaker-diarization-3.1"
        )
    # pyannote.audio 3.x used `use_auth_token=`; 4.x renamed it to `token=`.
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)


def _speaker_label(raw_label: str, speaker_index: dict[str, int]) -> str:
    if raw_label not in speaker_index:
        speaker_index[raw_label] = len(speaker_index)
    return f"Speaker_{speaker_index[raw_label]:02d}"


def _probe_duration(wav_path: str | Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def single_speaker_fallback(wav_path: str | Path, duration_sec: float | None = None) -> list[SpeakerTurn]:
    if duration_sec is None:
        duration_sec = _probe_duration(wav_path)
    return [SpeakerTurn(start=0.0, end=duration_sec, speaker_id=FALLBACK_SPEAKER_ID)]


def run_diarization(
    wav_path: str | Path,
    hf_token: str | None = None,
    duration_sec: float | None = None,
    pipeline_loader: Callable[[str | None], object] = load_diarization_pipeline,
) -> list[SpeakerTurn]:
    """Diarize `wav_path` into speaker turns, or fall back to single-speaker mode."""
    try:
        pipeline = pipeline_loader(hf_token)
        result = pipeline(str(wav_path))
        # pyannote.audio 3.x's pipeline call returns an `Annotation` directly;
        # 4.x wraps it in a `DiarizeOutput` dataclass with a
        # `.speaker_diarization` attribute. `getattr` with a fallback to the
        # result itself supports both without a hard version pin.
        annotation = getattr(result, "speaker_diarization", result)
    except Exception:
        logger.exception("Diarization failed; falling back to single-speaker mode (%s)", FALLBACK_SPEAKER_ID)
        return single_speaker_fallback(wav_path, duration_sec)

    speaker_index: dict[str, int] = {}
    turns = [
        SpeakerTurn(start=segment.start, end=segment.end, speaker_id=_speaker_label(raw_label, speaker_index))
        for segment, _track, raw_label in annotation.itertracks(yield_label=True)
    ]

    if not turns:
        logger.warning("Diarization produced no turns; falling back to single-speaker mode")
        return single_speaker_fallback(wav_path, duration_sec)

    logger.info("Diarization found %d speaker(s) across %d turn(s)", len(speaker_index), len(turns))
    return turns
