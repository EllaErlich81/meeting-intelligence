"""Part A: Speech Processing orchestrator.

Chains Segmentation (VAD) -> Diarization -> ASR -> Merge into the single
`RawUtterance` list Part C fuses with the visual track. Each underlying
stage (`segmentation.py`, `diarization.py`, `asr.py`, `merge.py`) is fully
usable on its own; this module exists for convenience and for the
`speech` CLI subcommand.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .asr import run_asr
from .diarization import run_diarization
from .merge import merge_words_into_turns
from .models import RawUtterance, VisualFrameContext
from .segmentation import run_vad

logger = logging.getLogger(__name__)


def run_speech_pipeline(
    wav_path: str | Path,
    hf_token: str | None = None,
    vision_frames: list[VisualFrameContext] | None = None,
    whisper_model_size: str = "small",
    whisper_device: str = "cpu",
    use_openai_whisper_api: bool = False,
    openai_api_key: str | None = None,
) -> list[RawUtterance]:
    """Run Segmentation -> Diarization -> ASR -> Merge end to end.

    `vision_frames` is the cross-module injection point: for each ASR
    chunk, only the OCR text of the slide active at that moment (Part B)
    is used as a prompt hint -- see `asr.hint_for_chunk`.
    """
    vad_segments = run_vad(wav_path)
    speaker_turns = run_diarization(wav_path, hf_token=hf_token)
    asr_segments = run_asr(
        wav_path,
        vad_segments,
        vision_frames=vision_frames,
        model_size=whisper_model_size,
        device=whisper_device,
        use_openai_api=use_openai_whisper_api,
        openai_api_key=openai_api_key,
    )
    return merge_words_into_turns(asr_segments, speaker_turns)


async def run_speech_pipeline_async(*args, **kwargs) -> list[RawUtterance]:
    """Async wrapper so the orchestrator can run Speech alongside Vision."""
    return await asyncio.to_thread(run_speech_pipeline, *args, **kwargs)
