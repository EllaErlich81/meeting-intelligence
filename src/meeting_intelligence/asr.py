"""Part A / ASR: transcribe VAD chunks via faster-whisper (default) or the
OpenAI Whisper API.

Cross-module injection: accepts the parallel Vision track's frames and, for
each chunk, looks up the *one* slide that was actually on screen at that
point in time (via `alignment.find_active_frame`) and folds only that
slide's OCR text into Whisper's `initial_prompt` -- not every slide seen
anywhere in the recording. This keeps the hint temporally grounded (a
chunk transcribed at minute 2 is never biased by a slide shown at minute
40) and keeps the prompt short enough that Whisper actually uses all of it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .alignment import find_active_frame, sort_frames_by_timestamp
from .models import AsrSegmentResult, VadSegment, VisualFrameContext, Word

logger = logging.getLogger(__name__)


def build_prompt(hints: list[str] | None) -> str | None:
    """Collapse a slide's OCR lines into a single deduplicated prompt string."""
    if not hints:
        return None
    deduped = list(dict.fromkeys(h.strip() for h in hints if h and h.strip()))
    return ", ".join(deduped) if deduped else None


def hint_for_chunk(
    chunk_end: float,
    sorted_frames: list[VisualFrameContext],
    sorted_timestamps: list[float],
) -> list[str] | None:
    """OCR text of the slide active at `chunk_end`, or None if no frame precedes it."""
    frame = find_active_frame(chunk_end, sorted_frames, sorted_timestamps)
    return frame.content.raw_text.splitlines() if frame and frame.content.raw_text else None


def load_faster_whisper_model(model_size: str, device: str):
    from faster_whisper import WhisperModel

    compute_type = "int8" if device == "cpu" else "float16"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _load_audio_array(wav_path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def transcribe_chunk_with_faster_whisper(
    model: Any,
    audio_chunk: np.ndarray,
    hints: list[str] | None = None,
) -> tuple[str, list[Word], str | None]:
    segments, info = model.transcribe(
        audio_chunk,
        word_timestamps=True,
        initial_prompt=build_prompt(hints),
    )
    text_parts: list[str] = []
    words: list[Word] = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        for w in seg.words or []:
            words.append(Word(start=w.start, end=w.end, text=w.word.strip(), probability=getattr(w, "probability", None)))
    return " ".join(p for p in text_parts if p), words, getattr(info, "language", None)


def run_asr(
    wav_path: str | Path,
    vad_segments: list[VadSegment],
    vision_frames: list[VisualFrameContext] | None = None,
    model_size: str = "small",
    device: str = "cpu",
    use_openai_api: bool = False,
    openai_api_key: str | None = None,
    model_loader: Callable[[str, str], Any] = load_faster_whisper_model,
) -> list[AsrSegmentResult]:
    """Transcribe each VAD-detected chunk of `wav_path`.

    Word/segment timestamps returned by Whisper are chunk-relative and are
    offset here by `vad_segment.start` so all downstream stages work in
    absolute recording time.
    """
    if not vad_segments:
        logger.warning("No VAD segments provided to ASR; returning no transcriptions")
        return []

    sorted_frames, sorted_timestamps = sort_frames_by_timestamp(vision_frames or [])

    if use_openai_api:
        return _run_openai_whisper_api(wav_path, vad_segments, sorted_frames, sorted_timestamps, api_key=openai_api_key)

    audio, sample_rate = _load_audio_array(wav_path)
    model = model_loader(model_size, device)

    results: list[AsrSegmentResult] = []
    for seg in vad_segments:
        start_sample = int(seg.start_s * sample_rate)
        end_sample = int(seg.end_s * sample_rate)
        chunk = audio[start_sample:end_sample]
        if chunk.size == 0:
            continue

        hint = hint_for_chunk(seg.end_s, sorted_frames, sorted_timestamps)
        prompt = build_prompt(hint)
        logger.info("ASR segment %s [%.3fs-%.3fs]: initial_prompt=%r", seg.segment_id, seg.start_s, seg.end_s, prompt)
        text, words, language = transcribe_chunk_with_faster_whisper(model, chunk, hints=hint)
        offset_words = [Word(start=w.start + seg.start_s, end=w.end + seg.start_s, text=w.text, probability=w.probability) for w in words]
        results.append(
            AsrSegmentResult(segment_id=seg.segment_id, start=seg.start_s, end=seg.end_s, text=text, words=offset_words, language=language)
        )

    logger.info("ASR transcribed %d/%d chunk(s)", len(results), len(vad_segments))
    return results


def _run_openai_whisper_api(
    wav_path: str | Path,
    vad_segments: list[VadSegment],
    sorted_frames: list[VisualFrameContext],
    sorted_timestamps: list[float],
    api_key: str | None,
    client_factory: Callable[[str | None], Any] | None = None,
) -> list[AsrSegmentResult]:
    if client_factory is None:
        def client_factory(key: str | None):
            from openai import OpenAI

            return OpenAI(api_key=key)

    import io
    import soundfile as sf

    client = client_factory(api_key)
    audio, sample_rate = _load_audio_array(wav_path)

    results: list[AsrSegmentResult] = []
    for seg in vad_segments:
        start_sample = int(seg.start_s * sample_rate)
        end_sample = int(seg.end_s * sample_rate)
        chunk = audio[start_sample:end_sample]
        if chunk.size == 0:
            continue

        buffer = io.BytesIO()
        sf.write(buffer, chunk, sample_rate, format="WAV")
        buffer.seek(0)
        buffer.name = "chunk.wav"

        prompt = build_prompt(hint_for_chunk(seg.end_s, sorted_frames, sorted_timestamps))
        logger.info("ASR segment %s [%.3fs-%.3fs]: initial_prompt=%r", seg.segment_id, seg.start_s, seg.end_s, prompt)
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            response_format="verbose_json",
            prompt=prompt,
        )
        results.append(
            AsrSegmentResult(
                segment_id=seg.segment_id,
                start=seg.start_s,
                end=seg.end_s,
                text=transcript.text.strip(),
                words=[],
                language=getattr(transcript, "language", None),
            )
        )

    return results
