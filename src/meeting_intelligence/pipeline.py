"""End-to-end orchestrator.

Wires together every stage in the architecture diagram, including its two
cross-module dependencies:

* Vision's OCR text -> ASR prompt hint (Part B -> Part A)
* Vision's frames -> Visual Speaker Identification -> Speaker Naming (Part B -> Part C)

Every stage's output is also persisted as a JSON artifact under
`output_dir` (see `io_utils.py` and the `ARTIFACT_*` constants below), so
the same run can be resumed, inspected, or re-driven one stage at a time
through the CLI in `main.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from .asr import run_asr
from .config import Settings
from .diarization import run_diarization
from .fusion import fuse_timeline
from .ingestion import run_ingestion
from .io_utils import read_model, read_model_list, write_model, write_model_list
from .llm import FusionLLMProcessor
from .merge import merge_words_into_turns
from .models import (
    AsrSegmentResult,
    FullyAlignedTimeline,
    IngestionResult,
    MeetingIntelligence,
    SceneFrame,
    SpeakerNameMap,
    SpeakerTurn,
    TranscriptFile,
    VadSegmentsFile,
    VisionTrackOutput,
    VisualSpeakerEventsFile,
)
from .ocr import run_ocr
from .scene_detection import detect_scenes
from .segmentation import run_vad
from .speaker_naming import build_speaker_name_map
from .transcript import build_transcript
from .visual_speaker import build_visual_speaker_events

logger = logging.getLogger(__name__)

# Standard artifact filenames written into --output_dir by each stage.
ARTIFACT_INGESTION = "ingestion.json"
ARTIFACT_SCENES = "scenes.json"
ARTIFACT_VISION = "vision_track.json"
ARTIFACT_VISUAL_SPEAKER_EVENTS = "visual_speaker_events.json"
ARTIFACT_VAD = "vad_segments.json"
ARTIFACT_SPEAKER_TURNS = "speaker_turns.json"
ARTIFACT_ASR = "asr_segments.json"
ARTIFACT_TRANSCRIPT = "transcript.json"
ARTIFACT_SPEAKER_MAP = "speaker_name_map.json"
ARTIFACT_TIMELINE = "fused_timeline.json"
ARTIFACT_OUTPUT = "meeting_intelligence.json"


def _write(path: Path, model: BaseModel) -> None:
    write_model(path, model)
    logger.info("Wrote %s", path)


def _write_list(path: Path, models: list[BaseModel]) -> None:
    write_model_list(path, models)
    logger.info("Wrote %s (%d item(s))", path, len(models))


def run_full_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    settings: Settings,
    skip_if_exists: bool = False,
) -> MeetingIntelligence:
    """Run every stage in sequence and return the final `MeetingIntelligence`.

    When `skip_if_exists` is set, a stage whose own output artifact is
    already present under `output_dir` is loaded from disk instead of
    recomputed -- lets a run be resumed after a downstream failure (or
    re-driven with different settings for a later stage) without redoing
    expensive upstream stages like ASR, diarization, or OCR.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _skip(path: Path) -> bool:
        if skip_if_exists and path.is_file():
            logger.info("Skipping (already exists): %s", path)
            return True
        return False

    logger.info("[1/9] Ingestion")
    ingestion_path = output_dir / ARTIFACT_INGESTION
    if _skip(ingestion_path):
        ingestion_result = read_model(ingestion_path, IngestionResult)
    else:
        ingestion_result = run_ingestion(video_path, output_dir)
        logger.info("Wrote %s", ingestion_result.wav_path)
        _write(ingestion_path, ingestion_result)

    logger.info("[2/9] Vision: Scene Detection")
    scenes_path = output_dir / ARTIFACT_SCENES
    if _skip(scenes_path):
        scenes = read_model_list(scenes_path, SceneFrame)
    else:
        scenes = detect_scenes(
            video_path, output_dir, sample_fps=settings.scene_sample_fps, diff_threshold=settings.scene_diff_threshold
        )
        logger.info("Wrote %d frame(s) under %s", len(scenes), output_dir / "frames")
        _write_list(scenes_path, scenes)

    logger.info("[3/9] Vision: OCR & Name Scan")
    vision_path = output_dir / ARTIFACT_VISION
    if _skip(vision_path):
        vision_output = read_model(vision_path, VisionTrackOutput)
    else:
        vision_output = run_ocr(
            scenes,
            use_gpt4o_fallback=settings.ocr_gpt4o_fallback,
            openai_api_key=settings.openai_api_key,
            gpt4o_model=settings.openai_model,
            zoom_settings=settings.zoom_layout_settings(),
        )
        _write(vision_path, vision_output)

    logger.info("[4/9] Visual Speaker Identification (Zoom UI border/badge -> VisualSpeakerEvent)")
    visual_events_path = output_dir / ARTIFACT_VISUAL_SPEAKER_EVENTS
    if _skip(visual_events_path):
        visual_speaker_events = read_model(visual_events_path, VisualSpeakerEventsFile).events
    else:
        visual_speaker_events = build_visual_speaker_events(vision_output.frames)
        _write(visual_events_path, VisualSpeakerEventsFile(events=visual_speaker_events))

    logger.info("[5/9] Speech: Segmentation (VAD) + Diarization")
    vad_path = output_dir / ARTIFACT_VAD
    if _skip(vad_path):
        vad_segments = read_model(vad_path, VadSegmentsFile).segments
    else:
        vad_segments = run_vad(ingestion_result.wav_path)
        _write(vad_path, VadSegmentsFile(segments=vad_segments))

    turns_path = output_dir / ARTIFACT_SPEAKER_TURNS
    if _skip(turns_path):
        speaker_turns = read_model_list(turns_path, SpeakerTurn)
    else:
        speaker_turns = run_diarization(
            ingestion_result.wav_path,
            hf_token=settings.huggingface_token,
            duration_sec=ingestion_result.duration_sec,
        )
        _write_list(turns_path, speaker_turns)

    logger.info("[6/9] Speech: ASR (per-chunk hint from the slide active at that moment) + Merge")
    asr_path = output_dir / ARTIFACT_ASR
    transcript_path = output_dir / ARTIFACT_TRANSCRIPT
    if _skip(transcript_path):
        transcript_segments = read_model(transcript_path, TranscriptFile).segments
    else:
        if _skip(asr_path):
            asr_segments = read_model_list(asr_path, AsrSegmentResult)
        else:
            asr_segments = run_asr(
                ingestion_result.wav_path,
                vad_segments,
                vision_frames=vision_output.frames,
                model_size=settings.whisper_model_size,
                device=settings.whisper_device,
                use_openai_api=settings.use_openai_whisper_api,
                openai_api_key=settings.openai_api_key,
                use_initial_prompt=settings.asr_use_initial_prompt,
            )
            _write_list(asr_path, asr_segments)

        utterances = merge_words_into_turns(asr_segments, speaker_turns)
        transcript_segments = build_transcript(utterances, max_segment_duration_sec=settings.transcript_max_segment_duration_sec)
        _write(transcript_path, TranscriptFile(segments=transcript_segments))

    logger.info("[7/9] Speaker Naming (visual speaker event <-> diarization turn alignment)")
    speaker_map_path = output_dir / ARTIFACT_SPEAKER_MAP
    if _skip(speaker_map_path):
        speaker_name_map = read_model(speaker_map_path, SpeakerNameMap)
    else:
        speaker_name_map = build_speaker_name_map(
            speaker_turns,
            visual_speaker_events,
            min_speaker_confidence=settings.speaker_naming_min_confidence,
            name_similarity_threshold=settings.speaker_naming_name_similarity_threshold,
        )
        _write(speaker_map_path, speaker_name_map)

    logger.info("[8/9] Fusion (interval-overlap timeline alignment)")
    timeline_path = output_dir / ARTIFACT_TIMELINE
    if _skip(timeline_path):
        timeline = read_model(timeline_path, FullyAlignedTimeline)
    else:
        timeline = fuse_timeline(transcript_segments, vision_output.frames, speaker_name_map)
        _write(timeline_path, timeline)

    logger.info("[9/9] LLM Enrichment (%s)", settings.llm_provider.value)
    output_path = output_dir / ARTIFACT_OUTPUT
    if _skip(output_path):
        meeting_intelligence = read_model(output_path, MeetingIntelligence)
    else:
        processor = FusionLLMProcessor(
            provider=settings.llm_provider,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
        )
        meeting_intelligence = processor.enrich(timeline)
        _write(output_path, meeting_intelligence)

    logger.info("Pipeline complete -> %s", output_dir / ARTIFACT_OUTPUT)
    return meeting_intelligence
