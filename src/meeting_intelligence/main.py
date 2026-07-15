"""CLI entry point for the Multimodal Meeting Intelligence Pipeline.

Every stage in the architecture diagram is exposed as its own subcommand,
in addition to a `pipeline` subcommand that runs all of them end to end.
Stages read their inputs from, and write their outputs to, JSON artifacts
inside `--output_dir` (see `pipeline.py`'s `ARTIFACT_*` constants), so any
subcommand can be re-run independently as long as its upstream artifacts
already exist there.

Examples
--------
Run everything at once::

    python -m meeting_intelligence.main pipeline --video_path meeting.mp4 --output_dir out/

Run stage by stage::

    python -m meeting_intelligence.main ingest --video_path meeting.mp4 --output_dir out/
    python -m meeting_intelligence.main vision --video_path meeting.mp4 --output_dir out/
    python -m meeting_intelligence.main visual-speaker --output_dir out/
    python -m meeting_intelligence.main speech --wav_path out/meeting.wav --output_dir out/
    python -m meeting_intelligence.main speaker-name --output_dir out/
    python -m meeting_intelligence.main fuse --output_dir out/
    python -m meeting_intelligence.main llm --output_dir out/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import pipeline as pl
from .asr import run_asr
from .config import LLMProvider, Settings, get_settings
from .diarization import run_diarization
from .fusion import fuse_timeline
from .ingestion import run_ingestion
from .io_utils import read_model, read_model_list, write_model, write_model_list
from .llm import FusionLLMProcessor
from .merge import merge_words_into_turns
from .models import (
    AsrSegmentResult,
    FullyAlignedTimeline,
    SceneFrame,
    SpeakerNameMap,
    SpeakerTurn,
    TranscriptFile,
    VadSegmentsFile,
    VisionTrackOutput,
    VisualSpeakerEvent,
    VisualSpeakerEventsFile,
)
from .ocr import run_ocr
from .scene_detection import detect_scenes
from .segmentation import run_vad
from .speaker_naming import build_speaker_name_map
from .transcript import build_transcript
from .visual_speaker import build_visual_speaker_events
from .zoom_layout import ZoomLayoutSettings

logger = logging.getLogger(__name__)


def _artifact(output_dir: Path, explicit: str | None, default_name: str) -> Path:
    return Path(explicit) if explicit else output_dir / default_name


def _skip(args: argparse.Namespace, path: Path, label: str) -> bool:
    """Whether a stage's own output artifact already exists and `--skip-if-exists` was passed."""
    if args.skip_if_exists and path.is_file():
        print(f"Skipping {label} (already exists): {path}")
        return True
    return False


def _read_vision_output(path: Path) -> VisionTrackOutput:
    if not path.is_file():
        logger.warning("Vision artifact not found at %s; proceeding with no visual context", path)
        return VisionTrackOutput()
    return read_model(path, VisionTrackOutput)


def _read_visual_speaker_events(path: Path) -> list[VisualSpeakerEvent]:
    if not path.is_file():
        logger.warning("Visual speaker events artifact not found at %s; proceeding with no visual speaker signal", path)
        return []
    return read_model(path, VisualSpeakerEventsFile).events


def _build_zoom_settings(args: argparse.Namespace, settings: Settings) -> ZoomLayoutSettings:
    """Start from Settings/.env, then apply any explicit CLI overrides."""
    zoom_settings = settings.zoom_layout_settings()
    if getattr(args, "no_active_speaker_detection", False):
        zoom_settings.enable_active_speaker_border = False
    if getattr(args, "no_presentation_badge_detection", False):
        zoom_settings.enable_presentation_badge = False
    for attr, cli_attr in [
        ("border_hue_min", "border_hue_min"),
        ("border_hue_max", "border_hue_max"),
        ("border_sat_min", "border_sat_min"),
        ("border_val_min", "border_val_min"),
        ("border_close_kernel_size", "border_close_kernel_size"),
        ("name_label_height_fraction", "name_label_height_fraction"),
        ("name_label_width_fraction", "name_label_width_fraction"),
        ("badge_top_fraction", "badge_top_fraction"),
        ("badge_right_fraction", "badge_right_fraction"),
        ("badge_name_height_fraction", "badge_name_height_fraction"),
    ]:
        value = getattr(args, cli_attr, None)
        if value is not None:
            setattr(zoom_settings, attr, value)
    return zoom_settings


# --------------------------------------------------------------------------
# Per-stage command handlers
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ingestion_path = output_dir / pl.ARTIFACT_INGESTION
    if _skip(args, ingestion_path, "ingestion"):
        return
    result = run_ingestion(args.video_path, output_dir)
    write_model(ingestion_path, result)
    print(f"Wrote {result.wav_path}")
    print(f"Wrote {ingestion_path}")


def cmd_scene_detect(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    scenes_path = output_dir / pl.ARTIFACT_SCENES
    if _skip(args, scenes_path, "scene detection"):
        return
    settings = get_settings()
    scenes = detect_scenes(
        args.video_path,
        output_dir,
        sample_fps=args.sample_fps or settings.scene_sample_fps,
        diff_threshold=args.diff_threshold or settings.scene_diff_threshold,
    )
    write_model_list(scenes_path, scenes)
    for scene in scenes:
        print(f"Wrote {scene.frame_path}")
    print(f"Wrote {scenes_path} ({len(scenes)} scene(s))")


def cmd_ocr(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    vision_path = output_dir / pl.ARTIFACT_VISION
    if _skip(args, vision_path, "OCR"):
        return
    settings = get_settings()
    scenes_path = _artifact(output_dir, args.scenes_json, pl.ARTIFACT_SCENES)
    scenes = read_model_list(scenes_path, SceneFrame)
    vision_output = run_ocr(
        scenes,
        use_gpt4o_fallback=not args.no_ocr_gpt4o_fallback and settings.ocr_gpt4o_fallback,
        openai_api_key=settings.openai_api_key,
        gpt4o_model=settings.openai_model,
        zoom_settings=_build_zoom_settings(args, settings),
    )
    write_model(vision_path, vision_output)
    print(f"Wrote {vision_path} ({len(vision_output.frames)} frame(s))")


def cmd_visual_speaker(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    events_path = output_dir / pl.ARTIFACT_VISUAL_SPEAKER_EVENTS
    if _skip(args, events_path, "visual speaker identification"):
        return
    vision_path = _artifact(output_dir, args.vision_json, pl.ARTIFACT_VISION)
    vision_output = _read_vision_output(vision_path)

    events = build_visual_speaker_events(vision_output.frames)
    write_model(events_path, VisualSpeakerEventsFile(events=events))
    print(f"Wrote {events_path} ({len(events)} event(s))")


def cmd_vision(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    settings = get_settings()

    scenes_path = output_dir / pl.ARTIFACT_SCENES
    if _skip(args, scenes_path, "scene detection"):
        scenes = read_model_list(scenes_path, SceneFrame)
    else:
        scenes = detect_scenes(
            args.video_path,
            output_dir,
            sample_fps=args.sample_fps or settings.scene_sample_fps,
            diff_threshold=args.diff_threshold or settings.scene_diff_threshold,
        )
        write_model_list(scenes_path, scenes)
        for scene in scenes:
            print(f"Wrote {scene.frame_path}")
        print(f"Wrote {scenes_path} ({len(scenes)} scene(s))")

    vision_path = output_dir / pl.ARTIFACT_VISION
    if _skip(args, vision_path, "OCR"):
        return
    vision_output = run_ocr(
        scenes,
        use_gpt4o_fallback=not args.no_ocr_gpt4o_fallback and settings.ocr_gpt4o_fallback,
        openai_api_key=settings.openai_api_key,
        gpt4o_model=settings.openai_model,
        zoom_settings=_build_zoom_settings(args, settings),
    )
    write_model(vision_path, vision_output)
    print(f"Wrote {vision_path} ({len(vision_output.frames)} frame(s))")


def cmd_vad(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    vad_path = output_dir / pl.ARTIFACT_VAD
    if _skip(args, vad_path, "VAD"):
        return
    segments = run_vad(args.wav_path)
    write_model(vad_path, VadSegmentsFile(segments=segments))
    print(f"Wrote {vad_path} ({len(segments)} segment(s))")


def cmd_diarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    turns_path = output_dir / pl.ARTIFACT_SPEAKER_TURNS
    if _skip(args, turns_path, "diarization"):
        return
    settings = get_settings()
    turns = run_diarization(args.wav_path, hf_token=settings.huggingface_token, duration_sec=args.duration_sec)
    write_model_list(turns_path, turns)
    print(f"Wrote {turns_path} ({len(turns)} turn(s))")


def cmd_asr(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    asr_path = output_dir / pl.ARTIFACT_ASR
    if _skip(args, asr_path, "ASR"):
        return
    settings = get_settings()

    vad_path = _artifact(output_dir, args.vad_json, pl.ARTIFACT_VAD)
    vad_segments = read_model(vad_path, VadSegmentsFile).segments

    vision_path = _artifact(output_dir, args.vision_json, pl.ARTIFACT_VISION)
    vision_frames = read_model(vision_path, VisionTrackOutput).frames if vision_path.is_file() else None

    asr_segments = run_asr(
        args.wav_path,
        vad_segments,
        vision_frames=vision_frames,
        model_size=args.model_size or settings.whisper_model_size,
        device=args.device or settings.whisper_device,
        use_openai_api=args.use_openai_api or settings.use_openai_whisper_api,
        openai_api_key=settings.openai_api_key,
    )
    write_model_list(asr_path, asr_segments)
    print(f"Wrote {asr_path} ({len(asr_segments)} chunk(s))")


def cmd_merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    transcript_path = output_dir / pl.ARTIFACT_TRANSCRIPT
    if _skip(args, transcript_path, "merge"):
        return
    settings = get_settings()
    asr_path = _artifact(output_dir, args.asr_json, pl.ARTIFACT_ASR)
    turns_path = _artifact(output_dir, args.turns_json, pl.ARTIFACT_SPEAKER_TURNS)

    asr_segments = read_model_list(asr_path, AsrSegmentResult)
    turns = read_model_list(turns_path, SpeakerTurn)

    utterances = merge_words_into_turns(asr_segments, turns)
    max_segment_duration_sec = (
        args.max_segment_duration_sec if args.max_segment_duration_sec is not None else settings.transcript_max_segment_duration_sec
    )
    transcript_segments = build_transcript(utterances, max_segment_duration_sec=max_segment_duration_sec)
    write_model(transcript_path, TranscriptFile(segments=transcript_segments))
    print(f"Wrote {transcript_path} ({len(transcript_segments)} segment(s))")


def cmd_speech(args: argparse.Namespace) -> None:
    """Convenience stage: Segmentation -> Diarization -> ASR -> Merge in one call."""
    output_dir = Path(args.output_dir)
    settings = get_settings()

    vad_path = output_dir / pl.ARTIFACT_VAD
    if _skip(args, vad_path, "VAD"):
        vad_segments = read_model(vad_path, VadSegmentsFile).segments
    else:
        vad_segments = run_vad(args.wav_path)
        write_model(vad_path, VadSegmentsFile(segments=vad_segments))

    turns_path = output_dir / pl.ARTIFACT_SPEAKER_TURNS
    if _skip(args, turns_path, "diarization"):
        speaker_turns = read_model_list(turns_path, SpeakerTurn)
    else:
        speaker_turns = run_diarization(args.wav_path, hf_token=settings.huggingface_token, duration_sec=args.duration_sec)
        write_model_list(turns_path, speaker_turns)

    transcript_path = output_dir / pl.ARTIFACT_TRANSCRIPT
    if _skip(args, transcript_path, "ASR + merge"):
        return

    asr_path = output_dir / pl.ARTIFACT_ASR
    if _skip(args, asr_path, "ASR"):
        asr_segments = read_model_list(asr_path, AsrSegmentResult)
    else:
        vision_path = _artifact(output_dir, args.vision_json, pl.ARTIFACT_VISION)
        vision_frames = read_model(vision_path, VisionTrackOutput).frames if vision_path.is_file() else None

        asr_segments = run_asr(
            args.wav_path,
            vad_segments,
            vision_frames=vision_frames,
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            use_openai_api=settings.use_openai_whisper_api,
            openai_api_key=settings.openai_api_key,
        )
        write_model_list(asr_path, asr_segments)

    utterances = merge_words_into_turns(asr_segments, speaker_turns)
    max_segment_duration_sec = (
        args.max_segment_duration_sec if args.max_segment_duration_sec is not None else settings.transcript_max_segment_duration_sec
    )
    transcript_segments = build_transcript(utterances, max_segment_duration_sec=max_segment_duration_sec)
    write_model(transcript_path, TranscriptFile(segments=transcript_segments))
    print(f"Wrote {transcript_path} ({len(transcript_segments)} segment(s))")


def cmd_speaker_name(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    speaker_map_path = output_dir / pl.ARTIFACT_SPEAKER_MAP
    if _skip(args, speaker_map_path, "speaker naming"):
        return
    settings = get_settings()
    turns_path = _artifact(output_dir, args.turns_json, pl.ARTIFACT_SPEAKER_TURNS)
    events_path = _artifact(output_dir, args.visual_events_json, pl.ARTIFACT_VISUAL_SPEAKER_EVENTS)

    turns = read_model_list(turns_path, SpeakerTurn)
    visual_speaker_events = _read_visual_speaker_events(events_path)

    speaker_map = build_speaker_name_map(
        turns,
        visual_speaker_events,
        min_speaker_confidence=args.min_confidence if args.min_confidence is not None else settings.speaker_naming_min_confidence,
        name_similarity_threshold=(
            args.name_similarity_threshold if args.name_similarity_threshold is not None else settings.speaker_naming_name_similarity_threshold
        ),
    )
    write_model(speaker_map_path, speaker_map)
    print(f"Wrote {speaker_map_path}: {speaker_map.mapping}")


def cmd_fuse(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    timeline_path = output_dir / pl.ARTIFACT_TIMELINE
    if _skip(args, timeline_path, "fusion"):
        return
    transcript_path = _artifact(output_dir, args.transcript_json, pl.ARTIFACT_TRANSCRIPT)
    vision_path = _artifact(output_dir, args.vision_json, pl.ARTIFACT_VISION)
    speaker_map_path = _artifact(output_dir, args.speaker_map_json, pl.ARTIFACT_SPEAKER_MAP)

    transcript_segments = read_model(transcript_path, TranscriptFile).segments
    vision_output = _read_vision_output(vision_path)
    speaker_map = read_model(speaker_map_path, SpeakerNameMap) if speaker_map_path.is_file() else None

    timeline = fuse_timeline(transcript_segments, vision_output.frames, speaker_map)
    write_model(timeline_path, timeline)
    print(f"Wrote {timeline_path} ({len(timeline.segments)} segment(s))")


def cmd_llm(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_path = output_dir / pl.ARTIFACT_OUTPUT
    if _skip(args, output_path, "LLM enrichment"):
        return
    settings = get_settings()
    timeline_path = _artifact(output_dir, args.timeline_json, pl.ARTIFACT_TIMELINE)
    timeline = read_model(timeline_path, FullyAlignedTimeline)

    provider = LLMProvider(args.llm_provider) if args.llm_provider else settings.llm_provider
    processor = FusionLLMProcessor(
        provider=provider,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
    )
    result = processor.enrich(timeline)
    write_model(output_path, result)
    print(f"Wrote {output_path} (llm_provider={result.llm_provider}, llm_model={result.llm_model})")


def cmd_pipeline(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.llm_provider:
        settings.llm_provider = LLMProvider(args.llm_provider)

    zoom_settings = _build_zoom_settings(args, settings)
    settings.zoom_active_speaker_detection = zoom_settings.enable_active_speaker_border
    settings.zoom_presentation_badge_detection = zoom_settings.enable_presentation_badge
    settings.zoom_border_hue_min = zoom_settings.border_hue_min
    settings.zoom_border_hue_max = zoom_settings.border_hue_max
    settings.zoom_border_sat_min = zoom_settings.border_sat_min
    settings.zoom_border_val_min = zoom_settings.border_val_min
    settings.zoom_border_close_kernel_size = zoom_settings.border_close_kernel_size
    settings.zoom_name_label_height_fraction = zoom_settings.name_label_height_fraction
    settings.zoom_name_label_width_fraction = zoom_settings.name_label_width_fraction
    settings.zoom_badge_top_fraction = zoom_settings.badge_top_fraction
    settings.zoom_badge_right_fraction = zoom_settings.badge_right_fraction
    settings.zoom_badge_name_height_fraction = zoom_settings.badge_name_height_fraction

    pl.run_full_pipeline(args.video_path, args.output_dir, settings, skip_if_exists=args.skip_if_exists)
    print(f"Wrote {Path(args.output_dir) / pl.ARTIFACT_OUTPUT}")


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-intelligence",
        description="Multimodal Meeting Intelligence Pipeline: video -> meeting_intelligence.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_output_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output_dir", required=True, help="Directory to read/write stage artifacts")

    def add_skip_if_exists(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--skip-if-exists",
            action="store_true",
            help="Skip this stage (and reuse its existing artifact) if its output file is already present under --output_dir",
        )

    def add_zoom_args(p: argparse.ArgumentParser) -> None:
        """Tunables for Zoom-specific speaker identification (see zoom_layout.py).

        Defaults come from Settings/.env; these are recalibration overrides
        for real footage where Zoom's active-speaker border color or the
        presentation badge's position/size differs from the defaults.
        """
        p.add_argument("--no-active-speaker-detection", action="store_true", help="Disable gallery-view active-speaker border detection")
        p.add_argument("--no-presentation-badge-detection", action="store_true", help="Disable presentation-mode top-right speaker badge detection")
        p.add_argument("--border-hue-min", type=int, default=None, help="Active-speaker border color: HSV hue lower bound (0-179, default 35=green)")
        p.add_argument("--border-hue-max", type=int, default=None, help="Active-speaker border color: HSV hue upper bound (0-179, default 65=green)")
        p.add_argument("--border-sat-min", type=int, default=None, help="Active-speaker border color: HSV saturation lower bound (0-255)")
        p.add_argument("--border-val-min", type=int, default=None, help="Active-speaker border color: HSV value/brightness lower bound (0-255)")
        p.add_argument(
            "--border-close-kernel-size",
            type=int,
            default=None,
            help="Morphological closing kernel (px) to bridge gaps in a broken border outline; 0 disables",
        )
        p.add_argument(
            "--name-label-height-fraction", type=float, default=None, help="Name-pill crop height within the bordered tile, as a fraction of its height"
        )
        p.add_argument(
            "--name-label-width-fraction", type=float, default=None, help="Name-pill crop width within the bordered tile, as a fraction of its width"
        )
        p.add_argument("--badge-top-fraction", type=float, default=None, help="Presentation badge crop height, as a fraction of frame height")
        p.add_argument("--badge-right-fraction", type=float, default=None, help="Presentation badge crop width, as a fraction of frame width")
        p.add_argument(
            "--badge-name-height-fraction",
            type=float,
            default=None,
            help="Name-label crop height within the presentation badge, as a fraction of the badge's height",
        )

    p_ingest = subparsers.add_parser("ingest", help="Core Setup & Ingestion: validate .mp4, extract 16kHz mono WAV")
    p_ingest.add_argument("--video_path", required=True)
    add_output_dir(p_ingest)
    add_skip_if_exists(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    p_scene = subparsers.add_parser("scene-detect", help="Part B: Scene Detection (1fps + whole-frame/regional diff)")
    p_scene.add_argument("--video_path", required=True)
    add_output_dir(p_scene)
    p_scene.add_argument("--sample-fps", type=float, default=None)
    p_scene.add_argument("--diff-threshold", type=float, default=None)
    add_skip_if_exists(p_scene)
    p_scene.set_defaults(func=cmd_scene_detect)

    p_ocr = subparsers.add_parser("ocr", help="Part B: OCR & Name Scan (PaddleOCR + GPT-4o fallback)")
    add_output_dir(p_ocr)
    p_ocr.add_argument("--scenes_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_SCENES}")
    p_ocr.add_argument(
        "--no-ocr-gpt4o-fallback",
        action="store_true",
        help=(
            "Disable GPT-4o vision in this stage entirely: no OCR-text recovery on low-confidence "
            "PaddleOCR frames, and no whole-frame vision fallback for display_name detection when the "
            "layout heuristic finds nothing. Unrelated to LLM_PROVIDER/OPENAI_MODEL (used later for "
            "meeting_intelligence.json enrichment)."
        ),
    )
    add_zoom_args(p_ocr)
    add_skip_if_exists(p_ocr)
    p_ocr.set_defaults(func=cmd_ocr)

    p_visual_speaker = subparsers.add_parser(
        "visual-speaker", help="Part C: Visual Speaker Identification (Zoom UI border/badge -> VisualSpeakerEvent)"
    )
    add_output_dir(p_visual_speaker)
    p_visual_speaker.add_argument("--vision_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_VISION}")
    add_skip_if_exists(p_visual_speaker)
    p_visual_speaker.set_defaults(func=cmd_visual_speaker)

    p_vision = subparsers.add_parser("vision", help="Part B end to end: Scene Detection -> OCR & Name Scan")
    p_vision.add_argument("--video_path", required=True)
    add_output_dir(p_vision)
    p_vision.add_argument("--sample-fps", type=float, default=None)
    p_vision.add_argument("--diff-threshold", type=float, default=None)
    p_vision.add_argument(
        "--no-ocr-gpt4o-fallback",
        action="store_true",
        help=(
            "Disable GPT-4o vision in the OCR stage entirely: no OCR-text recovery on low-confidence "
            "PaddleOCR frames, and no whole-frame vision fallback for display_name detection when the "
            "layout heuristic finds nothing. Unrelated to LLM_PROVIDER/OPENAI_MODEL (used later for "
            "meeting_intelligence.json enrichment)."
        ),
    )
    add_zoom_args(p_vision)
    add_skip_if_exists(p_vision)
    p_vision.set_defaults(func=cmd_vision)

    p_vad = subparsers.add_parser("vad", help="Part A: Segmentation (Silero VAD, default config)")
    p_vad.add_argument("--wav_path", required=True)
    add_output_dir(p_vad)
    add_skip_if_exists(p_vad)
    p_vad.set_defaults(func=cmd_vad)

    p_diarize = subparsers.add_parser("diarize", help="Part A: Diarization (pyannote.audio, single-speaker fallback)")
    p_diarize.add_argument("--wav_path", required=True)
    add_output_dir(p_diarize)
    p_diarize.add_argument("--duration-sec", type=float, default=None)
    add_skip_if_exists(p_diarize)
    p_diarize.set_defaults(func=cmd_diarize)

    p_asr = subparsers.add_parser("asr", help="Part A: ASR (faster-whisper / OpenAI Whisper API)")
    p_asr.add_argument("--wav_path", required=True)
    add_output_dir(p_asr)
    p_asr.add_argument("--vad_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_VAD}")
    p_asr.add_argument("--vision_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_VISION} (for ASR prompt hints)")
    p_asr.add_argument("--model-size", default=None)
    p_asr.add_argument("--device", default=None)
    p_asr.add_argument("--use-openai-api", action="store_true")
    add_skip_if_exists(p_asr)
    p_asr.set_defaults(func=cmd_asr)

    p_merge = subparsers.add_parser("merge", help="Part A: Merge ASR words into diarized speaker turns -> transcript.json")
    add_output_dir(p_merge)
    p_merge.add_argument("--asr_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_ASR}")
    p_merge.add_argument("--turns_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_SPEAKER_TURNS}")
    p_merge.add_argument(
        "--max-segment-duration-sec",
        type=float,
        default=None,
        help="Cap on how long (seconds) a merged same-speaker transcript segment can span (default 120); a new segment "
        "starts once adding the next utterance would exceed it, even mid-speaker",
    )
    add_skip_if_exists(p_merge)
    p_merge.set_defaults(func=cmd_merge)

    p_speech = subparsers.add_parser("speech", help="Part A end to end: Segmentation -> Diarization -> ASR -> Merge")
    p_speech.add_argument("--wav_path", required=True)
    add_output_dir(p_speech)
    p_speech.add_argument("--vision_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_VISION} (for ASR prompt hints)")
    p_speech.add_argument("--duration-sec", type=float, default=None)
    p_speech.add_argument(
        "--max-segment-duration-sec",
        type=float,
        default=None,
        help="Cap on how long (seconds) a merged same-speaker transcript segment can span (default 120); a new segment "
        "starts once adding the next utterance would exceed it, even mid-speaker",
    )
    add_skip_if_exists(p_speech)
    p_speech.set_defaults(func=cmd_speech)

    p_speaker = subparsers.add_parser("speaker-name", help="Part C: Speaker Naming (visual speaker event <-> diarization turn alignment)")
    add_output_dir(p_speaker)
    p_speaker.add_argument("--turns_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_SPEAKER_TURNS}")
    p_speaker.add_argument("--visual_events_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_VISUAL_SPEAKER_EVENTS}")
    p_speaker.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum share of a name's temporal-overlap-weighted evidence pointing at its best-match speaker (default 0.6); "
        "speakers below this are labeled 'Unknown'",
    )
    p_speaker.add_argument(
        "--name-similarity-threshold",
        type=float,
        default=None,
        help="String-similarity ratio (0-1, default 0.65) above which two OCR'd name variants (e.g. 'Maria Alvarez' / "
        "'Marla Alvarez') are treated as the same person before alignment",
    )
    add_skip_if_exists(p_speaker)
    p_speaker.set_defaults(func=cmd_speaker_name)

    p_fuse = subparsers.add_parser("fuse", help="Part C: Timeline Fusion (interval-overlap alignment)")
    add_output_dir(p_fuse)
    p_fuse.add_argument("--transcript_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_TRANSCRIPT}")
    p_fuse.add_argument("--vision_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_VISION}")
    p_fuse.add_argument("--speaker_map_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_SPEAKER_MAP}")
    add_skip_if_exists(p_fuse)
    p_fuse.set_defaults(func=cmd_fuse)

    p_llm = subparsers.add_parser("llm", help="Part C: LLM Enrichment -> meeting_intelligence.json")
    add_output_dir(p_llm)
    p_llm.add_argument("--timeline_json", default=None, help=f"Defaults to <output_dir>/{pl.ARTIFACT_TIMELINE}")
    p_llm.add_argument("--llm-provider", choices=[p.value for p in LLMProvider], default=None)
    add_skip_if_exists(p_llm)
    p_llm.set_defaults(func=cmd_llm)

    p_pipeline = subparsers.add_parser("pipeline", help="Run every stage end to end")
    p_pipeline.add_argument("--video_path", required=True)
    add_output_dir(p_pipeline)
    p_pipeline.add_argument("--llm-provider", choices=[p.value for p in LLMProvider], default=None)
    add_zoom_args(p_pipeline)
    add_skip_if_exists(p_pipeline)
    p_pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # surfaced to the user as a clean CLI error, not a traceback dump
        logger.error("%s failed: %s", args.command, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
