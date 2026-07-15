"""End-to-end orchestration test: verifies stage wiring, artifact
persistence, and -- critically -- the two cross-module dependencies from
the architecture diagram (Vision frames -> ASR hints, Vision frames'
candidate_speaker -> Speaker Naming), all without touching a single real
ML model.
"""

from __future__ import annotations

import json

import meeting_intelligence.pipeline as pipeline_module
from meeting_intelligence.config import LLMProvider, Settings
from meeting_intelligence.io_utils import write_model, write_model_list
from meeting_intelligence.models import (
    AsrSegmentResult,
    FullyAlignedTimeline,
    IngestionResult,
    MeetingIntelligence,
    RawUtterance,
    SceneFrame,
    SlideContent,
    SpeakerNameMap,
    SpeakerTurn,
    TranscriptFile,
    TranscriptSegment,
    VadSegment,
    VadSegmentsFile,
    VisionTrackOutput,
    VisualFrameContext,
    VisualSpeakerEventsFile,
)


def _settings() -> Settings:
    return Settings(_env_file=None, llm_provider=LLMProvider.OPENAI, openai_api_key="sk-fake")


def test_run_full_pipeline_wires_cross_module_dependencies(tmp_path, monkeypatch):
    captured = {}

    monkeypatch.setattr(
        pipeline_module,
        "run_ingestion",
        lambda video_path, output_dir: IngestionResult(video_path=str(video_path), wav_path=str(output_dir / "a.wav"), duration_sec=12.0),
    )
    monkeypatch.setattr(
        pipeline_module,
        "detect_scenes",
        lambda video_path, output_dir, sample_fps, diff_threshold: [SceneFrame(timestamp=0.0, frame_path="f.png")],
    )

    vision_output = VisionTrackOutput(
        frames=[
            VisualFrameContext(
                slide_id="slide_000000",
                start_time=0.0,
                end_time=1.0,
                frame_path="f.png",
                content=SlideContent(title="Q1 Results", raw_text="Q1 Results\nAcme Corp"),
                display_name="John Doe",
                display_name_source="active_speaker_border",
                detection_confidence=0.9,
            )
        ],
    )
    monkeypatch.setattr(pipeline_module, "run_ocr", lambda scenes, **kw: vision_output)
    monkeypatch.setattr(pipeline_module, "run_vad", lambda wav_path: [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)])
    monkeypatch.setattr(
        pipeline_module,
        "run_diarization",
        lambda wav_path, hf_token, duration_sec: [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")],
    )

    def fake_run_asr(wav_path, vad_segments, vision_frames, model_size, device, use_openai_api, openai_api_key):
        captured["vision_frames"] = vision_frames
        return [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hello")]

    monkeypatch.setattr(pipeline_module, "run_asr", fake_run_asr)
    monkeypatch.setattr(
        pipeline_module,
        "merge_words_into_turns",
        lambda asr_segments, turns: [RawUtterance(start=0.0, end=1.0, speaker_id="Speaker_00", transcript="hello")],
    )

    def fake_build_speaker_name_map(turns, events, **kwargs):
        captured["speaker_naming_events"] = events
        return SpeakerNameMap(mapping={"Speaker_00": "John Doe"})

    monkeypatch.setattr(pipeline_module, "build_speaker_name_map", fake_build_speaker_name_map)

    class _FakeProcessor:
        def __init__(self, **kwargs):
            pass

        def enrich(self, timeline):
            captured["timeline_segments"] = len(timeline.segments)
            return MeetingIntelligence(summary="ok", topics=[], action_items=[])

    monkeypatch.setattr(pipeline_module, "FusionLLMProcessor", _FakeProcessor)

    result = pipeline_module.run_full_pipeline("video.mp4", tmp_path, _settings())

    # cross-module dependency 1: OCR frames (with their own text/timestamps) -> ASR
    assert captured["vision_frames"] == vision_output.frames
    # cross-module dependency 2: frames' Zoom-UI-sourced display_name -> visual speaker events -> speaker naming
    assert captured["speaker_naming_events"][0].display_name == "John Doe"
    assert captured["timeline_segments"] == 1
    assert result.summary == "ok"

    # every stage's artifact was persisted
    for artifact in [
        pipeline_module.ARTIFACT_INGESTION,
        pipeline_module.ARTIFACT_SCENES,
        pipeline_module.ARTIFACT_VISION,
        pipeline_module.ARTIFACT_VISUAL_SPEAKER_EVENTS,
        pipeline_module.ARTIFACT_VAD,
        pipeline_module.ARTIFACT_SPEAKER_TURNS,
        pipeline_module.ARTIFACT_ASR,
        pipeline_module.ARTIFACT_TRANSCRIPT,
        pipeline_module.ARTIFACT_SPEAKER_MAP,
        pipeline_module.ARTIFACT_TIMELINE,
        pipeline_module.ARTIFACT_OUTPUT,
    ]:
        assert (tmp_path / artifact).is_file(), f"missing artifact {artifact}"

    final_json = json.loads((tmp_path / pipeline_module.ARTIFACT_OUTPUT).read_text())
    assert final_json["summary"] == "ok"


def test_run_full_pipeline_skip_if_exists_reuses_every_cached_artifact(tmp_path, monkeypatch):
    """When every artifact is already on disk, skip_if_exists=True must load
    each one back instead of recomputing -- none of the underlying stage
    functions should be called at all."""

    def fail(name):
        def _fail(*args, **kwargs):
            raise AssertionError(f"{name} should not be called when its artifact already exists")

        return _fail

    for attr in [
        "run_ingestion",
        "detect_scenes",
        "run_ocr",
        "build_visual_speaker_events",
        "run_vad",
        "run_diarization",
        "run_asr",
        "merge_words_into_turns",
        "build_transcript",
        "build_speaker_name_map",
        "fuse_timeline",
    ]:
        monkeypatch.setattr(pipeline_module, attr, fail(attr))

    class _FailingProcessor:
        def __init__(self, **kwargs):
            raise AssertionError("FusionLLMProcessor should not be constructed when the output artifact already exists")

    monkeypatch.setattr(pipeline_module, "FusionLLMProcessor", _FailingProcessor)

    write_model(tmp_path / pipeline_module.ARTIFACT_INGESTION, IngestionResult(video_path="video.mp4", wav_path=str(tmp_path / "a.wav"), duration_sec=12.0))
    write_model_list(tmp_path / pipeline_module.ARTIFACT_SCENES, [SceneFrame(timestamp=0.0, frame_path="f.png")])
    write_model(tmp_path / pipeline_module.ARTIFACT_VISION, VisionTrackOutput())
    write_model(tmp_path / pipeline_module.ARTIFACT_VISUAL_SPEAKER_EVENTS, VisualSpeakerEventsFile(events=[]))
    write_model(tmp_path / pipeline_module.ARTIFACT_VAD, VadSegmentsFile(segments=[VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]))
    write_model_list(tmp_path / pipeline_module.ARTIFACT_SPEAKER_TURNS, [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")])
    write_model_list(tmp_path / pipeline_module.ARTIFACT_ASR, [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="cached")])
    write_model(
        tmp_path / pipeline_module.ARTIFACT_TRANSCRIPT,
        TranscriptFile(segments=[TranscriptSegment(start_s=0.0, end_s=1.0, speaker_id="Speaker_00", text="cached")]),
    )
    write_model(tmp_path / pipeline_module.ARTIFACT_SPEAKER_MAP, SpeakerNameMap(mapping={"Speaker_00": "Jane"}))
    write_model(tmp_path / pipeline_module.ARTIFACT_TIMELINE, FullyAlignedTimeline(segments=[]))
    write_model(tmp_path / pipeline_module.ARTIFACT_OUTPUT, MeetingIntelligence(summary="cached summary", topics=[], action_items=[]))

    result = pipeline_module.run_full_pipeline("video.mp4", tmp_path, _settings(), skip_if_exists=True)

    assert result.summary == "cached summary"
