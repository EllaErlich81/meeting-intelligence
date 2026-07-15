"""Data contracts must enforce the fields the brief specifies as required."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from meeting_intelligence.models import (
    ActionItem,
    AlignedMeetingSegment,
    Evidence,
    FullyAlignedTimeline,
    IngestionResult,
    MeetingIntelligence,
    RawUtterance,
    SceneFrame,
    SlideContent,
    SlideContext,
    SpeakerTurn,
    TopicItem,
    VadSegment,
    VisualFrameContext,
    Word,
)


def test_visual_frame_context_requires_slide_id_start_time_frame_path_and_content():
    with pytest.raises(ValidationError):
        VisualFrameContext(start_time=1.0)  # missing slide_id, frame_path, content


def test_visual_frame_context_optional_fields_default_to_none():
    ctx = VisualFrameContext(slide_id="slide_000100", start_time=1.0, frame_path="frame.png", content=SlideContent(raw_text="hello"))
    assert ctx.end_time is None
    assert ctx.content.title is None
    assert ctx.display_name is None
    assert ctx.ocr_confidence is None
    assert ctx.detection_confidence is None


def test_raw_utterance_roundtrip():
    utt = RawUtterance(start=0.0, end=1.5, speaker_id="Speaker_00", transcript="hello there")
    assert RawUtterance.model_validate_json(utt.model_dump_json()) == utt


def test_aligned_meeting_segment_slide_optional():
    seg = AlignedMeetingSegment(start=0.0, end=1.0, speaker="Presenter", transcript="hi")
    assert seg.slide is None

    seg_with_slide = AlignedMeetingSegment(
        start=0.0,
        end=1.0,
        speaker="Presenter",
        transcript="hi",
        slide=SlideContext(title="Q1 Results", ocr_text=["Q1 Results", "Revenue up 12%"]),
    )
    assert seg_with_slide.slide.ocr_text == ["Q1 Results", "Revenue up 12%"]


def test_fully_aligned_timeline_serialization():
    timeline = FullyAlignedTimeline(
        segments=[AlignedMeetingSegment(start=0.0, end=1.0, speaker="Speaker_00", transcript="hi")]
    )
    dumped = timeline.model_dump()
    assert dumped["segments"][0]["speaker"] == "Speaker_00"


def test_meeting_intelligence_matches_spec_shape():
    mi = MeetingIntelligence(
        summary="Team reviewed Q1 numbers.",
        topics=[
            TopicItem(
                topic="Weekly card transactions",
                summary="Domestic and international trends reviewed.",
                evidence=Evidence(timestamps=[172.5, 201.3], speakers=["Presenter"], visual_reference="Weekly card transactions"),
            )
        ],
        action_items=[
            ActionItem(task="Review processing caps", assignee="Presenter", evidence_quote="Let's review the weekly card transactions...")
        ],
    )
    payload = mi.model_dump()
    # llm_provider/llm_model are a user-requested extension past the literal
    # spec (see models.py's module docstring), set by FusionLLMProcessor
    # after parsing -- not part of what the brief originally specified.
    assert set(payload.keys()) == {"summary", "topics", "action_items", "llm_provider", "llm_model"}
    assert payload["topics"][0]["evidence"]["timestamps"] == [172.5, 201.3]


def test_action_item_assignee_optional():
    item = ActionItem(task="Follow up", evidence_quote="we should follow up")
    assert item.assignee is None


# --------------------------------------------------------------------------
# RoundedFloat: every measured/computed float is rounded to 3 decimals,
# clearing binary floating-point noise regardless of which stage
# constructs the model.
# --------------------------------------------------------------------------


def test_raw_utterance_rounds_start_and_end():
    utt = RawUtterance(start=0.7319999999999998, end=1.0000000000000002, speaker_id="Speaker_00", transcript="hi")
    assert utt.start == 0.732
    assert utt.end == 1.0


def test_aligned_meeting_segment_rounds_start_and_end():
    seg = AlignedMeetingSegment(start=0.12344999, end=0.98765, speaker="Presenter", transcript="hi")
    assert seg.start == 0.123
    assert seg.end == 0.988


def test_evidence_rounds_each_timestamp():
    evidence = Evidence(timestamps=[172.54999, 201.30001], speakers=["Presenter"])
    assert evidence.timestamps == [172.55, 201.3]


def test_ingestion_result_rounds_duration_sec():
    result = IngestionResult(video_path="v.mp4", wav_path="a.wav", duration_sec=12.3456789)
    assert result.duration_sec == 12.346


def test_vad_segment_rounds_start_s_and_duration_and_computed_end_s():
    seg = VadSegment(segment_id="seg_000000-000100", start_s=0.0339999999999998, duration=0.7319999999999998)
    assert seg.start_s == 0.034
    assert seg.duration == 0.732
    assert seg.end_s == 0.766


def test_speaker_turn_rounds_start_and_end():
    turn = SpeakerTurn(start=0.033333333, end=1.666666666, speaker_id="Speaker_00")
    assert turn.start == 0.033
    assert turn.end == 1.667


def test_word_rounds_start_end_and_probability():
    word = Word(start=0.10001, end=0.20009, text="hi", probability=0.987654)
    assert word.start == 0.1
    assert word.end == 0.2
    assert word.probability == 0.988


def test_scene_frame_rounds_timestamp():
    frame = SceneFrame(timestamp=3.333333333, frame_path="f.png")
    assert frame.timestamp == 3.333


def test_visual_frame_context_rounds_times_and_confidences():
    ctx = VisualFrameContext(
        slide_id="slide_000123",
        start_time=1.2344999,
        end_time=5.6789999,
        frame_path="f.png",
        content=SlideContent(),
        ocr_confidence=0.987654,
        detection_confidence=0.123456,
    )
    assert ctx.start_time == 1.234
    assert ctx.end_time == 5.679
    assert ctx.ocr_confidence == 0.988
    assert ctx.detection_confidence == 0.123
