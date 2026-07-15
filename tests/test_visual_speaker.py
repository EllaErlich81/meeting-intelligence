from __future__ import annotations

from meeting_intelligence.models import SlideContent, VisualFrameContext
from meeting_intelligence.visual_speaker import build_visual_speaker_events


def _frame(
    start: float,
    end: float | None,
    display_name: str | None,
    display_name_source: str | None,
    detection_confidence: float | None = 0.8,
) -> VisualFrameContext:
    return VisualFrameContext(
        slide_id=f"slide_{round(start * 100):06d}",
        start_time=start,
        end_time=end,
        frame_path=f"f_{start}.png",
        content=SlideContent(),
        display_name=display_name,
        display_name_source=display_name_source,
        detection_confidence=detection_confidence,
    )


def test_ignores_frames_not_sourced_from_a_zoom_ui_signal():
    frames = [
        _frame(0.0, 1.0, "Acme Corp", "layout_heuristic"),
        _frame(1.0, 2.0, "Jane Roe", "gpt4o_vision"),
        _frame(2.0, 3.0, None, None),
    ]
    assert build_visual_speaker_events(frames) == []


def test_active_speaker_border_maps_to_gallery_view():
    frames = [_frame(0.0, 1.0, "John Doe", "active_speaker_border")]
    events = build_visual_speaker_events(frames)
    assert len(events) == 1
    assert events[0].layout == "gallery_view"
    assert events[0].signal == "active_speaker_border"
    assert events[0].display_name == "John Doe"


def test_presentation_badge_maps_to_shared_screen():
    frames = [_frame(0.0, 1.0, "Maria Alvarez", "presentation_badge")]
    events = build_visual_speaker_events(frames)
    assert len(events) == 1
    assert events[0].layout == "shared_screen"
    assert events[0].signal == "presentation_badge"


def test_consecutive_same_name_frames_merge_into_one_event():
    frames = [
        _frame(0.0, 1.0, "John Doe", "active_speaker_border"),
        _frame(1.0, 2.0, "John Doe", "active_speaker_border"),
        _frame(2.0, 3.0, "John Doe", "active_speaker_border"),
    ]
    events = build_visual_speaker_events(frames)
    assert len(events) == 1
    assert events[0].start == 0.0
    assert events[0].end == 3.0


def test_name_change_splits_into_separate_events():
    frames = [
        _frame(0.0, 1.0, "John Doe", "active_speaker_border"),
        _frame(1.0, 2.0, "Jane Roe", "active_speaker_border"),
    ]
    events = build_visual_speaker_events(frames)
    assert [e.display_name for e in events] == ["John Doe", "Jane Roe"]
    assert events[0].end == 1.0
    assert events[1].start == 1.0


def test_signal_change_splits_into_separate_events_even_with_the_same_name():
    frames = [
        _frame(0.0, 1.0, "John Doe", "active_speaker_border"),
        _frame(1.0, 2.0, "John Doe", "presentation_badge"),
    ]
    events = build_visual_speaker_events(frames)
    assert len(events) == 2
    assert events[0].signal == "active_speaker_border"
    assert events[1].signal == "presentation_badge"


def test_non_zoom_ui_frame_in_between_splits_the_event():
    frames = [
        _frame(0.0, 1.0, "John Doe", "active_speaker_border"),
        _frame(1.0, 2.0, "Acme Corp", "layout_heuristic"),
        _frame(2.0, 3.0, "John Doe", "active_speaker_border"),
    ]
    events = build_visual_speaker_events(frames)
    assert len(events) == 2
    assert events[0].start == 0.0 and events[0].end == 1.0
    assert events[1].start == 2.0 and events[1].end == 3.0


def test_unsorted_input_frames_are_handled_in_timestamp_order():
    frames = [
        _frame(2.0, 3.0, "John Doe", "active_speaker_border"),
        _frame(0.0, 1.0, "John Doe", "active_speaker_border"),
        _frame(1.0, 2.0, "John Doe", "active_speaker_border"),
    ]
    events = build_visual_speaker_events(frames)
    assert len(events) == 1
    assert events[0].start == 0.0
    assert events[0].end == 3.0


def test_confidence_blends_ocr_confidence_signal_confidence_and_consistency():
    # A single active_speaker_border frame: ocr_confidence=1.0, signal
    # confidence fixed at 0.85, consistency = 1/3 (only one frame so far).
    frames = [_frame(0.0, 1.0, "John Doe", "active_speaker_border", detection_confidence=1.0)]
    events = build_visual_speaker_events(frames)
    expected = round(0.5 * 1.0 + 0.3 * 0.85 + 0.2 * (1 / 3), 3)
    assert events[0].confidence == expected


def test_consistency_bonus_saturates_after_three_corroborating_frames():
    frames = [_frame(float(i), float(i + 1), "John Doe", "active_speaker_border", detection_confidence=1.0) for i in range(5)]
    events = build_visual_speaker_events(frames)
    assert len(events) == 1
    expected = round(0.5 * 1.0 + 0.3 * 0.85 + 0.2 * 1.0, 3)  # consistency capped at 1.0, not 5/3
    assert events[0].confidence == expected


def test_presentation_badge_has_lower_signal_confidence_than_border():
    border_frame = _frame(0.0, 1.0, "John Doe", "active_speaker_border", detection_confidence=1.0)
    badge_frame = _frame(0.0, 1.0, "John Doe", "presentation_badge", detection_confidence=1.0)

    border_event = build_visual_speaker_events([border_frame])[0]
    badge_event = build_visual_speaker_events([badge_frame])[0]

    assert border_event.confidence > badge_event.confidence


def test_missing_detection_confidence_treated_as_zero():
    frames = [_frame(0.0, 1.0, "John Doe", "active_speaker_border", detection_confidence=None)]
    events = build_visual_speaker_events(frames)
    expected = round(0.5 * 0.0 + 0.3 * 0.85 + 0.2 * (1 / 3), 3)
    assert events[0].confidence == expected


def test_last_frame_with_no_end_time_gets_a_nonzero_default_duration():
    """Regression test: the very last sampled frame in a recording has
    end_time=None (true duration unknown). Collapsing a single-frame
    trailing event to start==end would give it exactly zero alignment
    weight everywhere in speaker_naming, silently discarding otherwise
    unambiguous evidence for whoever's on screen as the recording ends."""
    frames = [_frame(5.0, None, "John Doe", "active_speaker_border")]
    events = build_visual_speaker_events(frames)
    assert events[0].start == 5.0
    assert events[0].end > 5.0
