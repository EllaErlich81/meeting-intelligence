from __future__ import annotations

from meeting_intelligence.fusion import fuse_timeline
from meeting_intelligence.models import SlideContent, SpeakerNameMap, TranscriptSegment, VisualFrameContext


def _frame(timestamp: float, lines: list[str]) -> VisualFrameContext:
    return VisualFrameContext(
        slide_id=f"slide_{round(timestamp * 100):06d}",
        start_time=timestamp,
        frame_path="f.png",
        content=SlideContent(title=lines[0] if lines else None, raw_text="\n".join(lines)),
    )


def test_fuse_timeline_attaches_active_slide():
    segments = [TranscriptSegment(start_s=0.0, end_s=5.0, speaker_id="Speaker_00", text="Let's begin")]
    frames = [_frame(0.0, ["Intro", "Welcome"])]

    timeline = fuse_timeline(segments, frames)

    assert len(timeline.segments) == 1
    assert timeline.segments[0].slide.title == "Intro"
    assert timeline.segments[0].slide.ocr_text == ["Intro", "Welcome"]


def test_fuse_timeline_picks_slide_active_at_utterance_end_for_mid_span_transition():
    segments = [TranscriptSegment(start_s=0.0, end_s=10.0, speaker_id="Speaker_00", text="talking over a slide change")]
    frames = [
        _frame(0.0, ["Slide 1"]),
        _frame(6.0, ["Slide 2"]),
    ]

    timeline = fuse_timeline(segments, frames)
    assert timeline.segments[0].slide.title == "Slide 2"


def test_fuse_timeline_no_slide_when_no_frames_precede_utterance():
    segments = [TranscriptSegment(start_s=0.0, end_s=1.0, speaker_id="Speaker_00", text="early talk")]
    frames = [_frame(5.0, ["Later slide"])]

    timeline = fuse_timeline(segments, frames)
    assert timeline.segments[0].slide is None


def test_fuse_timeline_resolves_speaker_name_from_map():
    segments = [TranscriptSegment(start_s=0.0, end_s=1.0, speaker_id="Speaker_00", text="hi")]
    speaker_map = SpeakerNameMap(mapping={"Speaker_00": "Presenter"})

    timeline = fuse_timeline(segments, [], speaker_map)
    assert timeline.segments[0].speaker == "Presenter"


def test_fuse_timeline_falls_back_to_raw_speaker_id_without_map():
    segments = [TranscriptSegment(start_s=0.0, end_s=1.0, speaker_id="Speaker_00", text="hi")]
    timeline = fuse_timeline(segments, [])
    assert timeline.segments[0].speaker == "Speaker_00"


def test_fuse_timeline_sorts_utterances_by_start():
    segments = [
        TranscriptSegment(start_s=5.0, end_s=6.0, speaker_id="Speaker_01", text="second"),
        TranscriptSegment(start_s=0.0, end_s=1.0, speaker_id="Speaker_00", text="first"),
    ]
    timeline = fuse_timeline(segments, [])
    assert [s.transcript for s in timeline.segments] == ["first", "second"]
