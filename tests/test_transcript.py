from __future__ import annotations

from meeting_intelligence.models import RawUtterance
from meeting_intelligence.transcript import build_transcript


def _utt(start, end, speaker_id, transcript, language=None):
    return RawUtterance(start=start, end=end, speaker_id=speaker_id, transcript=transcript, language=language)


def test_build_transcript_empty_input():
    assert build_transcript([]) == []


def test_build_transcript_single_utterance_becomes_one_segment():
    segments = build_transcript([_utt(0.0, 1.0, "Speaker_00", "hi there")])

    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 1.0
    assert segments[0].speaker_id == "Speaker_00"
    assert segments[0].text == "hi there"


def test_build_transcript_merges_consecutive_same_speaker_utterances():
    utterances = [
        _utt(0.0, 5.0, "Speaker_00", "first part"),
        _utt(5.2, 9.0, "Speaker_00", "second part"),
    ]

    segments = build_transcript(utterances)

    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 9.0
    assert segments[0].text == "first part second part"


def test_build_transcript_keeps_different_speakers_separate():
    utterances = [
        _utt(0.0, 5.0, "Speaker_00", "hello"),
        _utt(5.0, 9.0, "Speaker_01", "hi back"),
    ]

    segments = build_transcript(utterances)

    assert len(segments) == 2
    assert segments[0].speaker_id == "Speaker_00"
    assert segments[0].text == "hello"
    assert segments[1].speaker_id == "Speaker_01"
    assert segments[1].text == "hi back"


def test_build_transcript_does_not_merge_non_consecutive_same_speaker_turns():
    """Speaker_00 talks, then Speaker_01 interjects, then Speaker_00 talks
    again -- these are two distinct segments for Speaker_00, not merged
    across the intervening Speaker_01 turn."""
    utterances = [
        _utt(0.0, 5.0, "Speaker_00", "first"),
        _utt(5.0, 6.0, "Speaker_01", "interjection"),
        _utt(6.0, 10.0, "Speaker_00", "second"),
    ]

    segments = build_transcript(utterances)

    assert len(segments) == 3
    assert [s.speaker_id for s in segments] == ["Speaker_00", "Speaker_01", "Speaker_00"]
    assert segments[0].text == "first"
    assert segments[2].text == "second"


def test_build_transcript_merges_more_than_two_consecutive_turns():
    utterances = [
        _utt(0.0, 2.0, "Speaker_00", "one"),
        _utt(2.0, 4.0, "Speaker_00", "two"),
        _utt(4.0, 6.0, "Speaker_00", "three"),
    ]

    segments = build_transcript(utterances)

    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 6.0
    assert segments[0].text == "one two three"


def test_build_transcript_picks_majority_language_across_merged_run():
    utterances = [
        _utt(0.0, 2.0, "Speaker_00", "one", language="en"),
        _utt(2.0, 4.0, "Speaker_00", "two", language="en"),
        _utt(4.0, 6.0, "Speaker_00", "three", language="fr"),
    ]

    segments = build_transcript(utterances)

    assert segments[0].language == "en"


def test_build_transcript_language_none_when_no_utterance_has_one():
    segments = build_transcript([_utt(0.0, 1.0, "Speaker_00", "hi")])
    assert segments[0].language is None


def test_build_transcript_splits_same_speaker_run_once_duration_cap_is_exceeded():
    """A single speaker holding the floor across many consecutive
    diarization turns must not merge into one unbounded segment -- a new
    segment starts once the next utterance would push the run's span past
    max_segment_duration_sec, even though the speaker hasn't changed."""
    utterances = [
        _utt(0.0, 3.0, "Speaker_00", "one"),
        _utt(3.0, 6.0, "Speaker_00", "two"),
        _utt(6.0, 9.0, "Speaker_00", "three"),
        _utt(9.0, 12.0, "Speaker_00", "four"),
    ]

    segments = build_transcript(utterances, max_segment_duration_sec=5.0)

    # run [0,3] fits (span 3 <= 5); extending to end=6 -> span 6 > 5 (split) -> [0,3] | [3,6] fits...
    # extending to end=9 -> span 6 > 5 (split) -> [3,6] | [6,9] fits; extending to end=12 -> span 6 > 5 (split)
    assert len(segments) == 4
    assert [s.speaker_id for s in segments] == ["Speaker_00"] * 4
    assert [(s.start_s, s.end_s) for s in segments] == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.0)]


def test_build_transcript_split_points_only_fall_on_utterance_boundaries():
    """The cap only ever splits between existing utterances, never inside
    one -- a single utterance that alone exceeds the cap still becomes one
    (long) segment rather than being cut mid-text."""
    utterances = [_utt(0.0, 100.0, "Speaker_00", "a very long single utterance")]  # 100s span, alone over a 10s cap

    segments = build_transcript(utterances, max_segment_duration_sec=10.0)

    assert len(segments) == 1
    assert segments[0].text == "a very long single utterance"


def test_build_transcript_does_not_split_across_a_speaker_change_regardless_of_cap():
    utterances = [
        _utt(0.0, 1.0, "Speaker_00", "hello"),
        _utt(1.0, 2.0, "Speaker_01", "hi"),
    ]

    segments = build_transcript(utterances, max_segment_duration_sec=1000.0)

    assert len(segments) == 2
    assert [s.speaker_id for s in segments] == ["Speaker_00", "Speaker_01"]


def test_build_transcript_default_cap_does_not_split_short_runs():
    utterances = [_utt(float(i), float(i + 1), "Speaker_00", "short") for i in range(5)]
    segments = build_transcript(utterances)
    assert len(segments) == 1


def test_build_transcript_default_cap_splits_a_run_longer_than_two_minutes():
    """Reproduces the observed real-world bug: a speaker holding the floor
    across dozens of consecutive diarization turns for ~18 minutes must not
    merge into one unbounded segment under the default cap (2 minutes)."""
    utterances = [_utt(float(i * 10), float(i * 10 + 10), "Speaker_00", "chunk") for i in range(80)]  # 800s total

    segments = build_transcript(utterances)

    assert len(segments) > 1
    assert all((s.end_s - s.start_s) <= 120.0 for s in segments)
