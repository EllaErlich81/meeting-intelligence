from __future__ import annotations

from meeting_intelligence.merge import merge_words_into_turns
from meeting_intelligence.models import AsrSegmentResult, SpeakerTurn, Word


def test_merge_assigns_words_to_overlapping_turns():
    asr_segments = [
        AsrSegmentResult(
            segment_id="seg_0000",
            start=0.0,
            end=2.0,
            text="hi there hello world",
            words=[
                Word(start=0.0, end=0.4, text="hi"),
                Word(start=0.4, end=0.9, text="there"),
                Word(start=1.1, end=1.5, text="hello"),
                Word(start=1.5, end=1.9, text="world"),
            ],
        )
    ]
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=1.0, end=2.0, speaker_id="Speaker_01"),
    ]

    utterances = merge_words_into_turns(asr_segments, turns)

    assert len(utterances) == 2
    assert utterances[0].speaker_id == "Speaker_00"
    assert utterances[0].transcript == "hi there"
    assert utterances[1].speaker_id == "Speaker_01"
    assert utterances[1].transcript == "hello world"


def test_merge_skips_turns_with_no_words():
    asr_segments = [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hi", words=[Word(start=0.0, end=0.4, text="hi")])]
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=5.0, end=6.0, speaker_id="Speaker_01"),  # no words overlap
    ]

    utterances = merge_words_into_turns(asr_segments, turns)

    assert len(utterances) == 1
    assert utterances[0].speaker_id == "Speaker_00"


def test_merge_falls_back_to_whole_segment_text_when_no_word_timestamps():
    asr_segments = [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="whole segment text", words=[])]
    turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")]

    utterances = merge_words_into_turns(asr_segments, turns)

    assert len(utterances) == 1
    assert utterances[0].transcript == "whole segment text"


def test_merge_returns_empty_when_no_speaker_turns():
    asr_segments = [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hi", words=[Word(start=0.0, end=0.4, text="hi")])]
    assert merge_words_into_turns(asr_segments, []) == []


def test_merge_carries_over_detected_language():
    asr_segments = [
        AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hi", words=[Word(start=0.0, end=0.4, text="hi")], language="en")
    ]
    turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")]

    utterances = merge_words_into_turns(asr_segments, turns)

    assert utterances[0].language == "en"


def test_merge_picks_majority_language_when_turn_spans_multiple_chunks():
    asr_segments = [
        AsrSegmentResult(segment_id="seg_0000", start=0.0, end=0.5, text="hi", words=[Word(start=0.0, end=0.4, text="hi")], language="fr"),
        AsrSegmentResult(segment_id="seg_0001", start=0.5, end=1.0, text="there", words=[Word(start=0.5, end=0.9, text="there")], language="en"),
        AsrSegmentResult(segment_id="seg_0002", start=1.0, end=1.5, text="world", words=[Word(start=1.0, end=1.4, text="world")], language="en"),
    ]
    turns = [SpeakerTurn(start=0.0, end=1.5, speaker_id="Speaker_00")]

    utterances = merge_words_into_turns(asr_segments, turns)

    assert utterances[0].language == "en"


def test_merge_assigns_boundary_word_to_turn_with_greater_overlap():
    # Word [0.9, 1.3] overlaps turn A by 0.1s and turn B by 0.3s -> should go to B.
    asr_segments = [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=2.0, text="boundary", words=[Word(start=0.9, end=1.3, text="boundary")])]
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=1.0, end=2.0, speaker_id="Speaker_01"),
    ]

    utterances = merge_words_into_turns(asr_segments, turns)

    assert len(utterances) == 1
    assert utterances[0].speaker_id == "Speaker_01"
