from __future__ import annotations

from meeting_intelligence.models import SpeakerTurn, VisualSpeakerEvent
from meeting_intelligence.speaker_naming import UNKNOWN_SPEAKER_LABEL, _canonicalize_names, build_speaker_name_map


def _event(name: str, start: float, end: float, confidence: float = 0.9, signal: str = "active_speaker_border") -> VisualSpeakerEvent:
    layout = "gallery_view" if signal == "active_speaker_border" else "shared_screen"
    return VisualSpeakerEvent(start=start, end=end, display_name=name, layout=layout, signal=signal, confidence=confidence)


def test_resolves_speaker_to_fully_overlapping_high_confidence_event():
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker_id="Speaker_01"),
    ]
    events = [
        _event("John Doe", 1.0, 3.0),
        _event("Jane Roe", 11.0, 13.0),
    ]

    result = build_speaker_name_map(turns, events)

    assert result.mapping["Speaker_00"] == "John Doe"
    assert result.mapping["Speaker_01"] == "Jane Roe"


def test_unresolved_speaker_is_labeled_unknown():
    turns = [SpeakerTurn(start=0.0, end=10.0, speaker_id="Speaker_00")]
    result = build_speaker_name_map(turns, visual_speaker_events=[])
    assert result.mapping["Speaker_00"] == UNKNOWN_SPEAKER_LABEL


def test_event_with_no_overlapping_turn_is_ignored():
    turns = [SpeakerTurn(start=0.0, end=10.0, speaker_id="Speaker_00")]
    events = [_event("John Doe", 100.0, 105.0)]  # nowhere near any turn

    result = build_speaker_name_map(turns, events)
    assert result.mapping["Speaker_00"] == UNKNOWN_SPEAKER_LABEL


def test_event_spanning_a_turn_boundary_splits_weight_by_overlap_fraction():
    """An event that only partially overlaps a turn should contribute
    proportionally less evidence than one fully contained in it."""
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker_id="Speaker_01"),
    ]
    # 9s in Speaker_00's turn, 1s in Speaker_01's -- Speaker_00 should win.
    events = [_event("John Doe", 1.0, 11.0)]

    result = build_speaker_name_map(turns, events)

    assert result.mapping["Speaker_00"] == "John Doe"
    assert result.mapping["Speaker_01"] == UNKNOWN_SPEAKER_LABEL


def test_name_appearing_equally_near_two_speakers_is_left_unknown():
    """A name split evenly across two speakers' turns must not be
    confidently -- and arbitrarily -- assigned to either one."""
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker_id="Speaker_01"),
    ]
    events = [
        _event("Acme Corp", 4.0, 6.0),
        _event("Acme Corp", 14.0, 16.0),
    ]

    result = build_speaker_name_map(turns, events)

    assert result.mapping["Speaker_00"] == UNKNOWN_SPEAKER_LABEL
    assert result.mapping["Speaker_01"] == UNKNOWN_SPEAKER_LABEL


def test_low_confidence_event_alone_does_not_clear_threshold_against_noise():
    """A single low-confidence event competing against other evidence for
    the same name shouldn't dominate enough to resolve confidently."""
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker_id="Speaker_01"),
    ]
    events = [
        _event("Diego Santos", 1.0, 3.0, confidence=0.3, signal="presentation_badge"),
        _event("Diego Santos", 11.0, 13.0, confidence=0.9, signal="active_speaker_border"),
    ]

    result = build_speaker_name_map(turns, events, min_speaker_confidence=0.6)

    # Speaker_01's event carries far more weight (0.9 vs 0.3), so "Diego
    # Santos" resolves there, not to Speaker_00.
    assert result.mapping["Speaker_01"] == "Diego Santos"
    assert result.mapping["Speaker_00"] == UNKNOWN_SPEAKER_LABEL


def test_higher_min_confidence_makes_resolution_stricter():
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker_id="Speaker_01"),
    ]
    events = [
        _event("John Doe", 1.0, 3.0, confidence=0.9),
        _event("John Doe", 12.0, 12.5, confidence=0.2),  # a little noise near Speaker_01 too
    ]

    lenient = build_speaker_name_map(turns, events, min_speaker_confidence=0.5)
    assert lenient.mapping["Speaker_00"] == "John Doe"

    strict = build_speaker_name_map(turns, events, min_speaker_confidence=0.99)
    assert strict.mapping["Speaker_00"] == UNKNOWN_SPEAKER_LABEL


def test_two_names_resolving_to_the_same_speaker_keeps_the_one_with_more_total_evidence():
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker_id="Speaker_01"),
    ]
    events = [
        _event("John Doe", 1.0, 2.0, confidence=0.95),  # only ever seen near Speaker_00 -> weight 0.95, confidence 1.0
        # "Presenter" is textually dissimilar to "John Doe" (must not canonicalize together -- this test
        # exercises the *dedup* step, not name-similarity clustering; see the _canonicalize_names tests below).
        _event("Presenter", 3.0, 4.0, confidence=0.9),  # mostly Speaker_00 (weight 0.9)...
        _event("Presenter", 15.0, 15.5, confidence=0.3),  # ...but some competing noise near Speaker_01 too
    ]

    result = build_speaker_name_map(turns, events)
    # "John Doe" wins on both criteria here (weight 0.95 > 0.9, confidence 1.0 > 0.75) --
    # see the next test for a case where they disagree and weight must win.
    assert result.mapping["Speaker_00"] == "John Doe"


def test_dedup_prefers_more_total_evidence_over_a_trivially_uncontested_single_event():
    """Regression test for real footage: a single isolated event is
    trivially 100% "confident" (its only speaker has 100% of its weight --
    there's no competing evidence to dilute the ratio), but that's not more
    trustworthy than a well-corroborated cluster of many events that happens
    to have a little competing noise elsewhere. The dedup step must pick by
    total weight of evidence for the winning speaker, not the confidence
    ratio alone."""
    turns = [
        SpeakerTurn(start=0.0, end=100.0, speaker_id="Speaker_00"),
        SpeakerTurn(start=200.0, end=250.0, speaker_id="Speaker_01"),
    ]
    events = [
        # "Isolated Name" -- one clean event fully inside Speaker_00's turn -> trivially
        # confidence=1.0 (its only speaker), but weight is just its own confidence (1.0).
        _event("Isolated Name", 0.0, 1.0, confidence=1.0, signal="presentation_badge"),
        # "Well Corroborated" -- three events fully inside Speaker_00 (weight 0.9 each,
        # total 2.7) plus one stray event fully inside the unrelated Speaker_01 turn
        # (weight 0.9) drags the Speaker_00 ratio down to 2.7/3.6=0.75 -- still clears
        # min_speaker_confidence, still less than "Isolated Name"'s 1.0, but backed by
        # far more real evidence (weight 2.7 vs. 1.0).
        _event("Well Corroborated", 10.0, 20.0, confidence=0.9),
        _event("Well Corroborated", 30.0, 40.0, confidence=0.9),
        _event("Well Corroborated", 50.0, 60.0, confidence=0.9),
        _event("Well Corroborated", 210.0, 220.0, confidence=0.9),  # stray, inside Speaker_01's turn
    ]

    result = build_speaker_name_map(turns, events, min_speaker_confidence=0.6)
    assert result.mapping["Speaker_00"] == "Well Corroborated"


def test_canonicalize_names_clusters_similar_ocr_variants():
    """A person's badge/border name-crop OCR'd slightly differently across
    frames (misread letters, dropped/substituted characters) must cluster
    into one name."""
    events = [
        _event("Maria Alvarez", 0.0, 27.0, confidence=0.793),
        _event("Marla Alvarez", 27.0, 42.0, confidence=0.747),
        _event("Maria Alvares", 96.5, 141.5, confidence=0.931),  # highest total confidence -> canonical
        _event("Mara Alvarez", 147.0, 159.0, confidence=0.666),
    ]

    canonical_for = _canonicalize_names(events, similarity_threshold=0.65)

    assert len({canonical_for[e.display_name] for e in events}) == 1
    assert canonical_for["Maria Alvarez"] == "Maria Alvares"


def test_canonicalize_names_does_not_merge_different_people():
    events = [
        _event("Maria Alvarez", 0.0, 10.0),
        _event("Diego Santos", 10.0, 20.0),
        _event("Priya Nair", 20.0, 30.0),
    ]

    canonical_for = _canonicalize_names(events, similarity_threshold=0.65)

    assert len({canonical_for[e.display_name] for e in events}) == 3


def test_canonicalize_names_picks_the_highest_total_confidence_variant_as_canonical():
    events = [
        _event("Alexander K.", 0.0, 1.0, confidence=0.5),
        _event("Alexander K..", 1.0, 2.0, confidence=0.5),
        _event("Alexander K...", 2.0, 3.0, confidence=0.9),  # single highest, but...
        _event("Alexander K", 3.0, 4.0, confidence=0.5),
        _event("Alexander K", 4.0, 5.0, confidence=0.5),  # ...appears twice, so higher *total* confidence (1.0)
    ]

    canonical_for = _canonicalize_names(events, similarity_threshold=0.65)

    assert canonical_for["Alexander K."] == "Alexander K"


def test_canonicalize_names_joins_the_closest_cluster_not_the_first_eligible_one():
    """Regression test: a name within similarity_threshold of more than one
    existing canonical must join whichever it's *closest* to, not whichever
    canonical happened to be accepted first (an accident of confidence
    ordering, unrelated to string similarity)."""
    events = [
        _event("Ana Lima", 0.0, 1.0, confidence=0.9),  # processed first (highest confidence) -> canonical
        _event("Ana Rios", 1.0, 2.0, confidence=0.7),  # dissimilar to "Ana Lima" (0.625 < 0.65) -> its own canonical
        # "Ana Lios" is within threshold of *both* ("Ana Lima"=0.75, "Ana Rios"=0.875) -- must join "Ana Rios",
        # the closer one, even though "Ana Lima" was accepted as a canonical first.
        _event("Ana Lios", 2.0, 3.0, confidence=0.3),
    ]

    canonical_for = _canonicalize_names(events, similarity_threshold=0.65)

    assert canonical_for["Ana Lima"] == "Ana Lima"
    assert canonical_for["Ana Rios"] == "Ana Rios"
    assert canonical_for["Ana Lios"] == "Ana Rios"


def test_fragmented_ocr_variants_resolve_to_the_dominant_speaker():
    """End-to-end: without name canonicalization, the OCR-fragmented
    variants below would each individually fail min_speaker_confidence and
    the clearly-dominant speaker would incorrectly stay Unknown."""
    turns = [SpeakerTurn(start=0.0, end=160.0, speaker_id="Speaker_00")]
    events = [
        _event("Maria Alvarez", 0.0, 27.0, confidence=0.793),
        _event("Marla Alvarez", 27.0, 42.0, confidence=0.747),
        _event("Maria Alvares", 96.5, 141.5, confidence=0.931),
        _event("Mara Alvarez", 147.0, 159.0, confidence=0.666),
    ]

    result = build_speaker_name_map(turns, events, min_speaker_confidence=0.6)

    assert result.mapping["Speaker_00"] == "Maria Alvares"
