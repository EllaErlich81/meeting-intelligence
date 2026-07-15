"""Part A / Merge: align ASR words back into diarized speaker-turn timeframes.

Uses word-level timestamp overlap: every ASR word (or, when word-level
timestamps aren't available, whole ASR segment) is assigned to whichever
speaker turn it overlaps the most, and turns are stitched back into one
`RawUtterance` per speaker turn -- an intermediate handoff, not itself
persisted; see `transcript.py`, which merges consecutive same-speaker
`RawUtterance` turns into the `TranscriptSegment` records that are
actually written to `transcript.json` and consumed by Fusion.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from .models import AsrSegmentResult, RawUtterance, SpeakerTurn

logger = logging.getLogger(__name__)


def _best_matching_turn(start: float, end: float, sorted_turns: list[SpeakerTurn]) -> int | None:
    """Return the index into `sorted_turns` with the greatest time overlap with [start, end]."""
    best_idx: int | None = None
    best_overlap = 0.0
    for idx, turn in enumerate(sorted_turns):
        if turn.end < start:
            continue
        if turn.start > end:
            break
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_idx is not None:
        return best_idx

    # Zero measured overlap (e.g. rounding at a turn boundary): fall back to
    # whichever turn contains the unit's midpoint.
    midpoint = (start + end) / 2
    for idx, turn in enumerate(sorted_turns):
        if turn.start <= midpoint <= turn.end:
            return idx
    return None


def merge_words_into_turns(
    asr_segments: list[AsrSegmentResult],
    speaker_turns: list[SpeakerTurn],
) -> list[RawUtterance]:
    """Merge ASR output and diarization turns into one `RawUtterance` per speaker turn."""
    if not speaker_turns:
        logger.warning("No speaker turns available; cannot merge ASR output")
        return []

    sorted_turns = sorted(speaker_turns, key=lambda t: t.start)

    # Prefer word-level units; fall back to whole-segment text when a
    # segment carries no word timestamps (e.g. the OpenAI Whisper API path).
    # Each unit carries its source ASR segment's detected language along
    # with it, since language is a per-chunk property, not per-word.
    units: list[tuple[float, float, str, str | None]] = []
    for seg in asr_segments:
        if seg.words:
            units.extend((w.start, w.end, w.text, seg.language) for w in seg.words)
        elif seg.text:
            units.append((seg.start, seg.end, seg.text, seg.language))
    units.sort(key=lambda u: u[0])

    words_by_turn: dict[int, list[str]] = defaultdict(list)
    languages_by_turn: dict[int, list[str]] = defaultdict(list)
    unmatched = 0
    for start, end, text, language in units:
        turn_idx = _best_matching_turn(start, end, sorted_turns)
        if turn_idx is None:
            unmatched += 1
            continue
        words_by_turn[turn_idx].append(text)
        if language:
            languages_by_turn[turn_idx].append(language)

    if unmatched:
        logger.warning("%d ASR unit(s) did not overlap any speaker turn and were dropped", unmatched)

    utterances = [
        RawUtterance(
            start=turn.start,
            end=turn.end,
            speaker_id=turn.speaker_id,
            transcript=" ".join(words_by_turn[idx]).strip(),
            # A turn can span words from more than one ASR chunk (each with
            # its own detected language); the majority vote is more robust
            # to one chunk's language misdetection than just taking the first.
            language=Counter(languages_by_turn[idx]).most_common(1)[0][0] if languages_by_turn.get(idx) else None,
        )
        for idx, turn in enumerate(sorted_turns)
        if words_by_turn.get(idx)
    ]

    logger.info("Merged into %d utterance(s) from %d speaker turn(s)", len(utterances), len(sorted_turns))
    return utterances
