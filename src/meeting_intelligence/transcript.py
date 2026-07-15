"""Part A / Transcript: merge consecutive same-speaker utterances.

Diarization occasionally splits one continuous turn from a single speaker
into several back-to-back turns (a brief pause misread as a turn boundary,
a diarization-model artifact, etc.). Surfacing each of those raw turns as
its own record produces a choppier transcript than the recording actually
warrants -- this merges any run of consecutive `RawUtterance`s that share
the same `speaker_id` into a single `TranscriptSegment`, which is what's
actually persisted to `transcript.json` and consumed by Fusion.

A speaker who holds the floor uninterrupted for many diarization turns in a
row (a long presentation, a councillor reading a lengthy report) would
otherwise merge into one enormous segment -- observed in practice at
nearly 5000 words over an ~18-minute run. `max_segment_duration_sec` caps
that: a run is closed out and a new one started once adding the next
utterance would push the segment's span (from the run's first start to the
new utterance's end) past the limit, even though the speaker hasn't
changed. This only ever splits *at* existing utterance (diarization turn)
boundaries, never mid-utterance, so every split point is already a real
turn boundary, not an arbitrary cut. A single utterance that alone exceeds
the cap still becomes one (long) segment -- splitting mid-utterance would
need to re-slice word timestamps, which is out of scope here.
"""

from __future__ import annotations

import logging
from collections import Counter

from .models import RawUtterance, TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEGMENT_DURATION_SEC = 120.0


def build_transcript(
    utterances: list[RawUtterance], max_segment_duration_sec: float = DEFAULT_MAX_SEGMENT_DURATION_SEC
) -> list[TranscriptSegment]:
    """Merge consecutive same-speaker `RawUtterance`s into `TranscriptSegment`s.

    `utterances` is assumed already in chronological order (as produced by
    `merge.merge_words_into_turns`); merging only ever considers adjacent
    entries, so an out-of-order list would merge the wrong turns together.
    """
    if not utterances:
        return []

    def _segment_from_run(run: list[RawUtterance]) -> TranscriptSegment:
        languages = [u.language for u in run if u.language]
        return TranscriptSegment(
            start_s=run[0].start,
            end_s=run[-1].end,
            speaker_id=run[0].speaker_id,
            text=" ".join(u.transcript for u in run if u.transcript).strip(),
            language=Counter(languages).most_common(1)[0][0] if languages else None,
        )

    segments: list[TranscriptSegment] = []
    current_run = [utterances[0]]
    for utt in utterances[1:]:
        same_speaker = utt.speaker_id == current_run[-1].speaker_id
        fits = (utt.end - current_run[0].start) <= max_segment_duration_sec
        if same_speaker and fits:
            current_run.append(utt)
            continue
        segments.append(_segment_from_run(current_run))
        current_run = [utt]
    segments.append(_segment_from_run(current_run))

    logger.info("Merged %d utterance(s) into %d transcript segment(s)", len(utterances), len(segments))
    return segments
