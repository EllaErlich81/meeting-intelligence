"""Part C / Speaker Naming: align `VisualSpeakerEvent`s (see `visual_speaker.py`)
with diarization speaker turns to resolve `SpeakerNameMap`.

Every event is matched against every speaker turn it overlaps in time; the
weight of that match is the event's own confidence (see `visual_speaker.py`
for how that's fused from OCR confidence, structural-signal confidence, and
multi-frame consistency) scaled by how much of the *event's* timespan the
turn actually covers -- an event that only barely overlaps a turn shouldn't
count nearly as much as one that overlaps it almost entirely. Summing that
weighted evidence per (name, speaker_id) pair and taking whichever speaker
holds the largest *share* of a name's total weight gives a single 0-1
"alignment confidence" that already captures the temporal-overlap and
cross-event-consistency factors the spec calls out: a name that's only
weakly/inconsistently linked to one speaker over another (e.g. visible near
two different speakers about equally) naturally lands near 0.5 and fails to
clear the threshold, rather than being confidently -- and possibly wrongly
-- assigned to whichever speaker it happened to edge out.

A speaker with no confidently-resolved name is labeled `"Unknown"` rather
than falling back to its raw diarization id, per the visual-speaker-
identification spec.

Before alignment, event display_names are canonicalized by string
similarity (`_canonicalize_names`): PaddleOCR reads the same badge/border
name-crop slightly differently across frames (e.g. "Maria Alvarez" /
"Marla Alvarez" / "Mara Alvarez" -- a substituted or dropped letter),
and without this step that OCR noise would fragment one person's evidence
across several near-identical exact strings, none of which individually
accumulates enough weight to resolve confidently. Names within
`name_similarity_threshold` (stdlib `difflib.SequenceMatcher` ratio,
case-insensitive) of each other are folded into one cluster, keyed to
whichever variant has the highest total confidence across all its
occurrences (a proxy for "most/best-corroborated reading", not necessarily
the most frequent one). The default threshold (0.65) leaves a comfortable
margin between single-letter OCR noise on the same name (which measures
well above it) and most genuinely different short names (which measure
well below it); recalibrate `SPEAKER_NAMING_NAME_SIMILARITY_THRESHOLD` if
your footage's OCR error rate runs unusually high or low. This is pure
string-shape similarity with no other identity signal, so it isn't
foolproof either way: two different participants who happen to share a
surname ("John Smith" / "Jane Smith") measure 0.8 similarity -- above the
default -- and would be merged. See README's "Known limitations".
"""

from __future__ import annotations

import logging
from collections import defaultdict
from difflib import SequenceMatcher

from .models import SpeakerNameMap, SpeakerTurn, VisualSpeakerEvent

logger = logging.getLogger(__name__)

DEFAULT_MIN_SPEAKER_CONFIDENCE = 0.6
DEFAULT_NAME_SIMILARITY_THRESHOLD = 0.65
UNKNOWN_SPEAKER_LABEL = "Unknown"


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _canonicalize_names(events: list[VisualSpeakerEvent], similarity_threshold: float) -> dict[str, str]:
    """Cluster event display_names by string similarity; map each raw name to its cluster's canonical form."""
    total_confidence: dict[str, float] = defaultdict(float)
    for event in events:
        total_confidence[event.display_name] += event.confidence

    # Process the most-corroborated names first, so a clean, well-attested
    # reading becomes the canonical form its noisier variants fold into,
    # not the other way around.
    names_by_confidence_desc = sorted(total_confidence, key=lambda n: total_confidence[n], reverse=True)

    canonical_for: dict[str, str] = {}
    canonical_names: list[str] = []
    for name in names_by_confidence_desc:
        # Best match, not first match: a name within threshold of more than
        # one existing canonical must join the closest one, not whichever
        # happened to be accepted earliest (unrelated to string similarity).
        ratios = ((c, SequenceMatcher(None, name.casefold(), c.casefold()).ratio()) for c in canonical_names)
        best = max((cr for cr in ratios if cr[1] >= similarity_threshold), key=lambda cr: cr[1], default=None)
        canonical_for[name] = best[0] if best is not None else name
        if best is None:
            canonical_names.append(name)
    return canonical_for


def build_speaker_name_map(
    speaker_turns: list[SpeakerTurn],
    visual_speaker_events: list[VisualSpeakerEvent],
    min_speaker_confidence: float = DEFAULT_MIN_SPEAKER_CONFIDENCE,
    name_similarity_threshold: float = DEFAULT_NAME_SIMILARITY_THRESHOLD,
) -> SpeakerNameMap:
    """Resolve each speaker_id to the display name its overlapping visual events most confidently agree on."""
    canonical_for = _canonicalize_names(visual_speaker_events, name_similarity_threshold)
    name_speaker_weight: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for event in visual_speaker_events:
        event_duration = max(event.end - event.start, 1e-6)
        canonical_name = canonical_for[event.display_name]
        for turn in speaker_turns:
            overlap = _overlap_seconds(event.start, event.end, turn.start, turn.end)
            if overlap <= 0:
                continue
            overlap_fraction = overlap / event_duration
            name_speaker_weight[canonical_name][turn.speaker_id] += event.confidence * overlap_fraction

    resolutions: list[tuple[str, str, float, float]] = []
    for name, weights in name_speaker_weight.items():
        total = sum(weights.values())
        if total <= 0:
            continue
        best_speaker, best_weight = max(weights.items(), key=lambda kv: kv[1])
        confidence = best_weight / total
        if confidence >= min_speaker_confidence:
            resolutions.append((name, best_speaker, confidence, best_weight))

    # If two different names both resolve to the same speaker, keep the one
    # backed by more total evidence (best_weight), not just the higher
    # confidence *ratio* -- a single isolated, uncontested event is
    # trivially 100% "confident" (no competing evidence to dilute it) but
    # represents far less real evidence than a well-corroborated cluster
    # that happens to have a little competing noise dragging its ratio down.
    speaker_names: dict[str, str] = {}
    for name, speaker_id, _confidence, _best_weight in sorted(resolutions, key=lambda r: r[3], reverse=True):
        if speaker_id not in speaker_names:
            speaker_names[speaker_id] = name

    all_speaker_ids = {turn.speaker_id for turn in speaker_turns}
    mapping = {speaker_id: speaker_names.get(speaker_id, UNKNOWN_SPEAKER_LABEL) for speaker_id in all_speaker_ids}

    resolved_count = sum(1 for name in mapping.values() if name != UNKNOWN_SPEAKER_LABEL)
    logger.info("Speaker naming resolved %d/%d speaker(s) to a real name", resolved_count, len(mapping))
    return SpeakerNameMap(mapping=mapping)
