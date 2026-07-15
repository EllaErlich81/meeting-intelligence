"""Part C / Timeline Fusion: interval-overlap alignment of speech and vision.

For every transcript segment, finds the slide/frame that was active
on-screen during that segment and nests it inside the segment block,
producing the `FullyAlignedTimeline` fed to the LLM enrichment stage.
"""

from __future__ import annotations

import logging

from .alignment import find_active_frame, sort_frames_by_timestamp
from .models import (
    AlignedMeetingSegment,
    FullyAlignedTimeline,
    SlideContext,
    SpeakerNameMap,
    TranscriptSegment,
    VisualFrameContext,
)

logger = logging.getLogger(__name__)


def fuse_timeline(
    transcript_segments: list[TranscriptSegment],
    vision_frames: list[VisualFrameContext],
    speaker_name_map: SpeakerNameMap | None = None,
) -> FullyAlignedTimeline:
    """Fuse Part A's transcript with Part B frames into one aligned timeline.

    For a transcript segment spanning a slide change, the slide active at
    the segment's *end* is used -- see `find_active_frame`.
    """
    sorted_frames, sorted_timestamps = sort_frames_by_timestamp(vision_frames)
    mapping = speaker_name_map.mapping if speaker_name_map else {}

    segments = []
    for seg in sorted(transcript_segments, key=lambda s: s.start_s):
        frame = find_active_frame(seg.end_s, sorted_frames, sorted_timestamps)
        slide = SlideContext(title=frame.content.title, ocr_text=frame.content.raw_text.splitlines()) if frame else None
        segments.append(
            AlignedMeetingSegment(
                start=seg.start_s,
                end=seg.end_s,
                speaker=mapping.get(seg.speaker_id, seg.speaker_id),
                transcript=seg.text,
                language=seg.language,
                slide=slide,
            )
        )

    logger.info("Fused %d transcript segment(s) with %d visual frame(s)", len(segments), len(sorted_frames))
    return FullyAlignedTimeline(segments=segments)
