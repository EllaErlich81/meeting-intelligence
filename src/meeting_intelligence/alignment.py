"""Shared time-alignment helper: which vision frame was on screen at a
given moment.

Used by both Fusion (attach the active slide to each utterance) and ASR
(bias each chunk's transcription with only *that* chunk's slide text,
instead of every slide seen anywhere in the recording).
"""

from __future__ import annotations

import bisect

from .models import VisualFrameContext


def sort_frames_by_timestamp(
    frames: list[VisualFrameContext],
) -> tuple[list[VisualFrameContext], list[float]]:
    sorted_frames = sorted(frames, key=lambda f: f.start_time)
    return sorted_frames, [f.start_time for f in sorted_frames]


def find_active_frame(
    at_time: float,
    sorted_frames: list[VisualFrameContext],
    sorted_timestamps: list[float],
) -> VisualFrameContext | None:
    """Return the most recent frame at or before `at_time`.

    Frames mark the *start* of a slide's visibility, which persists until
    the next detected transition, so the most recent transition at or
    before `at_time` is the slide that was on screen at that moment.
    """
    idx = bisect.bisect_right(sorted_timestamps, at_time) - 1
    if idx < 0:
        return None
    return sorted_frames[idx]
