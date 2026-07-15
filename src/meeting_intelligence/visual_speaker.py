"""Part C / Visual Speaker Identification: turn Zoom-UI structural signals
into timestamped `VisualSpeakerEvent`s.

`ocr.py`'s `detect_display_name` already tries the Zoom-specific
active-speaker-border and presentation-badge crops (`zoom_layout.py`)
*before* its generic whole-frame name-shape scan or GPT-4o vision fallback
(see that module's docstring) -- so a `VisualFrameContext.display_name_source`
of `"active_speaker_border"` or `"presentation_badge"` already means a
genuine Zoom UI element was matched, not a guess from arbitrary slide text.
This module only ever turns *those* frames into events, re-using OCR's
already-computed detection rather than re-running any pixel analysis:

1. **Detect layout**: which signal fired *is* the layout. Zoom only ever
   draws the colored active-speaker border in gallery/speaker-grid view,
   and only ever shows the "who's talking" badge during screen-share /
   presentation mode -- the two are mutually exclusive by construction, so
   "which detector matched" is a correct, zero-cost layout classification,
   not a heuristic guess.
2. **Merge consecutive frames**: consecutive frames with the same
   (signal, display_name) are merged into a single event spanning their
   combined timeframe -- multiple corroborating frames are evidence the
   detection wasn't a one-off misread, and factor into that event's
   confidence.
3. **Score confidence**: fuses (a) the average OCR confidence of the
   name-label crop across the event's frames, (b) a fixed per-signal
   structural-match confidence -- the active-speaker border requires an
   actual HSV color-mask + area-fraction match against calibrated
   thresholds, a real (if simple) structural test, while the presentation
   badge is a fixed-position crop with no content-based validation at all,
   so it's inherently less certain purely as a *detection* -- and (c) a
   saturating bonus for how many frames corroborate the event. Temporal
   overlap with diarization, the fourth factor called out by the spec, is
   necessarily a property of a *speaker-turn alignment*, not of an event in
   isolation, and is applied afterward in `speaker_naming.build_speaker_name_map`.
"""

from __future__ import annotations

import logging

from .models import VisualFrameContext, VisualSpeakerEvent

logger = logging.getLogger(__name__)

# Only these two `display_name_source` values come from a genuine Zoom UI
# element (see module docstring); everything else -- "layout_heuristic",
# "gpt4o_vision" -- is a guess from arbitrary on-screen text and is never
# turned into a VisualSpeakerEvent.
_LAYOUT_BY_SIGNAL: dict[str, str] = {
    "active_speaker_border": "gallery_view",
    "presentation_badge": "shared_screen",
}

# The border requires an actual calibrated color/geometry match; the badge
# is a blind fixed-region crop with no content-based confirmation that a
# badge is even visible there. See module docstring.
_SIGNAL_CONFIDENCE: dict[str, float] = {
    "active_speaker_border": 0.85,
    "presentation_badge": 0.65,
}

# How many corroborating frames it takes for the consistency bonus to
# saturate -- a handful of frames is enough evidence the name wasn't a
# one-off misread; more than that shouldn't keep pushing confidence up.
CONSISTENCY_SATURATION_FRAME_COUNT = 3

OCR_CONFIDENCE_WEIGHT = 0.5
SIGNAL_CONFIDENCE_WEIGHT = 0.3
CONSISTENCY_WEIGHT = 0.2

# `VisualFrameContext.end_time` is null only for the last sampled frame in
# the whole recording (true duration unknown -- see its docstring), which
# would otherwise collapse a single-frame trailing event to start==end. A
# zero-duration event contributes exactly zero alignment weight everywhere
# in speaker_naming.build_speaker_name_map, silently discarding otherwise
# unambiguous evidence for whoever's on screen as the recording ends.
_TRAILING_FRAME_DEFAULT_DURATION_SEC = 1.0


class _OpenEvent:
    def __init__(self, frame: VisualFrameContext) -> None:
        self.start = frame.start_time
        self.end = frame.end_time if frame.end_time is not None else frame.start_time + _TRAILING_FRAME_DEFAULT_DURATION_SEC
        self.display_name = frame.display_name
        self.signal = frame.display_name_source
        self.ocr_confidences = [frame.detection_confidence or 0.0]

    def extend(self, frame: VisualFrameContext) -> None:
        self.end = frame.end_time if frame.end_time is not None else frame.start_time + _TRAILING_FRAME_DEFAULT_DURATION_SEC
        self.ocr_confidences.append(frame.detection_confidence or 0.0)

    def matches(self, frame: VisualFrameContext) -> bool:
        return frame.display_name_source == self.signal and frame.display_name == self.display_name and frame.start_time <= self.end + 1e-6

    def finalize(self) -> VisualSpeakerEvent:
        ocr_confidence = sum(self.ocr_confidences) / len(self.ocr_confidences)
        consistency = min(1.0, len(self.ocr_confidences) / CONSISTENCY_SATURATION_FRAME_COUNT)
        confidence = (
            OCR_CONFIDENCE_WEIGHT * ocr_confidence + SIGNAL_CONFIDENCE_WEIGHT * _SIGNAL_CONFIDENCE[self.signal] + CONSISTENCY_WEIGHT * consistency
        )
        return VisualSpeakerEvent(
            start=self.start,
            end=self.end,
            display_name=self.display_name,
            layout=_LAYOUT_BY_SIGNAL[self.signal],
            signal=self.signal,
            confidence=confidence,
        )


def build_visual_speaker_events(frames: list[VisualFrameContext]) -> list[VisualSpeakerEvent]:
    """Turn Zoom-UI-sourced frames into merged, confidence-scored `VisualSpeakerEvent`s.

    Frames whose `display_name` came from the generic layout heuristic or
    GPT-4o vision (not a genuine Zoom border/badge match) never become an
    event -- see module docstring.
    """
    events: list[VisualSpeakerEvent] = []
    open_event: _OpenEvent | None = None

    for frame in sorted(frames, key=lambda f: f.start_time):
        if frame.display_name_source not in _LAYOUT_BY_SIGNAL or not frame.display_name:
            if open_event is not None:
                events.append(open_event.finalize())
                open_event = None
            continue

        if open_event is not None and open_event.matches(frame):
            open_event.extend(frame)
            continue

        if open_event is not None:
            events.append(open_event.finalize())
        open_event = _OpenEvent(frame)

    if open_event is not None:
        events.append(open_event.finalize())

    logger.info("Built %d visual speaker event(s) from %d frame(s)", len(events), len(frames))
    return events
