"""Pydantic data contracts shared between every pipeline stage.

Two groups of models live here:

1. **Spec contracts** -- reproduced as specified in the project brief
   (`RawUtterance`, `SlideContext`, `AlignedMeetingSegment`,
   `FullyAlignedTimeline`, `Evidence`, `TopicItem`, `ActionItem`,
   `MeetingIntelligence`). `VisualFrameContext` started as one of these
   too, but has since been restructured past the original spec (see its
   own docstring and `SlideContent`) at the user's request, into a
   `slide_id`/`start_time`/`end_time`/`content`/`display_name`/
   `ocr_confidence`/`detection_confidence` shape. `RawUtterance` gained an
   optional `language` field, and is no longer the format persisted to
   disk -- see `TranscriptSegment` below. `MeetingIntelligence` gained
   optional `llm_provider`/`llm_model` fields, set by `FusionLLMProcessor`
   after parsing (not requested of the LLM itself).
2. **Stage-internal contracts** -- models needed to pass data between the
   finer-grained stages shown in the architecture diagram (VAD,
   diarization, OCR, speaker naming, ...) that the brief does not spell
   out field-by-field but implies. Keeping these as first-class Pydantic
   models (rather than dicts) lets every stage validate its own
   inputs/outputs independently, which is what makes running a single
   stage in isolation safe. `TranscriptSegment`/`TranscriptFile` are the
   user-requested replacement for a plain `RawUtterance[]` artifact: one
   record per contiguous same-speaker run, not one per raw diarization
   turn (see `transcript.py`).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field


def _round3(value: float) -> float:
    return round(value, 3)


# Every measured/computed float in this file (timestamps, durations,
# confidence scores) goes through this type instead of a bare `float`, so
# binary floating-point noise from sample-count division, model outputs,
# etc. (e.g. 0.7319999999999998) never reaches JSON output -- regardless
# of which stage constructs the value. Applied here once rather than at
# every call site, since call sites are easy to add and easy to forget to
# round individually.
RoundedFloat = Annotated[float, AfterValidator(_round3)]

# --------------------------------------------------------------------------
# Spec contracts (verbatim from the architecture brief)
# --------------------------------------------------------------------------


class SlideContent(BaseModel):
    """Structured breakdown of a single frame's on-screen text."""

    title: str | None = Field(default=None, description="The slide's title, inferred as its largest-font line near the top (see ocr._pick_title_line)")
    bullet_points: list[str] = Field(default_factory=list, description="Short bullet-style lines (bullet markers stripped)")
    paragraphs: list[str] = Field(default_factory=list, description="Longer flowing-prose lines, if present")
    raw_text: str = Field(default="", description="Full raw OCR text for this frame, original lines newline-joined, unprocessed")


class VisualFrameContext(BaseModel):
    """One record per sampled Zoom video frame, processed in chronological order.

    Vision-only: built entirely from OCR + on-screen layout, never audio or
    diarization. Combines the slide's structured content (`content`) with
    the presenter's display name as shown in the Zoom interface
    (`display_name`) -- read directly off a Zoom speaker overlay
    (active-speaker border / presentation badge), never inferred from
    slide content or the actual active speaker.
    """

    slide_id: str = Field(..., description='Self-describing id encoding this frame\'s start time, e.g. "slide_000185"')
    start_time: RoundedFloat
    end_time: RoundedFloat | None = Field(default=None, description="Next frame's start_time, or None if this is the last sampled frame")
    frame_path: str
    content: SlideContent
    display_name: str | None = Field(
        default=None,
        description="Presenter display name read off a Zoom speaker overlay at this frame's timestamp; null if none is visible.",
    )
    display_name_source: str | None = Field(
        default=None,
        description=(
            "Which detector produced display_name: 'layout_heuristic' (name-shaped line found in "
            "the frame's own whole-frame OCR), 'gpt4o_vision', 'active_speaker_border', or 'presentation_badge'."
        ),
    )
    ocr_confidence: RoundedFloat | None = Field(default=None, description="Average OCR detection confidence (0-1) for this frame's whole-frame text")
    detection_confidence: RoundedFloat | None = Field(
        default=None,
        description="OCR confidence (0-1) of the display_name detection, if sourced from a local heuristic (whole-frame or crop); null for gpt4o_vision (no comparable numeric score) or when no name was found.",
    )


class RawUtterance(BaseModel):
    """Intermediate speech output contract: one record per diarized speaker
    turn that has words, produced by Part A's Merge step (`merge.py`).

    Purely an internal handoff to `transcript.py`'s consecutive-same-speaker
    merge, which is what actually gets persisted (`transcript.json`) and
    consumed by Fusion -- see `TranscriptSegment`.
    """

    start: RoundedFloat
    end: RoundedFloat
    speaker_id: str = Field(..., description='e.g. "Speaker_00"')
    transcript: str
    language: str | None = Field(default=None, description="ISO 639-1 language code detected by ASR for this turn's speech, e.g. \"en\"")


class TranscriptSegment(BaseModel):
    """One merged transcript segment: consecutive `RawUtterance` turns from
    the *same* speaker (e.g. a diarization artifact splitting one
    continuous turn into several back-to-back same-speaker turns) are
    merged into a single segment here, rather than surfacing each raw
    diarization turn as its own record. See `transcript.build_transcript`.
    """

    start_s: RoundedFloat
    end_s: RoundedFloat
    speaker_id: str = Field(..., description='e.g. "Speaker_00"')
    text: str
    language: str | None = Field(default=None, description="ISO 639-1 language code detected by ASR, e.g. \"en\"")


class TranscriptFile(BaseModel):
    """On-disk artifact wrapper for transcript.json (see `io_utils.write_model`/`read_model`)."""

    segments: list[TranscriptSegment]


class SlideContext(BaseModel):
    title: str | None = Field(None, description="The inferred or explicit title of the slide")
    ocr_text: list[str] = Field(..., description="Lines of text visible on screen during this utterance")


class AlignedMeetingSegment(BaseModel):
    start: RoundedFloat
    end: RoundedFloat
    speaker: str = Field(..., description="The resolved speaker name/tag (e.g. 'Presenter', 'John Doe')")
    transcript: str = Field(..., description="The spoken text content")
    language: str | None = Field(default=None, description="ISO 639-1 language code detected by ASR, e.g. \"en\"")
    slide: SlideContext | None = Field(None, description="The visual slide text matching this timeframe")


class FullyAlignedTimeline(BaseModel):
    segments: list[AlignedMeetingSegment]


class Evidence(BaseModel):
    timestamps: list[RoundedFloat]
    speakers: list[str]
    visual_reference: str | None = Field(None, description="Snippets of OCR text backing this insight")


class TopicItem(BaseModel):
    topic: str
    summary: str = Field(..., description="Summary of this topic, written in the meeting's dominant language")
    evidence: Evidence


class ActionItem(BaseModel):
    task: str
    assignee: str | None = Field(None, description="Real name or speaker tag assigned")
    evidence_quote: str


class MeetingIntelligence(BaseModel):
    summary: str = Field(..., description="High-level meeting synthesis")
    topics: list[TopicItem]
    action_items: list[ActionItem]
    llm_provider: str | None = Field(
        default=None,
        description="Which LLM provider produced this enrichment: 'openai' or 'gemini'. Set by FusionLLMProcessor after parsing, not by the model itself.",
    )
    llm_model: str | None = Field(
        default=None,
        description="The specific model name used, e.g. 'gpt-4o' or 'gemini-1.5-flash'. Set by FusionLLMProcessor after parsing, not by the model itself.",
    )


# --------------------------------------------------------------------------
# Stage-internal contracts
# --------------------------------------------------------------------------


class IngestionResult(BaseModel):
    """Output of the Ingestion stage."""

    video_path: str
    wav_path: str
    duration_sec: RoundedFloat
    sample_rate: int = 16_000
    channels: int = 1


class VadSegment(BaseModel):
    """A single voice-active chunk produced by Segmentation (Silero VAD)."""

    segment_id: str = Field(..., description='e.g. "seg_0000"')
    start_s: RoundedFloat
    duration: RoundedFloat

    @property
    def end_s(self) -> float:
        """Convenience accessor for downstream code; not a serialized field (see `duration`)."""
        return round(self.start_s + self.duration, 3)


class VadSegmentsFile(BaseModel):
    """On-disk artifact wrapper for vad_segments.json (see `io_utils.write_model`/`read_model`)."""

    segments: list[VadSegment]


class SpeakerTurn(BaseModel):
    """A speaker turn produced by Diarization (pyannote.audio)."""

    start: RoundedFloat
    end: RoundedFloat
    speaker_id: str = Field(..., description='e.g. "Speaker_00"')


class Word(BaseModel):
    """A single word-level timestamp emitted by the ASR stage."""

    start: RoundedFloat
    end: RoundedFloat
    text: str
    probability: RoundedFloat | None = None


class AsrSegmentResult(BaseModel):
    """ASR output for one VAD chunk, prior to speaker-turn merging."""

    segment_id: str = Field(..., description='The source VAD chunk\'s segment_id, e.g. "seg_0000"')
    start: RoundedFloat
    end: RoundedFloat
    text: str
    words: list[Word] = Field(default_factory=list)
    language: str | None = Field(default=None, description="ISO 639-1 language code detected by ASR for this chunk, e.g. \"en\"")


class SceneFrame(BaseModel):
    """A representative frame produced by Scene Detection (one per detected slide transition)."""

    timestamp: RoundedFloat
    frame_path: str


class VisionTrackOutput(BaseModel):
    """Full output of Part B (Vision): just the per-frame records.

    There's no separate flattened "hints" or "name tags" list here: each
    frame already carries everything about itself (`bullets`,
    `candidate_speaker`, timestamp), so ASR looks up the one slide active
    during a given chunk (see `alignment.find_active_frame`) and Speaker
    Naming reads `candidate_speaker` straight off each frame, instead of
    either being handed a separate aggregated list.
    """

    frames: list[VisualFrameContext] = Field(default_factory=list)


class SpeakerNameMap(BaseModel):
    """Output of the Speaker Naming correlation engine."""

    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="speaker_id (e.g. 'Speaker_00') -> resolved display name, or 'Unknown' if unresolved",
    )


class VisualSpeakerEvent(BaseModel):
    """A display name read directly off Zoom's own on-screen UI chrome
    (an active-speaker border or a presentation badge), spanning the
    timeframe it was visible -- built by `visual_speaker.build_visual_speaker_events`
    from consecutive `VisualFrameContext` records that share the same
    Zoom-UI-sourced name, then later aligned with diarization speaker turns
    by temporal overlap to resolve `SpeakerNameMap`.

    Deliberately scoped to structural Zoom UI signals only, not the whole-
    frame name-shape scan or GPT-4o vision fallback in `ocr.py` (which can
    pick up a name-shaped line from arbitrary slide text) -- see
    `visual_speaker.py`'s module docstring.
    """

    start: RoundedFloat
    end: RoundedFloat
    display_name: str
    layout: str = Field(..., description='Zoom layout this signal is only ever visible in: "gallery_view" or "shared_screen"')
    signal: str = Field(..., description='Which Zoom UI element produced this event: "active_speaker_border" or "presentation_badge"')
    confidence: RoundedFloat = Field(..., ge=0, le=1, description="Fused OCR + detection + multi-frame-consistency confidence for this event")


class VisualSpeakerEventsFile(BaseModel):
    """On-disk artifact wrapper for visual_speaker_events.json."""

    events: list[VisualSpeakerEvent] = Field(default_factory=list)
