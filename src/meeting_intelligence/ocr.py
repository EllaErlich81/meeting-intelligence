"""Part B / OCR & Name Scan: extract on-screen text with PaddleOCR, falling
back to GPT-4o vision when the layout is too complex for PaddleOCR to read
confidently.

Processes sampled Zoom video frames, in chronological order, into
structured slide content plus the presenter's display name -- vision only,
never audio or diarization. Each frame gets exactly one
`VisualFrameContext` record: `slide_id`, `start_time`/`end_time`,
structured `content` (title / bullet points / paragraphs / raw text),
`display_name` (+ `display_name_source`), and separate
`ocr_confidence` / `detection_confidence` scores.

`display_name` detection tries, in order:

1. The Zoom-specific border/badge pixel-region crops in `zoom_layout.py`
   -- anchored to Zoom's own UI chrome (a colored active-speaker border, or
   a "who's talking" badge), so more trustworthy than pattern-matching
   arbitrary slide text whenever that UI is actually present. Requires
   per-deployment color/position calibration; disabled detectors, or ones
   that find nothing in this frame, fall through to the next step.
2. A name-shaped line found in this frame's own whole-frame OCR pass
   (`detect_display_name_via_layout`) -- free, no extra OCR/API call, and
   robust across Zoom themes/resolutions since there's no calibration to
   get right, but can false-positive on any Title-Case two-word slide text
   (an org name, a place name) that happens to look name-shaped.
3. GPT-4o vision on the whole frame (`detect_display_name_via_gpt4o`), if
   configured -- a last resort for frames whose name tag isn't readable as
   a clean OCR line (occluded, stylized, very small).
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Callable

import cv2

from .models import SceneFrame, SlideContent, VisionTrackOutput, VisualFrameContext
from .zoom_layout import (
    ZoomLayoutSettings,
    crop_badge_name_label,
    crop_name_label,
    find_active_speaker_tile,
    presentation_badge_bbox,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# A line is classified as a bullet point if it starts with a common bullet
# marker or numbering scheme (the marker itself is stripped); otherwise a
# longer line (more than PARAGRAPH_WORD_THRESHOLD words) reads as flowing
# prose and is classified as a paragraph rather than a bullet fragment.
BULLET_PREFIX_RE = re.compile(r"^(?:[•‣◦⁃∙*\-]|\d+[.)]|[a-zA-Z][.)])\s+")
PARAGRAPH_WORD_THRESHOLD = 8

# A line reads as a person's name if it's a short (<=4 word), Title-Case
# "Firstname Lastname"-shaped phrase, with an optional honorific and
# optional middle initial/second surname. Deliberately stricter than "has
# a letter in it" (see `_looks_like_a_name`, still used for crop-scoped
# extraction below) since this is applied to every OCR line in the whole
# frame, including ordinary slide titles/bullets that must NOT match.
_NAME_LINE_RE = re.compile(r"^(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$")
_NAME_LINE_MAX_WORDS = 4


def is_name_shaped(text: str) -> bool:
    """Whether `text` reads as a short "Firstname Lastname"-style name, e.g. a Zoom display-name tag."""
    text = text.strip()
    if not text or len(text.split()) > _NAME_LINE_MAX_WORDS:
        return False
    return bool(_NAME_LINE_RE.match(text))


def _pick_title_line(lines: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the title as the first (i.e. topmost, since `lines` is reading-order)
    line whose font size is at least 70% of the tallest line on the frame --
    falling back to the first line outright if no line stands out (e.g. every
    line shares the same synthesized height because no real position data was
    available). Matches how slides are conventionally laid out: a visually
    prominent heading, not necessarily whatever text the OCR engine read first.
    """
    if not lines:
        return None
    max_height = max(line["height"] for line in lines)
    return next((line for line in lines if line["height"] >= 0.7 * max_height), lines[0])


def classify_content(lines: list[dict[str, Any]], excluded_lines: list[dict[str, Any]] = ()) -> SlideContent:
    """Split a frame's OCR lines into title / bullet points / paragraphs.

    Every line besides the selected title (see `_pick_title_line`) and any
    `excluded_lines` (the frame's detected `display_name` line, if any --
    see `process_frame`) is classified as a bullet point (marker stripped)
    or, if it reads as a longer flowing sentence, a paragraph. `raw_text`
    always preserves every original line verbatim, in reading order,
    regardless of exclusion.
    """
    if not lines:
        return SlideContent()

    title_line = _pick_title_line(lines)
    skip_ids = {id(title_line)} | {id(line) for line in excluded_lines}
    bullet_points: list[str] = []
    paragraphs: list[str] = []

    for line in lines:
        if id(line) in skip_ids:
            continue
        text = line["text"]
        match = BULLET_PREFIX_RE.match(text)
        if match:
            bullet_points.append(text[match.end() :].strip())
        elif len(text.split()) > PARAGRAPH_WORD_THRESHOLD:
            paragraphs.append(text)
        else:
            bullet_points.append(text)

    return SlideContent(
        title=title_line["text"],
        bullet_points=bullet_points,
        paragraphs=paragraphs,
        raw_text="\n".join(line["text"] for line in lines),
    )


def load_paddle_ocr():
    """Load PaddleOCR. Isolated so tests can inject a fake engine.

    PaddleOCR's constructor and result shape changed between the 2.x
    (`use_angle_cls=`, `.ocr(path, cls=True)` -> nested `[box, (text, conf)]`
    lists) and 3.x (`.predict(path)` -> list of `OCRResult` dict-likes with
    parallel `rec_texts`/`rec_scores` lists) API generations. This targets
    the 3.x API, which is what `pip install paddleocr` resolves to today.

    `engine="onnxruntime"` runs inference on the onnxruntime backend instead
    of PaddleOCR's default `paddle_static` engine, which needs the separate
    `paddlepaddle` package -- notorious for lagging PyPI wheel coverage on
    newer Python/OS combinations. onnxruntime has much broader, more
    current wheel coverage and PaddleOCR 3.x supports it directly.
    `use_doc_orientation_classify`/`use_doc_unwarping`/`use_textline_orientation`
    are disabled because they correct for a photographed paper document
    (skew, page curl, upside-down orientation) -- irrelevant, and a source
    of avoidable misreads, on flat screen-capture frames.
    """
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="en",
        engine="onnxruntime",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _bbox_from_quad(quad: Any) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def run_paddle_ocr(engine: Any, frame_path: str | Path) -> list[dict[str, Any]]:
    """Return one entry per text line PaddleOCR detects, in reading order:
    `{"text", "confidence", "left", "top", "width", "height"}`.

    Prefers PaddleOCR's axis-aligned `rec_boxes`; falls back to computing a
    bounding box from `rec_polys` (quadrilateral) when boxes aren't
    present. If neither is available (e.g. a test double supplying only
    `rec_texts`/`rec_scores`), each line gets a synthesized uniform-height
    box in reading order instead -- callers that use position data (title
    selection, the name-shape scan) then degrade gracefully to treating
    the first line as the title and finding no separate name candidate,
    rather than raising.
    """
    results = engine.predict(str(frame_path))
    lines: list[dict[str, Any]] = []
    for page in results or []:
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        boxes = page.get("rec_boxes")
        if boxes is None or len(boxes) == 0:
            boxes = [_bbox_from_quad(quad) for quad in (page.get("rec_polys") or [])]

        for index, (text, confidence) in enumerate(zip(texts, scores)):
            if not text or not text.strip():
                continue
            if index < len(boxes):
                left, top, right, bottom = (float(v) for v in boxes[index])
                width, height = right - left, bottom - top
            else:
                left, top, width, height = 0.0, float(index), 0.0, 1.0
            lines.append(
                {"text": text.strip(), "confidence": float(confidence), "left": left, "top": top, "width": width, "height": height}
            )
    return lines


def _wrap_plain_text_lines(text_lines: list[str]) -> list[dict[str, Any]]:
    """Wrap plain text lines (no position data, e.g. from the GPT-4o OCR
    fallback) into the same shape `run_paddle_ocr` returns, with a uniform
    synthesized height/top so title selection degrades to "first line"."""
    return [
        {"text": text.strip(), "confidence": 1.0, "left": 0.0, "top": float(index), "width": 0.0, "height": 1.0}
        for index, text in enumerate(text_lines)
        if text and text.strip()
    ]


def run_gpt4o_ocr_fallback(
    frame_path: str | Path,
    api_key: str | None,
    model: str = "gpt-4o",
    client_factory: Callable[[str | None], Any] | None = None,
) -> list[str]:
    """Ask GPT-4o vision to transcribe on-screen text when PaddleOCR is unreliable."""
    if client_factory is None:
        def client_factory(key: str | None):
            from openai import OpenAI

            return OpenAI(api_key=key)

    client = client_factory(api_key)
    image_b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe every distinct line of visible text in this meeting "
                            "screenshot (slide titles, bullet text, name tags). Return only the "
                            "text lines, one per line, nothing else."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        max_tokens=500,
    )
    content = response.choices[0].message.content or ""
    return [line.strip() for line in content.splitlines() if line.strip()]


# PaddleOCR's text detector reliably finds nothing below roughly this crop
# height (measured against real footage: an 11px-tall name-label crop from
# a dense gallery-view grid returned zero text regions; upscaled 3-4x to
# ~35-45px, the same crop read correctly). A name-label crop is a thin
# tile-height fraction of a small source frame, so this triggers often on
# dense grids, not just as an edge case.
_MIN_CROP_HEIGHT_FOR_OCR = 40


def _upscale_for_ocr(crop: Any) -> Any:
    """Upscale a crop shorter than `_MIN_CROP_HEIGHT_FOR_OCR` so PaddleOCR's
    text detector has enough pixels to find and read it; leaves
    already-tall-enough crops untouched."""
    height, width = crop.shape[:2]
    if height == 0 or height >= _MIN_CROP_HEIGHT_FOR_OCR:
        return crop
    scale = _MIN_CROP_HEIGHT_FOR_OCR / height
    return cv2.resize(crop, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC)


def _ocr_crop_for_name(
    crop: Any,
    source_frame_path: str | Path,
    crop_suffix: str,
    paddle_engine: Any,
    use_gpt4o_fallback: bool,
    openai_api_key: str | None,
    gpt4o_model: str,
    confidence_threshold: float,
) -> tuple[str | None, float | None]:
    """OCR a small, tightly-cropped region; return (first name-like line, OCR confidence).

    No name-*shape* regex is applied here (a single-word display name is
    trusted directly, unlike the whole-frame layout heuristic) -- but a
    line with no letters at all (a participant-count badge, mute-icon
    overlay, timer, or other stray digits the crop happened to catch) is
    never a person's display name, so such lines are skipped. The
    confidence returned is always PaddleOCR's measured confidence for the
    crop, even when its text is ultimately replaced by the GPT-4o fallback
    (which has no comparable numeric score).
    """
    if crop.size == 0:
        return None, None

    crop = _upscale_for_ocr(crop)
    crop_path = Path(source_frame_path).with_name(f"{Path(source_frame_path).stem}_{crop_suffix}.png")
    cv2.imwrite(str(crop_path), crop)

    lines = run_paddle_ocr(paddle_engine, crop_path)
    avg_confidence = sum(line["confidence"] for line in lines) / len(lines) if lines else None

    text_lines = [line["text"] for line in lines]
    if (not lines or (avg_confidence or 0.0) < confidence_threshold) and use_gpt4o_fallback and openai_api_key:
        try:
            text_lines = run_gpt4o_ocr_fallback(crop_path, openai_api_key, model=gpt4o_model)
        except Exception:
            logger.warning("GPT-4o OCR fallback failed on %s crop; using low-confidence PaddleOCR result instead", crop_suffix, exc_info=True)

    text_lines = [t.strip() for t in text_lines if t and t.strip()]
    name = next((t for t in text_lines if _looks_like_a_name(t)), None)
    return name, (avg_confidence if name else None)


def _looks_like_a_name(text: str) -> bool:
    """A person's display name always has at least one letter.

    Rejects lines that are purely digits/punctuation/symbols -- a
    participant-count badge ("4"), a timer ("00:12"), or similar UI chrome
    the crop happened to catch, none of which are ever a real display name.
    """
    return any(ch.isalpha() for ch in text)


def detect_display_name_via_layout(lines: list[dict[str, Any]], title_line: dict[str, Any] | None) -> dict[str, Any] | None:
    """Scan this frame's own whole-frame OCR lines for a name-shaped line.

    Excludes the slide's own title (so a name-shaped title, however rare,
    is never mistaken for a speaker overlay) and returns the first
    remaining line matching `is_name_shaped` -- e.g. Zoom's own
    display-name tag rendered under a participant's video tile. No
    per-deployment color/position calibration needed: this reuses the
    whole-frame OCR pass already done for slide content.
    """
    for line in lines:
        if line is title_line:
            continue
        if is_name_shaped(line["text"]):
            return line
    return None


DISPLAY_NAME_PROMPT = (
    "This is a screenshot from a Zoom meeting. Identify the display name of "
    "the current speaker by looking for either: (a) a colored highlight "
    "border around one participant's video tile in gallery view, or (b) a "
    "small floating \"who's talking\" badge with a name, usually in a corner "
    "of the screen, shown during screen-share/presentation mode. "
    "Respond with ONLY that person's name exactly as displayed on screen, "
    "and nothing else -- no punctuation, no explanation. "
    "If no such speaker indicator is visible anywhere in the image, "
    "respond with exactly: NONE"
)


def detect_display_name_via_gpt4o(
    frame_path: str | Path,
    api_key: str,
    model: str = "gpt-4o",
    client_factory: Callable[[str | None], Any] | None = None,
) -> str | None:
    """Ask GPT-4o vision to find and read the speaker's display name directly off the whole frame.

    A fallback for frames where the whole-frame name-shape scan
    (`detect_display_name_via_layout`) finds nothing -- e.g. the name tag
    is stylized, occluded, or too small to OCR as a clean text line -- at
    the cost of one vision API call per such frame. Returns None if GPT-4o
    reports no visible speaker indicator, or if its answer doesn't look
    like a name.
    """
    if client_factory is None:
        def client_factory(key: str | None):
            from openai import OpenAI

            return OpenAI(api_key=key)

    client = client_factory(api_key)
    image_b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DISPLAY_NAME_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        max_tokens=30,
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer or answer.upper() == "NONE" or not _looks_like_a_name(answer):
        return None
    return answer


_NOT_COMPUTED = object()  # sentinel: distinguishes "layout_match not supplied" from a supplied value of None (no match)


def detect_display_name(
    frame_path: str | Path,
    whole_frame_lines: list[dict[str, Any]],
    title_line: dict[str, Any] | None,
    paddle_engine: Any,
    zoom_settings: ZoomLayoutSettings,
    use_gpt4o_fallback: bool = True,
    openai_api_key: str | None = None,
    gpt4o_model: str = "gpt-4o",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    layout_match: dict[str, Any] | None = _NOT_COMPUTED,
) -> tuple[str | None, float | None, str | None]:
    """Find the presenter's display name as shown in the Zoom interface.

    Tries, in order: (1) the Zoom-specific border/badge pixel-region crops
    in `zoom_layout.py` -- anchored to Zoom's own UI chrome, so more
    trustworthy than pattern-matching arbitrary slide text whenever that UI
    is actually present and calibrated; (2) the whole-frame name-shape scan
    (free, no extra OCR/API call -- see module docstring), for recordings
    where the Zoom border/badge isn't visible or enabled; (3) GPT-4o vision
    on the whole frame if configured, as a last resort. This identifies the
    display name visible in the Zoom interface, not the actual active
    speaker.

    `layout_match` lets a caller that already ran `detect_display_name_via_layout`
    (e.g. `process_frame`, which needs it regardless for `excluded_lines`)
    pass the result straight through instead of this function re-running
    the same whole-frame scan; left unset, it's computed here as before.

    Returns (name, detection_confidence, source); all None if nothing found.
    """
    image = cv2.imread(str(frame_path))
    if image is not None:
        if zoom_settings.enable_active_speaker_border:
            tile_bbox = find_active_speaker_tile(image, zoom_settings)
            if tile_bbox is not None:
                crop = crop_name_label(image, tile_bbox, zoom_settings)
                name, confidence = _ocr_crop_for_name(
                    crop, frame_path, "active_speaker", paddle_engine, use_gpt4o_fallback, openai_api_key, gpt4o_model, confidence_threshold
                )
                if name:
                    return name, confidence, "active_speaker_border"

        if zoom_settings.enable_presentation_badge:
            badge_bbox = presentation_badge_bbox(image.shape, zoom_settings)
            crop = crop_badge_name_label(image, badge_bbox, zoom_settings)
            name, confidence = _ocr_crop_for_name(
                crop, frame_path, "presentation_badge", paddle_engine, use_gpt4o_fallback, openai_api_key, gpt4o_model, confidence_threshold
            )
            if name:
                return name, confidence, "presentation_badge"

    if layout_match is _NOT_COMPUTED:
        layout_match = detect_display_name_via_layout(whole_frame_lines, title_line)
    if layout_match:
        return layout_match["text"], layout_match["confidence"], "layout_heuristic"

    if use_gpt4o_fallback and openai_api_key:
        try:
            name = detect_display_name_via_gpt4o(frame_path, openai_api_key, model=gpt4o_model)
            if name:
                return name, None, "gpt4o_vision"
        except Exception:
            logger.warning("GPT-4o vision speaker-name detection failed at %s; falling back to local heuristics", frame_path, exc_info=True)

    return None, None, None


def process_frame(
    frame: SceneFrame,
    paddle_engine: Any,
    zoom_settings: ZoomLayoutSettings | None = None,
    use_gpt4o_fallback: bool = True,
    openai_api_key: str | None = None,
    gpt4o_model: str = "gpt-4o",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> VisualFrameContext | None:
    """Run whole-frame OCR plus Zoom speaker-overlay detection on one frame.

    Returns None only if there is neither on-screen text nor a detected
    display name -- a frame with just one of the two still produces a
    record, since the point is one record per image. `end_time` and the
    final `slide_id` numbering are resolved afterward in `run_ocr`, once
    every frame's `start_time` is known.
    """
    zoom_settings = zoom_settings or ZoomLayoutSettings()

    lines = run_paddle_ocr(paddle_engine, frame.frame_path)
    avg_confidence = sum(line["confidence"] for line in lines) / len(lines) if lines else None

    needs_fallback = not lines or (avg_confidence or 0.0) < confidence_threshold
    if needs_fallback and use_gpt4o_fallback and openai_api_key:
        logger.info("PaddleOCR confidence low (%s) at %.2fs; falling back to GPT-4o vision", avg_confidence, frame.timestamp)
        try:
            lines = _wrap_plain_text_lines(run_gpt4o_ocr_fallback(frame.frame_path, openai_api_key, model=gpt4o_model))
        except Exception:
            # A single frame's fallback failing (rate limit, quota, network)
            # must not abort OCR for every other frame in the run. Fall back
            # to PaddleOCR's low-confidence result (possibly empty) instead.
            logger.warning("GPT-4o OCR fallback failed at %.2fs; using low-confidence PaddleOCR result instead", frame.timestamp, exc_info=True)

    title_line = _pick_title_line(lines)
    layout_match = detect_display_name_via_layout(lines, title_line)
    display_name, detection_confidence, display_name_source = detect_display_name(
        frame.frame_path,
        lines,
        title_line,
        paddle_engine,
        zoom_settings,
        use_gpt4o_fallback=use_gpt4o_fallback,
        openai_api_key=openai_api_key,
        gpt4o_model=gpt4o_model,
        confidence_threshold=confidence_threshold,
        layout_match=layout_match,
    )

    if not lines and display_name is None:
        logger.debug("Skipping frame at %.2fs: no OCR text or display name detected", frame.timestamp)
        return None

    # Only exclude the layout-heuristic match from bullets/paragraphs when
    # it's actually the line `display_name` came from -- if a border/badge
    # crop won instead, `layout_match` (if it fired at all) is an unrelated
    # name-shaped line elsewhere on the frame (e.g. a different gallery-view
    # participant's tag) and belongs in the slide content, not excluded.
    excluded_lines = [layout_match] if layout_match is not None and display_name_source == "layout_heuristic" else []

    slide_id = f"slide_{round(frame.timestamp * 100):06d}"
    return VisualFrameContext(
        slide_id=slide_id,
        start_time=frame.timestamp,
        end_time=None,
        frame_path=str(frame.frame_path),
        content=classify_content(lines, excluded_lines=excluded_lines),
        display_name=display_name,
        display_name_source=display_name_source,
        ocr_confidence=avg_confidence if lines else None,
        detection_confidence=detection_confidence,
    )


def run_ocr(
    scenes: list[SceneFrame],
    use_gpt4o_fallback: bool = True,
    openai_api_key: str | None = None,
    gpt4o_model: str = "gpt-4o",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    paddle_loader: Callable[[], Any] = load_paddle_ocr,
    zoom_settings: ZoomLayoutSettings | None = None,
) -> VisionTrackOutput:
    """OCR every representative frame from Scene Detection into a `VisionTrackOutput`.

    Frames are processed in chronological order; each frame's `end_time` is
    filled in afterward as the next frame's `start_time` (the last frame's
    `end_time` is left as None -- unknown without the video's total duration).
    """
    if not scenes:
        return VisionTrackOutput()

    zoom_settings = zoom_settings or ZoomLayoutSettings()
    engine = paddle_loader()
    frames: list[VisualFrameContext] = []

    for scene in sorted(scenes, key=lambda s: s.timestamp):
        context = process_frame(
            scene,
            engine,
            zoom_settings=zoom_settings,
            use_gpt4o_fallback=use_gpt4o_fallback,
            openai_api_key=openai_api_key,
            gpt4o_model=gpt4o_model,
            confidence_threshold=confidence_threshold,
        )
        if context is not None:
            frames.append(context)

    for current_frame, next_frame in zip(frames, frames[1:]):
        current_frame.end_time = next_frame.start_time

    logger.info("OCR extracted content from %d/%d frame(s)", len(frames), len(scenes))
    return VisionTrackOutput(frames=frames)
