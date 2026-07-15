"""PaddleOCR is loaded behind an injectable `paddle_loader`, and the GPT-4o
fallback behind an injectable `client_factory`, so neither package needs to
be installed to unit test the OCR stage."""

from __future__ import annotations

import cv2
import numpy as np

from meeting_intelligence.models import SceneFrame
from meeting_intelligence.ocr import (
    _MIN_CROP_HEIGHT_FOR_OCR,
    _ocr_crop_for_name,
    _upscale_for_ocr,
    classify_content,
    detect_display_name,
    detect_display_name_via_gpt4o,
    detect_display_name_via_layout,
    is_name_shaped,
    process_frame,
    run_gpt4o_ocr_fallback,
    run_ocr,
    run_paddle_ocr,
)
from meeting_intelligence.zoom_layout import ZoomLayoutSettings


class _FakePaddleEngine:
    """Stands in for a real PaddleOCR engine. `lines_by_path` maps a frame
    path to a list of either `(text, confidence)` pairs -- which exercises
    the "no position data" fallback path, since real PaddleOCR always
    returns boxes -- or `(text, confidence, top, height)` quadruples, which
    supply an explicit `rec_boxes` line so position-dependent behavior
    (title selection, the name-shape scan) can be tested directly."""

    def __init__(self, lines_by_path):
        self.lines_by_path = lines_by_path

    def predict(self, path):
        lines = self.lines_by_path.get(str(path), [])
        texts = [line[0] for line in lines]
        scores = [line[1] for line in lines]
        page = {"rec_texts": texts, "rec_scores": scores}
        if lines and len(lines[0]) > 2:
            page["rec_boxes"] = [[0, top, 100, top + height] for _text, _conf, top, height in lines]
        return [page]


def _line(text, confidence=0.9, top=0.0, height=1.0):
    return {"text": text, "confidence": confidence, "left": 0.0, "top": top, "width": 10.0, "height": height}


def _write_frame_with_green_border(path, size=(200, 200), rect=(10, 10, 90, 180)):
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)
    cv2.imwrite(str(path), frame)


# Zoom overlay detectors are disabled by default in most of these tests
# (via ZoomLayoutSettings(...)=False) so the whole-frame OCR path can be
# tested in isolation from the speaker-overlay path.
_NO_ZOOM = ZoomLayoutSettings(enable_active_speaker_border=False, enable_presentation_badge=False)


# --------------------------------------------------------------------------
# run_paddle_ocr
# --------------------------------------------------------------------------


def test_run_paddle_ocr_parses_rec_boxes(tmp_path):
    frame_path = tmp_path / "frame.png"
    engine = _FakePaddleEngine({str(frame_path): [("Title", 0.95, 10.0, 30.0), ("Body", 0.9, 70.0, 16.0)]})

    lines = run_paddle_ocr(engine, frame_path)

    assert lines[0]["text"] == "Title"
    assert lines[0]["confidence"] == 0.95
    assert lines[0]["top"] == 10.0
    assert lines[0]["height"] == 30.0
    assert lines[1]["text"] == "Body"


def test_run_paddle_ocr_falls_back_to_rec_polys(tmp_path):
    class _QuadEngine:
        def predict(self, path):
            return [
                {
                    "rec_texts": ["Slanted"],
                    "rec_scores": [0.9],
                    "rec_boxes": [],
                    "rec_polys": [[[5, 10], [50, 12], [48, 40], [3, 38]]],
                }
            ]

    lines = run_paddle_ocr(_QuadEngine(), tmp_path / "frame.png")
    assert lines[0]["text"] == "Slanted"
    assert lines[0]["left"] == 3.0
    assert lines[0]["top"] == 10.0
    assert lines[0]["width"] == 47.0
    assert lines[0]["height"] == 30.0


def test_run_paddle_ocr_synthesizes_uniform_boxes_when_no_position_data(tmp_path):
    frame_path = tmp_path / "frame.png"
    engine = _FakePaddleEngine({str(frame_path): [("First", 0.9), ("Second", 0.9)]})

    lines = run_paddle_ocr(engine, frame_path)

    assert [line["height"] for line in lines] == [1.0, 1.0]
    assert [line["top"] for line in lines] == [0.0, 1.0]


# --------------------------------------------------------------------------
# is_name_shaped
# --------------------------------------------------------------------------


def test_is_name_shaped_accepts_firstname_lastname():
    assert is_name_shaped("Jane Doe")


def test_is_name_shaped_accepts_honorific_and_middle_initial():
    assert is_name_shaped("Dr. Jane A. Doe")


def test_is_name_shaped_rejects_slide_title_text():
    assert not is_name_shaped("Q3 Roadmap")
    assert not is_name_shaped("Revenue up 12%")


def test_is_name_shaped_rejects_single_word():
    assert not is_name_shaped("Presenter")


def test_is_name_shaped_rejects_long_sentence():
    assert not is_name_shaped("This Quarter We Saw Significant Growth Across Regions")


def test_is_name_shaped_rejects_empty_string():
    assert not is_name_shaped("")


# --------------------------------------------------------------------------
# classify_content
# --------------------------------------------------------------------------


def test_classify_content_empty_lines():
    content = classify_content([])
    assert content.title is None
    assert content.bullet_points == []
    assert content.paragraphs == []
    assert content.raw_text == ""


def test_classify_content_first_line_is_title_when_heights_are_uniform():
    content = classify_content([_line("Q1 Results"), _line("Revenue up 12%")])
    assert content.title == "Q1 Results"


def test_classify_content_picks_largest_font_line_near_top_as_title():
    # A small "Subtitle" line comes first in reading order but doesn't clear
    # the 70%-of-max-height bar, so the larger "MAIN TITLE" line below it --
    # not whichever line the OCR engine happened to read first -- wins.
    lines = [_line("Subtitle", top=5.0, height=16.0), _line("MAIN TITLE", top=20.0, height=40.0), _line("body text", top=70.0, height=16.0)]
    content = classify_content(lines)
    assert content.title == "MAIN TITLE"
    assert content.bullet_points == ["Subtitle", "body text"]


def test_classify_content_strips_bullet_markers():
    lines = [_line("Agenda"), _line("• Review budget"), _line("- Discuss roadmap"), _line("1. Wrap up")]
    content = classify_content(lines)
    assert content.bullet_points == ["Review budget", "Discuss roadmap", "Wrap up"]


def test_classify_content_classifies_long_lines_as_paragraphs():
    long_line = "This quarter we saw significant growth across all our core product lines and regions"
    content = classify_content([_line("Summary"), _line(long_line)])
    assert content.paragraphs == [long_line]
    assert content.bullet_points == []


def test_classify_content_short_unmarked_line_is_a_bullet():
    content = classify_content([_line("Agenda"), _line("Budget review")])
    assert content.bullet_points == ["Budget review"]


def test_classify_content_raw_text_preserves_all_original_lines():
    texts = ["Q1 Results", "• Revenue up 12%", "Some longer explanatory sentence goes right here today"]
    content = classify_content([_line(t) for t in texts])
    assert content.raw_text == "\n".join(texts)


# --------------------------------------------------------------------------
# detect_display_name_via_layout
# --------------------------------------------------------------------------


def test_detect_display_name_via_layout_finds_name_shaped_line():
    lines = [_line("Q3 Roadmap", top=10, height=40), _line("shipping this quarter", top=70, height=16), _line("Jane Doe", top=300, height=14)]
    title_line = lines[0]

    match = detect_display_name_via_layout(lines, title_line)

    assert match["text"] == "Jane Doe"


def test_detect_display_name_via_layout_excludes_the_title_line():
    # A single, standalone name-shaped line with nothing else on screen is
    # ambiguous (title vs. name tag) -- classify_content/title-selection
    # treats it as the title, so the layout scan must not also claim it.
    lines = [_line("John Doe")]
    title_line = lines[0]

    assert detect_display_name_via_layout(lines, title_line) is None


def test_detect_display_name_via_layout_returns_none_when_nothing_matches():
    lines = [_line("Q1 Results"), _line("Revenue up 12%")]
    assert detect_display_name_via_layout(lines, lines[0]) is None


# --------------------------------------------------------------------------
# process_frame
# --------------------------------------------------------------------------


def test_process_frame_skips_when_nothing_found(tmp_path):
    frame = SceneFrame(timestamp=1.0, frame_path=str(tmp_path / "frame.png"))
    engine = _FakePaddleEngine({str(tmp_path / "frame.png"): []})

    context = process_frame(frame, engine, zoom_settings=_NO_ZOOM, use_gpt4o_fallback=False)

    assert context is None


def test_process_frame_returns_context_with_all_fields(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame = SceneFrame(timestamp=2.0, frame_path=str(frame_path))
    engine = _FakePaddleEngine({str(frame_path): [("Q1 Results", 0.95), ("Revenue up 12%", 0.85)]})

    context = process_frame(frame, engine, zoom_settings=_NO_ZOOM, use_gpt4o_fallback=False)

    assert context is not None
    assert context.frame_path == str(frame_path)
    assert context.start_time == 2.0
    assert context.end_time is None  # filled in later by run_ocr, not process_frame
    assert context.slide_id == "slide_000200"
    assert context.content.title == "Q1 Results"
    assert context.content.bullet_points == ["Revenue up 12%"]
    assert context.content.raw_text == "Q1 Results\nRevenue up 12%"
    assert context.ocr_confidence == round((0.95 + 0.85) / 2, 3)
    assert context.display_name is None
    assert context.detection_confidence is None


def test_process_frame_detects_display_name_via_layout_heuristic(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame = SceneFrame(timestamp=2.0, frame_path=str(frame_path))
    engine = _FakePaddleEngine(
        {str(frame_path): [("Q1 Results", 0.95, 10.0, 40.0), ("Revenue up 12%", 0.9, 70.0, 16.0), ("Jane Doe", 0.92, 300.0, 14.0)]}
    )

    context = process_frame(frame, engine, zoom_settings=_NO_ZOOM, use_gpt4o_fallback=False)

    assert context.display_name == "Jane Doe"
    assert context.display_name_source == "layout_heuristic"
    assert context.detection_confidence == 0.92
    assert context.content.title == "Q1 Results"
    assert context.content.bullet_points == ["Revenue up 12%"]


def test_process_frame_keeps_unrelated_name_shaped_line_when_border_wins(tmp_path):
    """Regression test: when display_name comes from the Zoom border/badge
    crop (not the layout heuristic), an unrelated name-shaped line
    elsewhere on the frame (e.g. a different gallery-view participant's
    tag) must stay in the slide content, not be silently excluded as if it
    were the chosen speaker's own name tag."""
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)
    engine = _FakePaddleEngine(
        {
            str(frame_path): [("Q1 Results", 0.95, 10.0, 40.0), ("Jane Doe", 0.92, 300.0, 14.0)],
            str(tmp_path / "frame_active_speaker.png"): [("John Smith", 0.9)],
        }
    )
    frame = SceneFrame(timestamp=2.0, frame_path=str(frame_path))

    context = process_frame(frame, engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=False)

    assert context.display_name == "John Smith"
    assert context.display_name_source == "active_speaker_border"
    assert "Jane Doe" in context.content.bullet_points


def test_process_frame_falls_back_to_gpt4o_when_confidence_low(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"fake-png-bytes")
    frame = SceneFrame(timestamp=3.0, frame_path=str(frame_path))
    engine = _FakePaddleEngine({str(frame_path): [("garbled", 0.1)]})

    def fake_gpt4o(path, api_key, model="gpt-4o", client_factory=None):
        return ["Recovered Title", "Recovered bullet"]

    monkeypatch.setattr("meeting_intelligence.ocr.run_gpt4o_ocr_fallback", fake_gpt4o)

    context = process_frame(
        frame, engine, zoom_settings=_NO_ZOOM, use_gpt4o_fallback=True, openai_api_key="sk-fake", confidence_threshold=0.5
    )

    assert context.content.raw_text == "Recovered Title\nRecovered bullet"


def test_process_frame_survives_gpt4o_fallback_failure(tmp_path, monkeypatch):
    """A single frame's GPT-4o fallback erroring (rate limit/quota/network)
    must degrade to the low-confidence PaddleOCR text, not raise -- otherwise
    one bad frame would abort OCR for an entire video."""
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"fake-png-bytes")
    frame = SceneFrame(timestamp=176.0, frame_path=str(frame_path))
    engine = _FakePaddleEngine({str(frame_path): [("garbled text", 0.1)]})

    def failing_gpt4o(path, api_key, model="gpt-4o", client_factory=None):
        raise RuntimeError("429 insufficient_quota")

    monkeypatch.setattr("meeting_intelligence.ocr.run_gpt4o_ocr_fallback", failing_gpt4o)

    context = process_frame(
        frame, engine, zoom_settings=_NO_ZOOM, use_gpt4o_fallback=True, openai_api_key="sk-fake", confidence_threshold=0.5
    )

    assert context is not None
    assert context.content.raw_text == "garbled text"


def test_process_frame_skips_when_fallback_fails_and_paddle_result_empty(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame.png"
    frame = SceneFrame(timestamp=1.0, frame_path=str(frame_path))
    engine = _FakePaddleEngine({str(frame_path): []})

    def failing_gpt4o(path, api_key, model="gpt-4o", client_factory=None):
        raise RuntimeError("429 insufficient_quota")

    monkeypatch.setattr("meeting_intelligence.ocr.run_gpt4o_ocr_fallback", failing_gpt4o)

    context = process_frame(frame, engine, zoom_settings=_NO_ZOOM, use_gpt4o_fallback=True, openai_api_key="sk-fake")

    assert context is None


def test_run_gpt4o_ocr_fallback_parses_lines(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"fake-png-bytes")

    class _FakeResponse:
        choices = [type("Choice", (), {"message": type("Msg", (), {"content": "Line one\nLine two\n"})()})]

    class _FakeChatCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        chat = _FakeChat()

    lines = run_gpt4o_ocr_fallback(frame_path, api_key="sk-fake", client_factory=lambda key: _FakeClient())
    assert lines == ["Line one", "Line two"]


# --------------------------------------------------------------------------
# run_ocr
# --------------------------------------------------------------------------


def test_run_ocr_continues_past_a_frame_whose_fallback_fails(tmp_path, monkeypatch):
    """The bug this regression-tests: run_ocr must keep processing later
    frames even if one frame's GPT-4o fallback raises."""
    frame_a = tmp_path / "a.png"
    frame_b = tmp_path / "b.png"
    scenes = [SceneFrame(timestamp=0.0, frame_path=str(frame_a)), SceneFrame(timestamp=1.0, frame_path=str(frame_b))]

    engine = _FakePaddleEngine(
        {
            str(frame_a): [("garbled", 0.1)],  # low confidence -> triggers (failing) fallback
            str(frame_b): [("Agenda", 0.95)],  # high confidence -> no fallback needed
        }
    )

    def failing_gpt4o(path, api_key, model="gpt-4o", client_factory=None):
        raise RuntimeError("429 insufficient_quota")

    monkeypatch.setattr("meeting_intelligence.ocr.run_gpt4o_ocr_fallback", failing_gpt4o)

    output = run_ocr(scenes, use_gpt4o_fallback=True, openai_api_key="sk-fake", paddle_loader=lambda: engine, zoom_settings=_NO_ZOOM)

    assert len(output.frames) == 2
    assert output.frames[0].content.raw_text == "garbled"
    assert output.frames[1].content.raw_text == "Agenda"


def test_run_ocr_keeps_each_frames_own_content_separate(tmp_path):
    frame_a = tmp_path / "a.png"
    frame_b = tmp_path / "b.png"
    scenes = [SceneFrame(timestamp=0.0, frame_path=str(frame_a)), SceneFrame(timestamp=1.0, frame_path=str(frame_b))]

    engine = _FakePaddleEngine(
        {
            str(frame_a): [("Welcome", 0.9), ("Agenda for today", 0.9)],
            str(frame_b): [("Agenda", 0.9)],
        }
    )

    output = run_ocr(scenes, use_gpt4o_fallback=False, paddle_loader=lambda: engine, zoom_settings=_NO_ZOOM)

    assert len(output.frames) == 2
    # each frame keeps only its own slide's text -- not a flattened bag of every slide
    assert output.frames[0].content.raw_text == "Welcome\nAgenda for today"
    assert output.frames[1].content.raw_text == "Agenda"
    assert output.frames[0].frame_path == str(frame_a)
    assert output.frames[1].frame_path == str(frame_b)


def test_run_ocr_fills_end_time_from_next_frame_and_leaves_last_one_none(tmp_path):
    frame_a = tmp_path / "a.png"
    frame_b = tmp_path / "b.png"
    frame_c = tmp_path / "c.png"
    scenes = [
        SceneFrame(timestamp=0.0, frame_path=str(frame_a)),
        SceneFrame(timestamp=5.0, frame_path=str(frame_b)),
        SceneFrame(timestamp=12.0, frame_path=str(frame_c)),
    ]
    engine = _FakePaddleEngine(
        {str(frame_a): [("A", 0.9)], str(frame_b): [("B", 0.9)], str(frame_c): [("C", 0.9)]}
    )

    output = run_ocr(scenes, use_gpt4o_fallback=False, paddle_loader=lambda: engine, zoom_settings=_NO_ZOOM)

    assert output.frames[0].start_time == 0.0
    assert output.frames[0].end_time == 5.0
    assert output.frames[1].start_time == 5.0
    assert output.frames[1].end_time == 12.0
    assert output.frames[2].start_time == 12.0
    assert output.frames[2].end_time is None


def test_run_ocr_processes_frames_in_chronological_order_regardless_of_input_order(tmp_path):
    frame_a = tmp_path / "a.png"
    frame_b = tmp_path / "b.png"
    # Scenes deliberately passed out of order.
    scenes = [SceneFrame(timestamp=5.0, frame_path=str(frame_b)), SceneFrame(timestamp=0.0, frame_path=str(frame_a))]
    engine = _FakePaddleEngine({str(frame_a): [("First", 0.9)], str(frame_b): [("Second", 0.9)]})

    output = run_ocr(scenes, use_gpt4o_fallback=False, paddle_loader=lambda: engine, zoom_settings=_NO_ZOOM)

    assert [f.start_time for f in output.frames] == [0.0, 5.0]
    assert [f.content.title for f in output.frames] == ["First", "Second"]


def test_run_ocr_returns_empty_output_for_no_scenes():
    output = run_ocr([])
    assert output.frames == []


# --------------------------------------------------------------------------
# detect_display_name_via_gpt4o
# --------------------------------------------------------------------------


class _FakeGpt4oVisionResponse:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


class _FakeGpt4oVisionClient:
    def __init__(self, answer):
        self._answer = answer

    class _Completions:
        def __init__(self, answer):
            self._answer = answer

        def create(self, **kwargs):
            return _FakeGpt4oVisionResponse(self._answer)

    @property
    def chat(self):
        completions = self._Completions(self._answer)
        return type("Chat", (), {"completions": completions})()


def test_detect_display_name_via_gpt4o_returns_the_name(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"fake-png-bytes")

    name = detect_display_name_via_gpt4o(frame_path, api_key="sk-fake", client_factory=lambda key: _FakeGpt4oVisionClient("Jane Roe"))
    assert name == "Jane Roe"


def test_detect_display_name_via_gpt4o_returns_none_for_none_answer(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"fake-png-bytes")

    name = detect_display_name_via_gpt4o(frame_path, api_key="sk-fake", client_factory=lambda key: _FakeGpt4oVisionClient("NONE"))
    assert name is None


def test_detect_display_name_via_gpt4o_rejects_non_name_answer(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"fake-png-bytes")

    name = detect_display_name_via_gpt4o(frame_path, api_key="sk-fake", client_factory=lambda key: _FakeGpt4oVisionClient("42"))
    assert name is None


# --------------------------------------------------------------------------
# detect_display_name
# --------------------------------------------------------------------------


def test_detect_display_name_prefers_zoom_crop_over_layout_heuristic(tmp_path):
    """Regression test: a frame can have both a genuine Zoom active-speaker
    border AND a name-shaped line that's actually ordinary slide text (e.g.
    an org name like "Acme Corp"). The Zoom-anchored crop must win, since
    it's tied to actual UI chrome rather than pattern-matching slide text
    shape."""
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)
    engine = _FakePaddleEngine({str(tmp_path / "frame_active_speaker.png"): [("Maria Alvarez", 0.95)]})

    lines = [_line("Q1 Results", top=10, height=40), _line("Acme Corp", top=300, height=14)]
    name, confidence, source = detect_display_name(
        frame_path, lines, lines[0], paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=False
    )
    assert name == "Maria Alvarez"
    assert source == "active_speaker_border"


def test_detect_display_name_prefers_layout_heuristic_over_gpt4o(tmp_path, monkeypatch):
    """When neither Zoom detector finds anything (no border/badge match in
    this fixture), the whole-frame name-shape scan is tried before GPT-4o
    vision: if it finds a name, GPT-4o must not even be consulted."""
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), np.zeros((200, 200, 3), dtype=np.uint8))  # no green border, no badge match
    engine = _FakePaddleEngine({})

    monkeypatch.setattr("meeting_intelligence.ocr.detect_display_name_via_gpt4o", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not be called")))

    lines = [_line("Q1 Results", top=10, height=40), _line("Jane Doe", top=300, height=14)]
    name, confidence, source = detect_display_name(
        frame_path, lines, lines[0], paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=True, openai_api_key="sk-fake"
    )
    assert name == "Jane Doe"
    assert confidence == 0.9
    assert source == "layout_heuristic"


def test_detect_display_name_uses_passed_layout_match_instead_of_recomputing(tmp_path):
    """`process_frame` already runs `detect_display_name_via_layout` once
    for its own `excluded_lines` bookkeeping; `detect_display_name` must
    reuse that result via the `layout_match` param instead of re-running
    the same whole-frame scan a second time. Proven here by passing a
    match that couldn't possibly come from `lines` itself (no name-shaped
    line in it) -- if `detect_display_name` recomputed instead of trusting
    the passed value, it would find nothing."""
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), np.zeros((200, 200, 3), dtype=np.uint8))  # no green border, no badge match
    engine = _FakePaddleEngine({})

    lines = [_line("Q1 Results", top=10, height=40)]  # no name-shaped line in here at all
    precomputed_match = _line("Precomputed Name", top=300, height=14)

    name, confidence, source = detect_display_name(
        frame_path,
        lines,
        lines[0],
        paddle_engine=engine,
        zoom_settings=_NO_ZOOM,
        use_gpt4o_fallback=False,
        layout_match=precomputed_match,
    )

    assert name == "Precomputed Name"
    assert source == "layout_heuristic"


def test_detect_display_name_falls_back_to_gpt4o_when_layout_finds_nothing(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), np.zeros((200, 200, 3), dtype=np.uint8))
    engine = _FakePaddleEngine({})

    monkeypatch.setattr("meeting_intelligence.ocr.detect_display_name_via_gpt4o", lambda *a, **kw: "Jane Roe")

    lines = [_line("Q1 Results"), _line("Revenue up 12%")]
    name, confidence, source = detect_display_name(
        frame_path, lines, lines[0], paddle_engine=engine, zoom_settings=_NO_ZOOM, use_gpt4o_fallback=True, openai_api_key="sk-fake"
    )
    assert name == "Jane Roe"
    assert confidence is None  # GPT-4o vision has no comparable numeric confidence
    assert source == "gpt4o_vision"


def test_detect_display_name_falls_back_to_local_crop_heuristics_when_layout_and_gpt4o_find_nothing(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)
    engine = _FakePaddleEngine({str(tmp_path / "frame_active_speaker.png"): [("Jane Roe", 0.95)]})

    monkeypatch.setattr("meeting_intelligence.ocr.detect_display_name_via_gpt4o", lambda *a, **kw: None)

    name, confidence, source = detect_display_name(
        frame_path, [], None, paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=True, openai_api_key="sk-fake"
    )
    assert name == "Jane Roe"
    assert source == "active_speaker_border"


def test_detect_display_name_falls_back_to_local_heuristics_when_gpt4o_raises(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)
    engine = _FakePaddleEngine({str(tmp_path / "frame_active_speaker.png"): [("Jane Roe", 0.95)]})

    def failing_gpt4o(*args, **kwargs):
        raise RuntimeError("429 insufficient_quota")

    monkeypatch.setattr("meeting_intelligence.ocr.detect_display_name_via_gpt4o", failing_gpt4o)

    name, confidence, source = detect_display_name(
        frame_path, [], None, paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=True, openai_api_key="sk-fake"
    )
    assert name == "Jane Roe"
    assert source == "active_speaker_border"


def test_detect_display_name_skips_gpt4o_vision_without_api_key(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)
    engine = _FakePaddleEngine({str(tmp_path / "frame_active_speaker.png"): [("Jane Roe", 0.95)]})

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("GPT-4o vision must not be called without an API key")

    monkeypatch.setattr("meeting_intelligence.ocr.detect_display_name_via_gpt4o", should_not_be_called)

    name, confidence, source = detect_display_name(
        frame_path, [], None, paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=True, openai_api_key=None
    )
    assert name == "Jane Roe"
    assert source == "active_speaker_border"


def test_detect_display_name_prefers_active_speaker_border_over_presentation_badge(tmp_path):
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)

    engine = _FakePaddleEngine(
        {
            str(tmp_path / "frame_active_speaker.png"): [("John Doe", 0.95)],
            str(tmp_path / "frame_presentation_badge.png"): [("Presenter", 0.95)],
        }
    )

    name, confidence, source = detect_display_name(
        frame_path, [], None, paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=False
    )
    assert name == "John Doe"
    assert confidence == 0.95
    assert source == "active_speaker_border"


def test_detect_display_name_falls_back_to_presentation_badge(tmp_path):
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), np.zeros((200, 200, 3), dtype=np.uint8))  # no green border

    engine = _FakePaddleEngine({str(tmp_path / "frame_presentation_badge.png"): [("Presenter", 0.9)]})

    name, confidence, source = detect_display_name(
        frame_path, [], None, paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=False
    )
    assert name == "Presenter"
    assert confidence == 0.9
    assert source == "presentation_badge"


def test_detect_display_name_skips_purely_numeric_lines_in_crop(tmp_path):
    """Regression test: a participant-count badge, timer, or other stray
    digits the crop happened to catch (e.g. "4") must never be returned as
    a display name -- a real name always has at least one letter."""
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)

    engine = _FakePaddleEngine(
        {
            str(tmp_path / "frame_active_speaker.png"): [("4", 0.9), ("Jane Roe", 0.9)],
        }
    )

    name, confidence, source = detect_display_name(
        frame_path, [], None, paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=False
    )
    assert name == "Jane Roe"
    assert confidence == 0.9
    assert source == "active_speaker_border"


def test_detect_display_name_none_when_only_numeric_lines_found(tmp_path):
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)

    engine = _FakePaddleEngine({str(tmp_path / "frame_active_speaker.png"): [("4", 0.9), ("00:12", 0.9)]})

    name, confidence, source = detect_display_name(
        frame_path, [], None, paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=False
    )
    assert name is None
    assert confidence is None
    assert source is None


def test_upscale_for_ocr_enlarges_crops_below_minimum_height():
    """Regression test: a dense gallery-view grid's name-label crop can be
    only ~11px tall in a low-res source frame -- too small for PaddleOCR's
    text detector to find any text region at all (measured against real
    footage). Upscaling gives it enough pixels to work with."""
    tiny_crop = np.zeros((11, 67, 3), dtype=np.uint8)
    upscaled = _upscale_for_ocr(tiny_crop)
    assert upscaled.shape[0] >= _MIN_CROP_HEIGHT_FOR_OCR
    # Aspect ratio is preserved (up to integer-pixel rounding).
    assert abs(upscaled.shape[1] / upscaled.shape[0] - 67 / 11) < 0.05


def test_upscale_for_ocr_leaves_tall_enough_crops_unchanged():
    crop = np.zeros((_MIN_CROP_HEIGHT_FOR_OCR + 20, 100, 3), dtype=np.uint8)
    assert _upscale_for_ocr(crop) is crop


def test_upscale_for_ocr_handles_zero_height_crop_without_dividing_by_zero():
    crop = np.zeros((0, 50, 3), dtype=np.uint8)
    assert _upscale_for_ocr(crop) is crop


def test_ocr_crop_for_name_writes_the_upscaled_crop_to_disk(tmp_path):
    """The crop PaddleOCR actually reads (and that gets saved as the
    debug artifact next to the frame) must be the upscaled version, not
    the original too-small-to-read one."""
    frame_path = tmp_path / "frame.png"
    tiny_crop = np.zeros((11, 67, 3), dtype=np.uint8)
    crop_path = tmp_path / "frame_active_speaker.png"

    engine = _FakePaddleEngine({str(crop_path): [("Jane Roe", 0.9)]})
    name, confidence = _ocr_crop_for_name(
        tiny_crop, frame_path, "active_speaker", engine, use_gpt4o_fallback=False, openai_api_key=None, gpt4o_model="gpt-4o", confidence_threshold=0.5
    )

    assert name == "Jane Roe"
    written = cv2.imread(str(crop_path))
    assert written.shape[0] >= _MIN_CROP_HEIGHT_FOR_OCR


def test_detect_display_name_respects_disabled_detectors(tmp_path):
    frame_path = tmp_path / "frame.png"
    _write_frame_with_green_border(frame_path)
    engine = _FakePaddleEngine({})

    name, confidence, source = detect_display_name(
        frame_path, [], None, paddle_engine=engine, zoom_settings=_NO_ZOOM, use_gpt4o_fallback=False
    )
    assert name is None
    assert confidence is None
    assert source is None


def test_detect_display_name_returns_none_for_unreadable_frame(tmp_path):
    engine = _FakePaddleEngine({})
    name, confidence, source = detect_display_name(
        tmp_path / "does_not_exist.png", [], None, paddle_engine=engine, zoom_settings=ZoomLayoutSettings(), use_gpt4o_fallback=False
    )
    assert name is None
    assert source is None
    assert confidence is None


def test_run_ocr_merges_content_with_display_name_and_detection_confidence(tmp_path):
    frame_path = tmp_path / "a.png"
    _write_frame_with_green_border(frame_path)

    engine = _FakePaddleEngine(
        {
            str(frame_path): [("Weekly Update", 0.95)],
            str(tmp_path / "a_active_speaker.png"): [("Jane Roe", 0.8)],
            str(tmp_path / "a_presentation_badge.png"): [("Jane Roe", 0.95)],
        }
    )

    scenes = [SceneFrame(timestamp=3.0, frame_path=str(frame_path))]
    output = run_ocr(scenes, use_gpt4o_fallback=False, paddle_loader=lambda: engine)

    assert output.frames[0].content.raw_text == "Weekly Update"
    assert output.frames[0].display_name == "Jane Roe"
    assert output.frames[0].detection_confidence == 0.8


def test_run_ocr_produces_frame_for_pure_gallery_view_with_no_slide_text(tmp_path):
    """A pure gallery-view frame with no slide content at all must still
    produce one record (empty content, but a resolved `display_name`),
    since the point is one record per image."""
    frame_path = tmp_path / "a.png"
    _write_frame_with_green_border(frame_path)

    engine = _FakePaddleEngine(
        {
            str(frame_path): [],  # no whole-frame slide text
            str(tmp_path / "a_active_speaker.png"): [("Jane Roe", 0.95)],
            str(tmp_path / "a_presentation_badge.png"): [],
        }
    )

    scenes = [SceneFrame(timestamp=3.0, frame_path=str(frame_path))]
    output = run_ocr(scenes, use_gpt4o_fallback=False, paddle_loader=lambda: engine)

    assert len(output.frames) == 1
    assert output.frames[0].content.raw_text == ""
    assert output.frames[0].content.title is None
    assert output.frames[0].display_name == "Jane Roe"
    assert output.frames[0].ocr_confidence is None  # no OCR text -> no meaningful OCR confidence


def test_run_ocr_picks_up_a_name_shaped_line_alongside_slide_content_via_layout_heuristic(tmp_path):
    """Unlike a lone name-shaped line (ambiguous, see above), a name-shaped
    line found *alongside* other slide content is exactly what a Zoom
    gallery-view name tag looks like in a whole-frame OCR pass -- the
    layout heuristic picks it up with no Zoom-specific border/badge
    detection needed at all."""
    frame_path = tmp_path / "a.png"
    cv2.imwrite(str(frame_path), np.zeros((200, 200, 3), dtype=np.uint8))

    engine = _FakePaddleEngine(
        {str(frame_path): [("Q1 Results", 0.95, 10.0, 40.0), ("Revenue up 12%", 0.9, 70.0, 16.0), ("John Doe", 0.92, 300.0, 14.0)]}
    )

    scenes = [SceneFrame(timestamp=1.0, frame_path=str(frame_path))]
    output = run_ocr(scenes, use_gpt4o_fallback=False, paddle_loader=lambda: engine, zoom_settings=_NO_ZOOM)

    assert output.frames[0].display_name == "John Doe"
    assert output.frames[0].display_name_source == "layout_heuristic"
