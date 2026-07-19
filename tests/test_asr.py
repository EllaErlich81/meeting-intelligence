"""faster-whisper is loaded behind an injectable `model_loader`, and the
OpenAI Whisper API path behind an injectable `client_factory`, so neither
package needs to be installed to unit test the ASR stage."""

from __future__ import annotations

from types import SimpleNamespace

from meeting_intelligence.asr import build_prompt, hint_for_chunk, run_asr
from meeting_intelligence.models import SlideContent, VadSegment, VisualFrameContext


def _frame(timestamp: float, lines: list[str]) -> VisualFrameContext:
    return VisualFrameContext(
        slide_id=f"slide_{round(timestamp * 100):06d}",
        start_time=timestamp,
        frame_path="f.png",
        content=SlideContent(title=lines[0] if lines else None, raw_text="\n".join(lines)),
    )


def test_build_prompt_dedupes_and_joins():
    hints = ["Q1 Results", "Q1 Results", "  ", "Revenue", None]
    assert build_prompt(hints) == "Q1 Results, Revenue"


def test_build_prompt_none_when_no_hints():
    assert build_prompt(None) is None
    assert build_prompt([]) is None


def test_hint_for_chunk_uses_slide_active_at_chunk_end():
    frames = [
        _frame(0.0, ["Slide A", "Acme Corp"]),
        _frame(0.4, ["Slide B", "Q2 Roadmap"]),
    ]
    sorted_frames = sorted(frames, key=lambda f: f.start_time)
    timestamps = [f.start_time for f in sorted_frames]

    assert hint_for_chunk(0.3, sorted_frames, timestamps) == ["Slide A", "Acme Corp"]
    assert hint_for_chunk(0.8, sorted_frames, timestamps) == ["Slide B", "Q2 Roadmap"]


def test_hint_for_chunk_none_when_no_frame_precedes_it():
    frames = [_frame(5.0, ["Later slide"])]
    sorted_frames = sorted(frames, key=lambda f: f.start_time)
    timestamps = [f.start_time for f in sorted_frames]

    assert hint_for_chunk(1.0, sorted_frames, timestamps) is None


class _FakeWord:
    def __init__(self, start, end, word, probability=0.9):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class _FakeSegment:
    def __init__(self, text, words):
        self.text = text
        self.words = words


class _FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_chunk, word_timestamps, initial_prompt):
        self.calls.append({"initial_prompt": initial_prompt, "chunk_len": len(audio_chunk)})
        segments = [_FakeSegment("hello world", [_FakeWord(0.0, 0.4, "hello"), _FakeWord(0.4, 0.9, "world")])]
        return segments, SimpleNamespace(language="en")


def test_run_asr_offsets_word_timestamps_by_chunk_start(sample_wav):
    fake_model = _FakeModel()
    vad_segments = [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]
    vision_frames = [_frame(0.0, ["Acme Corp"])]

    results = run_asr(
        sample_wav,
        vad_segments,
        vision_frames=vision_frames,
        model_loader=lambda size, device: fake_model,
    )

    assert len(results) == 1
    assert results[0].text == "hello world"
    assert results[0].segment_id == "seg_0000"
    assert [w.text for w in results[0].words] == ["hello", "world"]
    assert results[0].words[0].start == 0.0
    assert results[0].words[1].start == 0.4
    assert fake_model.calls[0]["initial_prompt"] == "Acme Corp"


def test_run_asr_captures_detected_language(sample_wav):
    fake_model = _FakeModel()
    vad_segments = [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]

    results = run_asr(sample_wav, vad_segments, model_loader=lambda size, device: fake_model)

    assert results[0].language == "en"


def test_run_asr_openai_api_captures_language(sample_wav, monkeypatch):
    vad_segments = [VadSegment(segment_id="seg_0000", start_s=0.0, duration=0.3)]

    class _FakeTranscript:
        text = "hello"
        language = "en"

    class _FakeTranscriptions:
        def create(self, **kwargs):
            return _FakeTranscript()

    class _FakeAudio:
        transcriptions = _FakeTranscriptions()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.audio = _FakeAudio()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    results = run_asr(sample_wav, vad_segments, use_openai_api=True, openai_api_key="sk-fake")

    assert results[0].language == "en"
    assert results[0].segment_id == "seg_0000"


def test_run_asr_gives_each_chunk_only_its_own_active_slides_hint(sample_wav):
    """The regression this guards against: chunks used to all receive the
    same flattened hint list for the whole file. Each chunk here must only
    see the slide that was actually on screen when it ended."""
    fake_model = _FakeModel()
    vad_segments = [VadSegment(segment_id="seg_0000", start_s=0.0, duration=0.3), VadSegment(segment_id="seg_0001", start_s=0.5, duration=0.3)]
    vision_frames = [
        _frame(0.0, ["Slide A text"]),
        _frame(0.4, ["Slide B text"]),
    ]

    run_asr(sample_wav, vad_segments, vision_frames=vision_frames, model_loader=lambda size, device: fake_model)

    assert len(fake_model.calls) == 2
    assert fake_model.calls[0]["initial_prompt"] == "Slide A text"
    assert fake_model.calls[1]["initial_prompt"] == "Slide B text"


def test_run_asr_no_hint_when_no_vision_frames(sample_wav):
    fake_model = _FakeModel()
    vad_segments = [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]

    run_asr(sample_wav, vad_segments, model_loader=lambda size, device: fake_model)

    assert fake_model.calls[0]["initial_prompt"] is None


def test_run_asr_offsets_by_nonzero_segment_start(sample_wav):
    fake_model = _FakeModel()
    vad_segments = [VadSegment(segment_id="seg_0000", start_s=0.5, duration=0.5)]

    results = run_asr(sample_wav, vad_segments, model_loader=lambda size, device: fake_model)

    assert results[0].start == 0.5
    # word originally at 0.0s within the chunk -> 0.5s absolute
    assert results[0].words[0].start == 0.5


def test_run_asr_use_initial_prompt_false_skips_hint(sample_wav):
    fake_model = _FakeModel()
    vad_segments = [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]
    vision_frames = [_frame(0.0, ["Acme Corp"])]

    results = run_asr(
        sample_wav,
        vad_segments,
        vision_frames=vision_frames,
        model_loader=lambda size, device: fake_model,
        use_initial_prompt=False,
    )

    assert fake_model.calls[0]["initial_prompt"] is None
    assert results[0].text == "hello world"


def test_run_asr_use_initial_prompt_false_openai_api_path(sample_wav, monkeypatch):
    vad_segments = [VadSegment(segment_id="seg_0000", start_s=0.0, duration=0.3)]
    vision_frames = [_frame(0.0, ["Acme Corp"])]
    captured_prompts = []

    class _FakeTranscript:
        text = "hello"
        language = "en"

    class _FakeTranscriptions:
        def create(self, **kwargs):
            captured_prompts.append(kwargs.get("prompt"))
            return _FakeTranscript()

    class _FakeAudio:
        transcriptions = _FakeTranscriptions()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.audio = _FakeAudio()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    run_asr(
        sample_wav,
        vad_segments,
        vision_frames=vision_frames,
        use_openai_api=True,
        openai_api_key="sk-fake",
        use_initial_prompt=False,
    )

    assert captured_prompts == [None]


def test_run_asr_returns_empty_for_no_vad_segments(sample_wav):
    results = run_asr(sample_wav, [], model_loader=lambda size, device: _FakeModel())
    assert results == []


def test_run_asr_logs_initial_prompt_per_segment(sample_wav, caplog):
    fake_model = _FakeModel()
    # First segment ends before any slide has appeared -> no hint.
    # Second segment ends after the slide appears -> gets its text as a hint.
    vad_segments = [
        VadSegment(segment_id="seg_000000-000020", start_s=0.0, duration=0.2),
        VadSegment(segment_id="seg_000060-000090", start_s=0.6, duration=0.3),
    ]
    vision_frames = [_frame(0.5, ["Acme Corp"])]

    with caplog.at_level("INFO", logger="meeting_intelligence.asr"):
        run_asr(sample_wav, vad_segments, vision_frames=vision_frames, model_loader=lambda size, device: fake_model)

    log_text = caplog.text
    assert "seg_000000-000020" in log_text
    assert "initial_prompt=None" in log_text
    assert "seg_000060-000090" in log_text
    assert "initial_prompt='Acme Corp'" in log_text


def test_run_asr_logs_initial_prompt_per_segment_openai_api_path(sample_wav, caplog, monkeypatch):
    vad_segments = [VadSegment(segment_id="seg_000000-000030", start_s=0.0, duration=0.3)]
    vision_frames = [_frame(0.0, ["Acme Corp"])]

    class _FakeTranscript:
        text = "hello"

    class _FakeTranscriptions:
        def create(self, **kwargs):
            return _FakeTranscript()

    class _FakeAudio:
        transcriptions = _FakeTranscriptions()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.audio = _FakeAudio()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    with caplog.at_level("INFO", logger="meeting_intelligence.asr"):
        run_asr(
            sample_wav,
            vad_segments,
            vision_frames=vision_frames,
            use_openai_api=True,
            openai_api_key="sk-fake",
        )

    assert "seg_000000-000030" in caplog.text
    assert "initial_prompt='Acme Corp'" in caplog.text
