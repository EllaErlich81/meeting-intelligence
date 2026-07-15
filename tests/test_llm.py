"""FusionLLMProcessor takes injectable client factories, so neither
the `openai` nor `google-genai` package needs to be installed to
unit test the LLM enrichment stage."""

from __future__ import annotations

from meeting_intelligence.config import LLMProvider
from meeting_intelligence.llm import SYSTEM_PROMPT, FusionLLMProcessor
from meeting_intelligence.models import AlignedMeetingSegment, FullyAlignedTimeline, MeetingIntelligence

_SAMPLE_MI = MeetingIntelligence(
    summary="Team reviewed Q1 numbers.",
    topics=[],
    action_items=[],
)


def _timeline() -> FullyAlignedTimeline:
    return FullyAlignedTimeline(
        segments=[AlignedMeetingSegment(start=0.0, end=1.0, speaker="Presenter", transcript="Let's review Q1.")]
    )


class _FakeParsedMessage:
    def __init__(self, parsed):
        self.parsed = parsed


class _FakeChoice:
    def __init__(self, parsed):
        self.message = _FakeParsedMessage(parsed)


class _FakeOpenAIResponse:
    def __init__(self, parsed):
        self.choices = [_FakeChoice(parsed)]


class _FakeParseEndpoint:
    def __init__(self, parsed, captured):
        self._parsed = parsed
        self._captured = captured

    def parse(self, **kwargs):
        self._captured.append(kwargs)
        return _FakeOpenAIResponse(self._parsed)


class _FakeOpenAIClient:
    def __init__(self, parsed, captured):
        self.beta = type("Beta", (), {"chat": type("Chat", (), {"completions": _FakeParseEndpoint(parsed, captured)})()})()


def test_enrich_with_openai_returns_parsed_model():
    captured = []
    processor = FusionLLMProcessor(
        provider=LLMProvider.OPENAI,
        openai_model="gpt-4o",
        openai_client_factory=lambda api_key: _FakeOpenAIClient(_SAMPLE_MI, captured),
    )

    result = processor.enrich(_timeline())

    assert result.summary == _SAMPLE_MI.summary
    assert captured[0]["messages"][0]["content"] == SYSTEM_PROMPT
    assert captured[0]["response_format"] is MeetingIntelligence


def test_enrich_with_openai_stamps_provider_and_model():
    processor = FusionLLMProcessor(
        provider=LLMProvider.OPENAI,
        openai_model="gpt-4o",
        openai_client_factory=lambda api_key: _FakeOpenAIClient(_SAMPLE_MI, []),
    )

    result = processor.enrich(_timeline())

    assert result.llm_provider == "openai"
    assert result.llm_model == "gpt-4o"


def test_enrich_does_not_mutate_the_shared_parsed_object():
    """Regression test: stamping llm_provider/llm_model must not mutate the
    object returned by the client factory in place -- that object could be
    a cached/shared instance (as it is in these very tests), and mutating
    it would corrupt every other test/call holding a reference to it."""
    processor = FusionLLMProcessor(
        provider=LLMProvider.OPENAI,
        openai_client_factory=lambda api_key: _FakeOpenAIClient(_SAMPLE_MI, []),
    )

    processor.enrich(_timeline())

    assert _SAMPLE_MI.llm_provider is None
    assert _SAMPLE_MI.llm_model is None


class _FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class _FakeGeminiModels:
    def __init__(self, text, captured):
        self._text = text
        self._captured = captured

    def generate_content(self, model, contents, config):
        self._captured.append({"model": model, "contents": contents, "config": config})
        return _FakeGeminiResponse(self._text)


class _FakeGeminiClient:
    def __init__(self, text, captured):
        self.models = _FakeGeminiModels(text, captured)


def test_enrich_with_gemini_parses_json_text():
    captured = []
    text = _SAMPLE_MI.model_dump_json()

    processor = FusionLLMProcessor(
        provider=LLMProvider.GEMINI,
        gemini_model="gemini-1.5-flash",
        gemini_client_factory=lambda api_key: _FakeGeminiClient(text, captured),
    )

    result = processor.enrich(_timeline())

    assert result.summary == _SAMPLE_MI.summary
    assert captured[0]["config"].response_schema is MeetingIntelligence


def test_enrich_with_gemini_stamps_provider_and_model():
    text = _SAMPLE_MI.model_dump_json()
    processor = FusionLLMProcessor(
        provider=LLMProvider.GEMINI,
        gemini_model="gemini-1.5-flash",
        gemini_client_factory=lambda api_key: _FakeGeminiClient(text, []),
    )

    result = processor.enrich(_timeline())

    assert result.llm_provider == "gemini"
    assert result.llm_model == "gemini-1.5-flash"


def test_enrich_dispatches_to_openai_by_default():
    captured = []
    processor = FusionLLMProcessor(openai_client_factory=lambda api_key: _FakeOpenAIClient(_SAMPLE_MI, captured))
    processor.enrich(_timeline())
    assert len(captured) == 1
