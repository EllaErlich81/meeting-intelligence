"""Part C / LLM Enrichment: FusionLLMProcessor.

Feeds the fully-fused meeting timeline to GPT-4o or Gemini Flash using
each provider's native structured-output mode, constrained to the
`MeetingIntelligence` schema, and returns the parsed
`meeting_intelligence.json` payload.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import LLMProvider
from .models import FullyAlignedTimeline, MeetingIntelligence

logger = logging.getLogger(__name__)

# Exact system instruction required by the brief to control hallucination
# and preserve grounding in the source timeline.

SYSTEM_PROMPT = """You are an advanced Multimodal Conversation Intelligence Engine. \
Your input is a strictly sequential, interval-aligned meeting timeline containing verbal \
transcripts seamlessly coupled with screen-captured OCR text.

Instructions:
1. Grounding: Maintain absolute faithfulness to the timestamps, speaker roles, and OCR \
texts. If a claim cannot be verified directly by an utterance or an OCR field, do not \
include it.
2. Cross-Modality: Synthesize verbal statements with visible items on screen. If a \
speaker refers to charts or text lines, match them with the corresponding 'slide' node.
3. Multilingual Capability: Generate the intelligence summaries and text fields in the \
meeting's native language, but preserve English for metadata keys.
4. Completeness: Ensure every single action item has an 'evidence_quote' mapped word-\
for-word from the transcript."""

# SYSTEM_PROMPT = """
# You are an expert Multimodal AI Fusion system designed for conversation intelligence.
# Your task is to ingest a combined chronological timeline of a recorded meeting containing both audio utterances (speech) and visual events (OCR from slide/scene changes).
#
# You must analyze this timeline and produce a structured, high-quality meeting intelligence report.
#
# CRITICAL RULES FOR ACCURACY & GROUNDING:
# 1. STRICT GROUNDING: Every topic and action item MUST be strictly grounded in the provided timeline. Do not assume or extrapolate info not present.
# 2. EVIDENCE MATCHING: For every topic, you must extract the exact timestamps, the relevant speakers, and any specific on-screen text (OCR) that was visible at that moment as evidence.
# 3. ABSOLUTE HALLLUCINATION CONTROL: If a task or action item is mentioned but no clear assignee can be inferred from the text, leave the 'assignee' field as null. Do not guess names or speakers.
# 4. MULTIMODAL SYNTHESIS: Connect what was said with what was shown. If a speaker says "as you can see here" at timestamp 45.2, correlate their words with the visual event at or immediately before timestamp 45.2.
#
# Output Requirements:
# - The final output must strictly follow the JSON schema provided.
# - Ensure the 'summary' fields are concise yet highly informative.
# - Keep the language of the summaries aligned with the primary language of the conversation.
# """


def default_openai_client_factory(api_key: str | None) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def default_gemini_client_factory(api_key: str | None) -> Any:
    """Isolated so tests can inject a fake client.

    Uses the `google-genai` SDK (`google.genai`), not the deprecated
    `google-generativeai` package -- the deprecated SDK's schema converter
    rejects any Pydantic model with an `Optional`/default-valued field
    ("Unknown field for Schema: default"), which every non-trivial nested
    model in this project's `MeetingIntelligence` schema has (e.g.
    `ActionItem.assignee`, `Evidence.visual_reference`). `google-genai`'s
    schema transformer explicitly supports this pattern (see its
    `handle_null_fields`), and is Google's actively maintained replacement.
    """
    from google import genai

    return genai.Client(api_key=api_key)


class FusionLLMProcessor:
    """Turns a `FullyAlignedTimeline` into structured `MeetingIntelligence`.

    `openai_client_factory` / `gemini_client_factory` default to lazily
    importing the real SDKs, but can be swapped for fakes in tests without
    installing either package.
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o",
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-1.5-flash",
        openai_client_factory: Callable[[str | None], Any] = default_openai_client_factory,
        gemini_client_factory: Callable[[str | None], Any] = default_gemini_client_factory,
    ) -> None:
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.openai_client_factory = openai_client_factory
        self.gemini_client_factory = gemini_client_factory

    def enrich(self, timeline: FullyAlignedTimeline) -> MeetingIntelligence:
        payload = timeline.model_dump_json(indent=2)
        if self.provider == LLMProvider.GEMINI:
            result = self._enrich_with_gemini(payload)
            model_name = self.gemini_model
        else:
            result = self._enrich_with_openai(payload)
            model_name = self.openai_model
        # A copy, not an in-place mutation: `result` may be a cached/shared
        # object from the client factory (e.g. in tests), and this is
        # metadata we already know locally -- not something to trust the
        # model to report accurately about itself -- so it's stamped on
        # here rather than requested of the LLM.
        return result.model_copy(update={"llm_provider": self.provider.value, "llm_model": model_name})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def _enrich_with_openai(self, payload: str) -> MeetingIntelligence:
        client = self.openai_client_factory(self.openai_api_key)
        response = client.beta.chat.completions.parse(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            response_format=MeetingIntelligence,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed structured output")
        logger.info(
            "LLM enrichment (OpenAI/%s) produced %d topic(s), %d action item(s)",
            self.openai_model,
            len(parsed.topics),
            len(parsed.action_items),
        )
        return parsed

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def _enrich_with_gemini(self, payload: str) -> MeetingIntelligence:
        from google.genai import types

        client = self.gemini_client_factory(self.gemini_api_key)
        response = client.models.generate_content(
            model=self.gemini_model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=MeetingIntelligence,
            ),
        )
        parsed = MeetingIntelligence.model_validate_json(response.text)
        logger.info(
            "LLM enrichment (Gemini/%s) produced %d topic(s), %d action item(s)",
            self.gemini_model,
            len(parsed.topics),
            len(parsed.action_items),
        )
        return parsed
