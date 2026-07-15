"""Orchestration test: Segmentation -> Diarization -> ASR -> Merge wiring,
with the vision_frames cross-module injection point verified explicitly."""

from __future__ import annotations

import asyncio

import meeting_intelligence.speech as speech_module
from meeting_intelligence.models import AsrSegmentResult, RawUtterance, SlideContent, SpeakerTurn, VadSegment, VisualFrameContext


def test_run_speech_pipeline_wires_stages_in_order(monkeypatch):
    calls = []
    frames = [
        VisualFrameContext(
            slide_id="slide_000000", start_time=0.0, frame_path="f.png", content=SlideContent(title="Slide A", raw_text="Acme Corp")
        )
    ]

    def fake_run_vad(wav_path):
        calls.append("vad")
        return [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]

    def fake_run_diarization(wav_path, hf_token=None):
        calls.append("diarize")
        return [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")]

    def fake_run_asr(
        wav_path, vad_segments, vision_frames=None, model_size="small", device="cpu", use_openai_api=False, openai_api_key=None
    ):
        calls.append(("asr", vision_frames))
        return [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hello")]

    def fake_merge(asr_segments, speaker_turns):
        calls.append("merge")
        return [RawUtterance(start=0.0, end=1.0, speaker_id="Speaker_00", transcript="hello")]

    monkeypatch.setattr(speech_module, "run_vad", fake_run_vad)
    monkeypatch.setattr(speech_module, "run_diarization", fake_run_diarization)
    monkeypatch.setattr(speech_module, "run_asr", fake_run_asr)
    monkeypatch.setattr(speech_module, "merge_words_into_turns", fake_merge)

    utterances = speech_module.run_speech_pipeline("audio.wav", vision_frames=frames)

    assert calls[0] == "vad"
    assert calls[1] == "diarize"
    assert calls[2] == ("asr", frames)
    assert calls[3] == "merge"
    assert utterances[0].transcript == "hello"


def test_run_speech_pipeline_async_wraps_sync_version(monkeypatch):
    monkeypatch.setattr(speech_module, "run_speech_pipeline", lambda *a, **kw: "sentinel")
    result = asyncio.run(speech_module.run_speech_pipeline_async("audio.wav"))
    assert result == "sentinel"
