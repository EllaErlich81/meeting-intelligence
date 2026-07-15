"""pyannote is loaded behind an injectable `pipeline_loader`; the graceful
single-speaker fallback is a hard requirement from the brief and is tested
explicitly for both a loader failure and an inference failure."""

from __future__ import annotations

from collections import namedtuple

import pytest

from meeting_intelligence.diarization import FALLBACK_SPEAKER_ID, run_diarization

_Segment = namedtuple("_Segment", ["start", "end"])


class _FakeDiarization:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label: bool = True):
        yield from self._tracks


class _FakeDiarizeOutput:
    """Mimics pyannote.audio 4.x's `DiarizeOutput` wrapper dataclass, which
    holds the `Annotation` behind a `.speaker_diarization` attribute instead
    of being the `Annotation` itself (pyannote.audio 3.x's shape)."""

    def __init__(self, annotation):
        self.speaker_diarization = annotation


def test_run_diarization_maps_raw_labels_to_speaker_ids():
    tracks = [
        (_Segment(0.0, 1.0), "track_a", "SPEAKER_07"),
        (_Segment(1.0, 2.0), "track_b", "SPEAKER_02"),
        (_Segment(2.0, 3.0), "track_c", "SPEAKER_07"),
    ]

    def loader(hf_token):
        return lambda wav_path: _FakeDiarization(tracks)

    turns = run_diarization("unused.wav", hf_token="fake-token", pipeline_loader=loader)

    assert [t.speaker_id for t in turns] == ["Speaker_00", "Speaker_01", "Speaker_00"]
    assert turns[0].start == 0.0 and turns[0].end == 1.0


def test_run_diarization_supports_pyannote_4x_diarize_output_wrapper():
    """pyannote.audio 4.x wraps the Annotation in a DiarizeOutput dataclass
    (`.speaker_diarization`) instead of returning it directly -- this is a
    regression test for that exact shape mismatch."""
    tracks = [(_Segment(0.0, 1.0), "track_a", "SPEAKER_00")]

    def loader(hf_token):
        return lambda wav_path: _FakeDiarizeOutput(_FakeDiarization(tracks))

    turns = run_diarization("unused.wav", hf_token="fake-token", pipeline_loader=loader)

    assert [t.speaker_id for t in turns] == ["Speaker_00"]
    assert turns[0].start == 0.0 and turns[0].end == 1.0


def test_run_diarization_falls_back_when_loader_raises(sample_wav):
    def failing_loader(hf_token):
        raise RuntimeError("no HF token configured")

    turns = run_diarization(sample_wav, hf_token=None, pipeline_loader=failing_loader)

    assert len(turns) == 1
    assert turns[0].speaker_id == FALLBACK_SPEAKER_ID
    assert turns[0].start == 0.0
    assert turns[0].end == pytest.approx(1.0, abs=0.05)


def test_run_diarization_falls_back_when_pipeline_raises_during_inference(sample_wav):
    def loader(hf_token):
        def pipeline(wav_path):
            raise RuntimeError("inference blew up")

        return pipeline

    turns = run_diarization(sample_wav, hf_token="token", pipeline_loader=loader)
    assert len(turns) == 1
    assert turns[0].speaker_id == FALLBACK_SPEAKER_ID


def test_run_diarization_falls_back_when_no_turns_produced(sample_wav):
    def loader(hf_token):
        return lambda wav_path: _FakeDiarization([])

    turns = run_diarization(sample_wav, hf_token="token", pipeline_loader=loader)
    assert len(turns) == 1
    assert turns[0].speaker_id == FALLBACK_SPEAKER_ID
