"""Silero VAD is loaded behind an injectable `model_loader`, so these tests
never touch torch.hub or the network."""

from __future__ import annotations

from meeting_intelligence.segmentation import run_vad


class _FakeModel:
    pass


def _fake_model_loader():
    def get_speech_timestamps(wav, model, sampling_rate):
        # Two speech-active windows, in samples.
        return [
            {"start": 0, "end": sampling_rate},
            {"start": 2 * sampling_rate, "end": 3 * sampling_rate},
        ]

    def read_audio(path, sampling_rate):
        return b"fake-audio"

    return _FakeModel(), {"get_speech_timestamps": get_speech_timestamps, "read_audio": read_audio}


def test_run_vad_converts_sample_timestamps_to_seconds(tmp_path):
    wav_path = tmp_path / "audio.wav"
    wav_path.touch()

    segments = run_vad(wav_path, sample_rate=16_000, model_loader=_fake_model_loader)

    assert len(segments) == 2
    # segment_id encodes the segment's own start/end as zero-padded
    # centiseconds (no "."), not a running counter
    assert segments[0].segment_id == "seg_000000-000100"
    assert segments[0].start_s == 0.0
    assert segments[0].duration == 1.0
    assert segments[0].end_s == 1.0  # computed convenience property, not a serialized field
    assert segments[1].segment_id == "seg_000200-000300"
    assert segments[1].start_s == 2.0
    assert segments[1].duration == 1.0
    assert segments[1].end_s == 3.0


def test_segment_id_matches_expected_centisecond_format(tmp_path):
    """A segment from 1.85s to 6.73s must produce id 'seg_000185-000673'."""

    def loader():
        def get_speech_timestamps(wav, model, sampling_rate):
            return [{"start": int(1.85 * sampling_rate), "end": int(6.73 * sampling_rate)}]

        def read_audio(path, sampling_rate):
            return b"fake-audio"

        return _FakeModel(), {"get_speech_timestamps": get_speech_timestamps, "read_audio": read_audio}

    wav_path = tmp_path / "audio.wav"
    wav_path.touch()

    segments = run_vad(wav_path, sample_rate=16_000, model_loader=loader)

    assert segments[0].segment_id == "seg_000185-000673"


def test_run_vad_returns_empty_list_when_no_speech(tmp_path, caplog):
    def empty_loader():
        def get_speech_timestamps(wav, model, sampling_rate):
            return []

        def read_audio(path, sampling_rate):
            return b""

        return _FakeModel(), {"get_speech_timestamps": get_speech_timestamps, "read_audio": read_audio}

    wav_path = tmp_path / "silent.wav"
    wav_path.touch()

    segments = run_vad(wav_path, model_loader=empty_loader)
    assert segments == []
