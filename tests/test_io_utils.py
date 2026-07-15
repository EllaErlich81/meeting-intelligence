from __future__ import annotations

from meeting_intelligence.io_utils import read_model, read_model_list, write_model, write_model_list
from meeting_intelligence.models import RawUtterance, SpeakerNameMap


def test_write_and_read_single_model_roundtrip(tmp_path):
    model = SpeakerNameMap(mapping={"Speaker_00": "Presenter"})
    path = tmp_path / "map.json"

    write_model(path, model)
    loaded = read_model(path, SpeakerNameMap)

    assert loaded == model


def test_write_and_read_model_list_roundtrip(tmp_path):
    models = [
        RawUtterance(start=0.0, end=1.0, speaker_id="Speaker_00", transcript="hi"),
        RawUtterance(start=1.0, end=2.0, speaker_id="Speaker_01", transcript="hello"),
    ]
    path = tmp_path / "utterances.json"

    write_model_list(path, models)
    loaded = read_model_list(path, RawUtterance)

    assert loaded == models


def test_write_model_list_produces_readable_json(tmp_path):
    path = tmp_path / "empty.json"
    write_model_list(path, [])
    assert path.read_text(encoding="utf-8").strip() == "[]"
