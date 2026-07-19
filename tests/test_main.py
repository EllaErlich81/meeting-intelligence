"""CLI tests: verify each stage subcommand reads its default artifacts from
--output_dir, writes its own artifact, and that the CLI surfaces stage
failures as a clean exit code rather than a raw traceback.
"""

from __future__ import annotations

import meeting_intelligence.main as main_module
from meeting_intelligence.io_utils import read_model, read_model_list, write_model, write_model_list
from meeting_intelligence.models import (
    AsrSegmentResult,
    IngestionResult,
    MeetingIntelligence,
    RawUtterance,
    SceneFrame,
    SlideContent,
    SpeakerNameMap,
    SpeakerTurn,
    TranscriptFile,
    TranscriptSegment,
    VadSegment,
    VadSegmentsFile,
    VisionTrackOutput,
    VisualFrameContext,
    VisualSpeakerEvent,
    VisualSpeakerEventsFile,
)


def _frame(timestamp: float, title: str | None = None, raw_text: str = "", display_name: str | None = None) -> VisualFrameContext:
    return VisualFrameContext(
        slide_id=f"slide_{round(timestamp * 100):06d}",
        start_time=timestamp,
        frame_path="f.png",
        content=SlideContent(title=title, raw_text=raw_text),
        display_name=display_name,
    )


def test_build_parser_registers_every_stage_subcommand():
    parser = main_module.build_parser()
    subcommands = {action.dest: action.choices for action in parser._subparsers._group_actions}["command"]
    expected = {
        "ingest",
        "scene-detect",
        "ocr",
        "visual-speaker",
        "vision",
        "vad",
        "diarize",
        "asr",
        "merge",
        "speech",
        "speaker-name",
        "fuse",
        "llm",
        "pipeline",
    }
    assert expected.issubset(subcommands.keys())


def test_cmd_ingest_writes_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "run_ingestion",
        lambda video_path, output_dir: IngestionResult(video_path=str(video_path), wav_path=str(output_dir / "a.wav"), duration_sec=5.0),
    )
    args = main_module.build_parser().parse_args(["ingest", "--video_path", "in.mp4", "--output_dir", str(tmp_path)])
    args.func(args)

    result = read_model(tmp_path / "ingestion.json", IngestionResult)
    assert result.duration_sec == 5.0


def test_cmd_vad_writes_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "run_vad", lambda wav_path: [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)])
    args = main_module.build_parser().parse_args(["vad", "--wav_path", "a.wav", "--output_dir", str(tmp_path)])
    args.func(args)

    result = read_model(tmp_path / "vad_segments.json", VadSegmentsFile)
    assert result.segments == [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]


def test_cmd_vad_skips_if_exists(tmp_path, monkeypatch):
    write_model(tmp_path / "vad_segments.json", VadSegmentsFile(segments=[VadSegment(segment_id="seg_cached", start_s=0.0, duration=1.0)]))

    def fail_if_called(wav_path):
        raise AssertionError("run_vad should not be called when the artifact already exists")

    monkeypatch.setattr(main_module, "run_vad", fail_if_called)
    args = main_module.build_parser().parse_args(["vad", "--wav_path", "a.wav", "--output_dir", str(tmp_path), "--skip-if-exists"])
    args.func(args)

    result = read_model(tmp_path / "vad_segments.json", VadSegmentsFile)
    assert result.segments[0].segment_id == "seg_cached"


def test_cmd_vad_does_not_skip_by_default(tmp_path, monkeypatch):
    write_model(tmp_path / "vad_segments.json", VadSegmentsFile(segments=[VadSegment(segment_id="seg_cached", start_s=0.0, duration=1.0)]))
    monkeypatch.setattr(main_module, "run_vad", lambda wav_path: [VadSegment(segment_id="seg_fresh", start_s=0.0, duration=1.0)])

    args = main_module.build_parser().parse_args(["vad", "--wav_path", "a.wav", "--output_dir", str(tmp_path)])
    args.func(args)

    result = read_model(tmp_path / "vad_segments.json", VadSegmentsFile)
    assert result.segments[0].segment_id == "seg_fresh"


def test_cmd_scene_detect_writes_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_module, "detect_scenes", lambda video_path, output_dir, sample_fps, diff_threshold: [SceneFrame(timestamp=0.0, frame_path="f.png")]
    )
    args = main_module.build_parser().parse_args(["scene-detect", "--video_path", "in.mp4", "--output_dir", str(tmp_path)])
    args.func(args)

    scenes = read_model_list(tmp_path / "scenes.json", SceneFrame)
    assert len(scenes) == 1


def test_cmd_ocr_reads_scenes_and_writes_vision_track(tmp_path, monkeypatch):
    write_model_list(tmp_path / "scenes.json", [SceneFrame(timestamp=0.0, frame_path="f.png")])
    monkeypatch.setattr(
        main_module,
        "run_ocr",
        lambda scenes, use_gpt4o_fallback, openai_api_key, gpt4o_model, zoom_settings: VisionTrackOutput(
            frames=[_frame(0.0, title="hi", raw_text="hi")]
        ),
    )
    args = main_module.build_parser().parse_args(["ocr", "--output_dir", str(tmp_path)])
    args.func(args)

    output = read_model(tmp_path / "vision_track.json", VisionTrackOutput)
    assert output.frames[0].content.raw_text == "hi"


def test_cmd_visual_speaker_reads_vision_track_and_writes_events(tmp_path):
    frame = VisualFrameContext(
        slide_id="slide_000000",
        start_time=0.0,
        end_time=1.0,
        frame_path="f.png",
        content=SlideContent(),
        display_name="John Doe",
        display_name_source="active_speaker_border",
        detection_confidence=0.9,
    )
    write_model(tmp_path / "vision_track.json", VisionTrackOutput(frames=[frame]))

    args = main_module.build_parser().parse_args(["visual-speaker", "--output_dir", str(tmp_path)])
    args.func(args)

    result = read_model(tmp_path / "visual_speaker_events.json", VisualSpeakerEventsFile)
    assert len(result.events) == 1
    assert result.events[0].display_name == "John Doe"
    assert result.events[0].layout == "gallery_view"


def test_cmd_vision_runs_scene_detect_then_ocr(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        main_module,
        "detect_scenes",
        lambda video_path, output_dir, sample_fps, diff_threshold: calls.append("scenes") or [SceneFrame(timestamp=0.0, frame_path="f.png")],
    )
    monkeypatch.setattr(
        main_module,
        "run_ocr",
        lambda scenes, use_gpt4o_fallback, openai_api_key, gpt4o_model, zoom_settings: calls.append("ocr")
        or VisionTrackOutput(frames=[_frame(0.0, title="hi", raw_text="hi")]),
    )
    args = main_module.build_parser().parse_args(["vision", "--video_path", "in.mp4", "--output_dir", str(tmp_path)])
    args.func(args)

    assert calls == ["scenes", "ocr"]
    assert (tmp_path / "vision_track.json").is_file()


def test_cmd_diarize_writes_speaker_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_module, "run_diarization", lambda wav_path, hf_token, duration_sec: [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")]
    )
    args = main_module.build_parser().parse_args(["diarize", "--wav_path", "a.wav", "--output_dir", str(tmp_path)])
    args.func(args)

    turns = read_model_list(tmp_path / "speaker_turns.json", SpeakerTurn)
    assert turns[0].speaker_id == "Speaker_00"


def test_cmd_speech_runs_full_part_a_and_writes_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "run_vad", lambda wav_path: [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)])
    monkeypatch.setattr(
        main_module, "run_diarization", lambda wav_path, hf_token, duration_sec: [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")]
    )
    monkeypatch.setattr(
        main_module,
        "run_asr",
        lambda wav_path, vad_segments, vision_frames, model_size, device, use_openai_api, openai_api_key, use_initial_prompt: [
            AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hi")
        ],
    )
    monkeypatch.setattr(
        main_module,
        "merge_words_into_turns",
        lambda asr, turns: [RawUtterance(start=0.0, end=1.0, speaker_id="Speaker_00", transcript="hi")],
    )

    args = main_module.build_parser().parse_args(["speech", "--wav_path", "a.wav", "--output_dir", str(tmp_path)])
    args.func(args)

    transcript = read_model(tmp_path / "transcript.json", TranscriptFile)
    assert transcript.segments[0].text == "hi"


def test_cmd_speech_skips_asr_but_not_merge_when_only_asr_artifact_exists(tmp_path, monkeypatch):
    """transcript.json is absent, so merge must still run; asr_segments.json is
    present, so run_asr itself must be skipped and its cached output reused."""
    write_model(tmp_path / "vad_segments.json", VadSegmentsFile(segments=[VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]))
    write_model_list(tmp_path / "speaker_turns.json", [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")])
    write_model_list(tmp_path / "asr_segments.json", [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="cached")])

    monkeypatch.setattr(main_module, "run_vad", lambda wav_path: (_ for _ in ()).throw(AssertionError("run_vad should be skipped")))
    monkeypatch.setattr(
        main_module, "run_diarization", lambda wav_path, hf_token, duration_sec: (_ for _ in ()).throw(AssertionError("run_diarization should be skipped"))
    )
    monkeypatch.setattr(main_module, "run_asr", lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_asr should be skipped")))
    monkeypatch.setattr(
        main_module,
        "merge_words_into_turns",
        lambda asr, turns: [RawUtterance(start=0.0, end=1.0, speaker_id="Speaker_00", transcript=asr[0].text)],
    )

    args = main_module.build_parser().parse_args(["speech", "--wav_path", "a.wav", "--output_dir", str(tmp_path), "--skip-if-exists"])
    args.func(args)

    transcript = read_model(tmp_path / "transcript.json", TranscriptFile)
    assert transcript.segments[0].text == "cached"


def test_cmd_pipeline_dispatches_to_run_full_pipeline(tmp_path, monkeypatch):
    captured = {}

    def fake_run_full_pipeline(video_path, output_dir, settings, skip_if_exists=False):
        captured["video_path"] = video_path
        captured["provider"] = settings.llm_provider
        captured["skip_if_exists"] = skip_if_exists
        return MeetingIntelligence(summary="ok", topics=[], action_items=[])

    monkeypatch.setattr(main_module.pl, "run_full_pipeline", fake_run_full_pipeline)

    args = main_module.build_parser().parse_args(
        ["pipeline", "--video_path", "in.mp4", "--output_dir", str(tmp_path), "--llm-provider", "gemini", "--skip-if-exists"]
    )
    args.func(args)

    assert captured["video_path"] == "in.mp4"
    assert captured["provider"].value == "gemini"
    assert captured["skip_if_exists"] is True


def test_cmd_asr_picks_up_vision_frames_from_output_dir(tmp_path, monkeypatch):
    write_model(tmp_path / "vad_segments.json", VadSegmentsFile(segments=[VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]))
    frame = _frame(0.0, title="Slide A", raw_text="Acme Corp")
    write_model(tmp_path / "vision_track.json", VisionTrackOutput(frames=[frame]))

    captured = {}

    def fake_run_asr(wav_path, vad_segments, vision_frames, model_size, device, use_openai_api, openai_api_key, use_initial_prompt):
        captured["vision_frames"] = vision_frames
        return [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hi")]

    monkeypatch.setattr(main_module, "run_asr", fake_run_asr)

    args = main_module.build_parser().parse_args(["asr", "--wav_path", "a.wav", "--output_dir", str(tmp_path)])
    args.func(args)

    assert captured["vision_frames"] == [frame]
    results = read_model_list(tmp_path / "asr_segments.json", AsrSegmentResult)
    assert results[0].text == "hi"


def test_cmd_asr_runs_without_vision_artifact(tmp_path, monkeypatch):
    write_model(tmp_path / "vad_segments.json", VadSegmentsFile(segments=[VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]))
    captured = {}

    def fake_run_asr(wav_path, vad_segments, vision_frames, model_size, device, use_openai_api, openai_api_key, use_initial_prompt):
        captured["vision_frames"] = vision_frames
        return []

    monkeypatch.setattr(main_module, "run_asr", fake_run_asr)
    args = main_module.build_parser().parse_args(["asr", "--wav_path", "a.wav", "--output_dir", str(tmp_path)])
    args.func(args)

    assert captured["vision_frames"] is None


def test_cmd_asr_no_initial_prompt_flag_disables_hint(tmp_path, monkeypatch):
    write_model(tmp_path / "vad_segments.json", VadSegmentsFile(segments=[VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)]))
    captured = {}

    def fake_run_asr(wav_path, vad_segments, vision_frames, model_size, device, use_openai_api, openai_api_key, use_initial_prompt):
        captured["use_initial_prompt"] = use_initial_prompt
        return []

    monkeypatch.setattr(main_module, "run_asr", fake_run_asr)
    args = main_module.build_parser().parse_args(["asr", "--wav_path", "a.wav", "--output_dir", str(tmp_path), "--no-initial-prompt"])
    args.func(args)

    assert captured["use_initial_prompt"] is False


def test_cmd_speech_no_initial_prompt_flag_disables_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "run_vad", lambda wav_path: [VadSegment(segment_id="seg_0000", start_s=0.0, duration=1.0)])
    monkeypatch.setattr(
        main_module, "run_diarization", lambda wav_path, hf_token, duration_sec: [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")]
    )
    captured = {}

    def fake_run_asr(wav_path, vad_segments, vision_frames, model_size, device, use_openai_api, openai_api_key, use_initial_prompt):
        captured["use_initial_prompt"] = use_initial_prompt
        return [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hi")]

    monkeypatch.setattr(main_module, "run_asr", fake_run_asr)
    monkeypatch.setattr(
        main_module,
        "merge_words_into_turns",
        lambda asr, turns: [RawUtterance(start=0.0, end=1.0, speaker_id="Speaker_00", transcript="hi")],
    )

    args = main_module.build_parser().parse_args(["speech", "--wav_path", "a.wav", "--output_dir", str(tmp_path), "--no-initial-prompt"])
    args.func(args)

    assert captured["use_initial_prompt"] is False


def test_cmd_merge_reads_defaults_and_writes_transcript(tmp_path, monkeypatch):
    write_model_list(tmp_path / "asr_segments.json", [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hi")])
    write_model_list(tmp_path / "speaker_turns.json", [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")])

    monkeypatch.setattr(
        main_module,
        "merge_words_into_turns",
        lambda asr, turns: [RawUtterance(start=0.0, end=1.0, speaker_id="Speaker_00", transcript="hi")],
    )

    args = main_module.build_parser().parse_args(["merge", "--output_dir", str(tmp_path)])
    args.func(args)

    transcript = read_model(tmp_path / "transcript.json", TranscriptFile)
    assert transcript.segments[0].text == "hi"


def test_cmd_merge_passes_max_segment_duration_sec_flag_through(tmp_path, monkeypatch):
    write_model_list(tmp_path / "asr_segments.json", [AsrSegmentResult(segment_id="seg_0000", start=0.0, end=1.0, text="hi")])
    write_model_list(tmp_path / "speaker_turns.json", [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")])
    monkeypatch.setattr(
        main_module,
        "merge_words_into_turns",
        lambda asr, turns: [RawUtterance(start=0.0, end=1.0, speaker_id="Speaker_00", transcript="hi")],
    )

    captured = {}

    def fake_build_transcript(utterances, max_segment_duration_sec):
        captured["max_segment_duration_sec"] = max_segment_duration_sec
        return []

    monkeypatch.setattr(main_module, "build_transcript", fake_build_transcript)

    args = main_module.build_parser().parse_args(["merge", "--output_dir", str(tmp_path), "--max-segment-duration-sec", "42"])
    args.func(args)

    assert captured["max_segment_duration_sec"] == 42


def test_cmd_speaker_name_uses_defaults(tmp_path, monkeypatch):
    write_model_list(tmp_path / "speaker_turns.json", [SpeakerTurn(start=0.0, end=1.0, speaker_id="Speaker_00")])
    event = VisualSpeakerEvent(start=0.0, end=1.0, display_name="Jane Roe", layout="gallery_view", signal="active_speaker_border", confidence=0.9)
    write_model(tmp_path / "visual_speaker_events.json", VisualSpeakerEventsFile(events=[event]))

    monkeypatch.setattr(
        main_module,
        "build_speaker_name_map",
        lambda turns, events, **kwargs: SpeakerNameMap(mapping={"Speaker_00": "Jane Roe"}),
    )

    args = main_module.build_parser().parse_args(["speaker-name", "--output_dir", str(tmp_path)])
    args.func(args)

    result = read_model(tmp_path / "speaker_name_map.json", SpeakerNameMap)
    assert result.mapping == {"Speaker_00": "Jane Roe"}


def test_cmd_fuse_falls_back_when_speaker_map_missing(tmp_path, monkeypatch):
    write_model(
        tmp_path / "transcript.json",
        TranscriptFile(segments=[TranscriptSegment(start_s=0.0, end_s=1.0, speaker_id="Speaker_00", text="hi")]),
    )

    captured = {}

    def fake_fuse(transcript_segments, frames, speaker_map):
        captured["speaker_map"] = speaker_map
        from meeting_intelligence.models import AlignedMeetingSegment, FullyAlignedTimeline

        return FullyAlignedTimeline(segments=[AlignedMeetingSegment(start=0.0, end=1.0, speaker="Speaker_00", transcript="hi")])

    monkeypatch.setattr(main_module, "fuse_timeline", fake_fuse)

    args = main_module.build_parser().parse_args(["fuse", "--output_dir", str(tmp_path)])
    args.func(args)

    assert captured["speaker_map"] is None  # no speaker_name_map.json present -> None, not an error


def test_cmd_llm_writes_meeting_intelligence(tmp_path, monkeypatch):
    from meeting_intelligence.models import AlignedMeetingSegment, FullyAlignedTimeline

    write_model(tmp_path / "fused_timeline.json", FullyAlignedTimeline(segments=[AlignedMeetingSegment(start=0.0, end=1.0, speaker="Presenter", transcript="hi")]))

    fake_mi = MeetingIntelligence(summary="ok", topics=[], action_items=[])

    class _FakeProcessor:
        def __init__(self, **kwargs):
            pass

        def enrich(self, timeline):
            return fake_mi

    monkeypatch.setattr(main_module, "FusionLLMProcessor", _FakeProcessor)

    args = main_module.build_parser().parse_args(["llm", "--output_dir", str(tmp_path)])
    args.func(args)

    result = read_model(tmp_path / "meeting_intelligence.json", MeetingIntelligence)
    assert result.summary == "ok"


def test_main_returns_nonzero_on_stage_failure(tmp_path, monkeypatch):
    def raise_error(video_path, output_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "run_ingestion", raise_error)
    exit_code = main_module.main(["ingest", "--video_path", "in.mp4", "--output_dir", str(tmp_path)])
    assert exit_code == 1


def test_main_returns_zero_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "run_ingestion",
        lambda video_path, output_dir: IngestionResult(video_path=str(video_path), wav_path=str(output_dir / "a.wav"), duration_sec=1.0),
    )
    exit_code = main_module.main(["ingest", "--video_path", "in.mp4", "--output_dir", str(tmp_path)])
    assert exit_code == 0
