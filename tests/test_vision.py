"""Orchestration test: Scene Detection -> OCR & Name Scan wiring."""

from __future__ import annotations

import meeting_intelligence.vision as vision_module
from meeting_intelligence.models import SceneFrame, SlideContent, VisionTrackOutput, VisualFrameContext


def test_run_vision_pipeline_wires_scene_detection_into_ocr(monkeypatch, tmp_path):
    calls = []

    def fake_detect_scenes(video_path, output_dir, sample_fps, diff_threshold):
        calls.append(("detect_scenes", sample_fps, diff_threshold))
        return [SceneFrame(timestamp=0.0, frame_path="frame_0.png")]

    def fake_run_ocr(scenes, use_gpt4o_fallback, openai_api_key, gpt4o_model):
        calls.append(("run_ocr", len(scenes), use_gpt4o_fallback))
        frame = VisualFrameContext(slide_id="slide_000000", start_time=0.0, frame_path="f.png", content=SlideContent(title="T", raw_text="T"))
        return VisionTrackOutput(frames=[frame])

    monkeypatch.setattr(vision_module, "detect_scenes", fake_detect_scenes)
    monkeypatch.setattr(vision_module, "run_ocr", fake_run_ocr)

    output = vision_module.run_vision_pipeline(
        "video.mp4", tmp_path, sample_fps=2.0, diff_threshold=0.2, use_gpt4o_fallback=False
    )

    assert calls[0] == ("detect_scenes", 2.0, 0.2)
    assert calls[1] == ("run_ocr", 1, False)
    assert output.frames[0].content.title == "T"
