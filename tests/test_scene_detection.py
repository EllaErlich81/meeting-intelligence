"""Scene detection runs against real ffmpeg-generated video + real OpenCV
(no ML model involved), so this exercises the actual diff algorithm."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import requires_ffmpeg
from meeting_intelligence.scene_detection import detect_scenes, regional_diff, whole_frame_diff


def test_whole_frame_diff_zero_for_identical_frames():
    frame = np.full((10, 10), 128, dtype=np.uint8)
    assert whole_frame_diff(frame, frame) == 0.0


def test_whole_frame_diff_positive_for_different_frames():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.full((10, 10), 255, dtype=np.uint8)
    assert whole_frame_diff(a, b) == 1.0


def test_regional_diff_detects_localized_change_whole_frame_diff_misses():
    a = np.zeros((30, 30), dtype=np.uint8)
    b = a.copy()
    b[0:10, 0:10] = 255  # change only one grid cell out of 9

    whole = whole_frame_diff(a, b)
    regional = regional_diff(a, b, grid_size=3)

    assert regional > whole  # localized change is diluted in the whole-frame average


@requires_ffmpeg
def test_detect_scenes_flags_red_to_blue_transition(tmp_path, sample_video):
    scenes = detect_scenes(sample_video, tmp_path, sample_fps=1.0, diff_threshold=0.12)

    # 2s video, 1fps sampling -> at least the initial frame plus the red->blue transition.
    assert len(scenes) >= 2
    for scene in scenes:
        assert Path(scene.frame_path).is_file()


@requires_ffmpeg
def test_detect_scenes_first_frame_always_kept(tmp_path, sample_video):
    scenes = detect_scenes(sample_video, tmp_path, sample_fps=1.0)
    assert scenes[0].timestamp == 0.0
