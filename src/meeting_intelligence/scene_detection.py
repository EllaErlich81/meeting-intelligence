"""Part B / Scene Detection: sample frames at a fixed rate and flag slide
transitions.

Combines a whole-frame pixel difference with a regional (grid-cell) diff
mask, per the brief: whole-frame diff catches full slide changes cheaply,
while the regional max-cell diff catches localized changes (e.g. a
callout box or name tag appearing) that a whole-frame average could dilute
below threshold. Only frames flagged as a new scene are written to disk
and returned, keeping OCR's downstream workload proportional to the
number of *distinct* slides rather than every sampled frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .models import SceneFrame

logger = logging.getLogger(__name__)

REGIONAL_GRID_SIZE = 3
REGIONAL_THRESHOLD_MULTIPLIER = 1.5
DIFF_FRAME_SIZE = (320, 180)


def _grayscale_resize(frame: np.ndarray, size: tuple[int, int] = DIFF_FRAME_SIZE) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def whole_frame_diff(prev: np.ndarray, curr: np.ndarray) -> float:
    return float(np.mean(np.abs(curr.astype(np.int16) - prev.astype(np.int16))) / 255.0)


def regional_diff(prev: np.ndarray, curr: np.ndarray, grid_size: int = REGIONAL_GRID_SIZE) -> float:
    """Max mean-abs-diff across a `grid_size` x `grid_size` partition of the frame."""
    h, w = curr.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    max_cell_diff = 0.0
    for row in range(grid_size):
        y0 = row * cell_h
        y1 = h if row == grid_size - 1 else y0 + cell_h
        for col in range(grid_size):
            x0 = col * cell_w
            x1 = w if col == grid_size - 1 else x0 + cell_w
            cell_diff = whole_frame_diff(prev[y0:y1, x0:x1], curr[y0:y1, x0:x1])
            max_cell_diff = max(max_cell_diff, cell_diff)
    return max_cell_diff


def detect_scenes(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 1.0,
    diff_threshold: float = 0.12,
) -> list[SceneFrame]:
    """Sample `video_path` at `sample_fps` and return one `SceneFrame` per detected transition."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for scene detection: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(native_fps / sample_fps))

    scenes: list[SceneFrame] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0

    try:
        while True:
            if not cap.grab():
                break

            if frame_idx % frame_interval == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break

                timestamp = frame_idx / native_fps
                gray = _grayscale_resize(frame)

                is_new_scene = prev_gray is None or (
                    whole_frame_diff(prev_gray, gray) > diff_threshold
                    or regional_diff(prev_gray, gray) > diff_threshold * REGIONAL_THRESHOLD_MULTIPLIER
                )

                if is_new_scene:
                    frame_path = frames_dir / f"frame_{len(scenes) + 1:05d}.png"
                    cv2.imwrite(str(frame_path), frame)
                    scenes.append(SceneFrame(timestamp=timestamp, frame_path=str(frame_path)))
                    prev_gray = gray

            frame_idx += 1
    finally:
        cap.release()

    logger.info("Scene detection kept %d distinct frame(s) from %s", len(scenes), video_path)
    return scenes
