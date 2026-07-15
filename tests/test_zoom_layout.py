"""Zoom-specific layout heuristics are pure OpenCV/numpy math -- no OCR
engine or model involved -- so these run against real, synthetically
constructed frames."""

from __future__ import annotations

import cv2
import numpy as np

from meeting_intelligence.zoom_layout import (
    ZoomLayoutSettings,
    crop_badge_name_label,
    crop_name_label,
    crop_region,
    find_active_speaker_tile,
    presentation_badge_bbox,
)


def _frame_with_green_border(size=(200, 200), rect=(10, 10, 90, 180)) -> np.ndarray:
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)  # BGR green
    return frame


def test_find_active_speaker_tile_detects_green_border():
    frame = _frame_with_green_border()
    bbox = find_active_speaker_tile(frame, ZoomLayoutSettings())

    assert bbox is not None
    x, y, w, h = bbox
    # Bounding box should roughly match the drawn rectangle (allow a couple
    # pixels of slack for the border's line thickness).
    assert abs(x - 10) <= 4
    assert abs(y - 10) <= 4
    assert abs(w - 90) <= 8
    assert abs(h - 180) <= 8


def test_find_active_speaker_tile_returns_none_with_no_green():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert find_active_speaker_tile(frame, ZoomLayoutSettings()) is None


def test_find_active_speaker_tile_ignores_tiny_noise_blob():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (5, 5), (8, 8), (0, 255, 0), thickness=-1)  # ~3x3px speck
    assert find_active_speaker_tile(frame, ZoomLayoutSettings()) is None


def test_find_active_speaker_tile_respects_custom_hue_range():
    # Yellow (BGR 0,255,255) is outside the default green hue range.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (100, 190), (0, 255, 255), thickness=4)
    assert find_active_speaker_tile(frame, ZoomLayoutSettings()) is None

    yellow_settings = ZoomLayoutSettings(border_hue_min=20, border_hue_max=35)
    bbox = find_active_speaker_tile(frame, yellow_settings)
    assert bbox is not None


def test_find_active_speaker_tile_bridges_broken_border_with_closing():
    """A border outline broken into disconnected dashes (anti-aliasing /
    compression artifacts) should still be found as one full-size tile
    once the morphological-closing gap is bridged."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (100, 190), (0, 255, 0), thickness=4)
    # Punch small gaps into the border to simulate a broken outline.
    frame[10:14, 40:44] = 0
    frame[10:14, 70:74] = 0

    settings = ZoomLayoutSettings(border_close_kernel_size=7)
    bbox = find_active_speaker_tile(frame, settings)

    assert bbox is not None
    x, y, w, h = bbox
    assert abs(w - 90) <= 8
    assert abs(h - 180) <= 8


def test_crop_name_label_takes_bottom_left_corner_of_tile():
    frame = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
    tile_bbox = (10, 10, 90, 180)
    settings = ZoomLayoutSettings(name_label_height_fraction=0.2, name_label_width_fraction=0.5)

    crop = crop_name_label(frame, tile_bbox, settings)

    expected_label_h = int(180 * 0.2)
    expected_label_w = int(90 * 0.5)
    assert crop.shape == (expected_label_h, expected_label_w, 3)
    np.testing.assert_array_equal(crop, frame[190 - expected_label_h : 190, 10 : 10 + expected_label_w])


def test_presentation_badge_bbox_is_top_right_corner():
    settings = ZoomLayoutSettings(badge_top_fraction=0.25, badge_right_fraction=0.1)
    bbox = presentation_badge_bbox((400, 1000), settings)

    x, y, w, h = bbox
    assert w == 100  # 10% of width
    assert h == 100  # 25% of height
    assert x == 900  # flush with the right edge
    assert y == 0  # flush with the top edge


def test_crop_badge_name_label_takes_bottom_strip_of_badge():
    frame = np.arange(400 * 1000 * 3, dtype=np.uint8).reshape(400, 1000, 3)
    settings = ZoomLayoutSettings(badge_top_fraction=0.25, badge_right_fraction=0.1, badge_name_height_fraction=0.4)
    badge_bbox = presentation_badge_bbox((400, 1000), settings)

    crop = crop_badge_name_label(frame, badge_bbox, settings)

    x, y, w, h = badge_bbox
    expected_label_h = int(h * 0.4)
    assert crop.shape == (expected_label_h, w, 3)
    np.testing.assert_array_equal(crop, frame[y + h - expected_label_h : y + h, x : x + w])


def test_crop_region_extracts_expected_slice():
    frame = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    crop = crop_region(frame, (2, 3, 5, 4))
    assert crop.shape == (4, 5, 3)
    np.testing.assert_array_equal(crop, frame[3:7, 2:7])
