"""Zoom-specific layout heuristics for the OCR & Name Scan stage.

Primary detector for `detect_display_name` (see `ocr.py`'s module
docstring for the full detector priority order): tried *before* the
generic whole-frame name-shape scan or GPT-4o vision, since it targets
Zoom's two predictable speaker-identifying UI spots directly by color and
fixed position -- more trustworthy than pattern-matching arbitrary slide
text whenever that UI is actually present, at the cost of per-deployment
calibration:

1. **Gallery/speaker view**: Zoom draws a colored border around the tile
   of whoever is currently talking. `find_active_speaker_tile` looks for
   that border (green by default -- measured off real footage; Zoom's
   exact highlight color has varied by version/theme, so it's tunable)
   and returns its bounding box; the caller then crops the bottom-*left*
   strip of that tile -- where Zoom actually overlays the participant's
   name pill, not the full
   width, which can otherwise include mute/video-status icons or a
   participant-count badge at the bottom-right -- and OCRs just that crop.
2. **Presentation/screen-share mode**: Zoom overlays a small "who's
   talking" badge (video thumbnail + name) in the top-right corner of the
   shared screen. `presentation_badge_bbox` returns a fixed crop region for
   the whole badge; `crop_badge_name_label` then narrows that down further
   to just the bottom strip, where the name sits below the thumbnail.

All thresholds here are heuristics tuned on typical Zoom recordings, not
guarantees -- they are exposed as parameters (and, via `config.Settings`,
as env vars / CLI flags) specifically so they can be recalibrated against
real footage without touching code.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel


class ZoomLayoutSettings(BaseModel):
    """Tunable parameters for Zoom-specific speaker-identification heuristics."""

    enable_active_speaker_border: bool = True
    enable_presentation_badge: bool = True

    # HSV bounds for the active-speaker border color (default: green --
    # measured off real Zoom footage at hue ~42-50; widened to 35-65 for
    # margin. Recalibrate against your own recordings if Zoom's theme/
    # version highlights a different color, e.g. the older yellow border
    # some versions use, roughly hue 20-35).
    border_hue_min: int = 35
    border_hue_max: int = 65
    border_sat_min: int = 100
    border_val_min: int = 100

    # A qualifying tile's bounding box must cover between these fractions
    # of the frame's area, to reject small color-noise blobs and reject a
    # near-full-frame false match.
    border_min_area_fraction: float = 0.03
    border_max_area_fraction: float = 0.6

    # Morphological closing kernel (pixels) applied to the border color
    # mask before contour detection, to bridge small gaps a thin border
    # outline can develop from anti-aliasing or video compression --
    # without this, a broken outline can fail to form one contour large
    # enough to pass the area-fraction check above. 0 disables closing.
    border_close_kernel_size: int = 5

    # Fraction of the detected tile's height/width, from the bottom-left
    # corner, that's cropped and OCR'd for the participant's name overlay.
    # Zoom overlays the name as a pill anchored bottom-left of the tile,
    # not spanning its full width.
    name_label_height_fraction: float = 0.18
    name_label_width_fraction: float = 0.65

    # Fixed crop region (top-right corner) for the presentation-mode
    # "who's talking" badge, as a fraction of the frame's width/height.
    badge_top_fraction: float = 0.22
    badge_right_fraction: float = 0.22

    # Within that badge, the fraction of its height (from the bottom)
    # where the name sits below the speaker's video thumbnail.
    badge_name_height_fraction: float = 0.4


def find_active_speaker_tile(
    frame: np.ndarray,
    settings: ZoomLayoutSettings | None = None,
) -> tuple[int, int, int, int] | None:
    """Return the (x, y, w, h) bounding box of the active-speaker-bordered tile, if any."""
    settings = settings or ZoomLayoutSettings()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([settings.border_hue_min, settings.border_sat_min, settings.border_val_min])
    upper = np.array([settings.border_hue_max, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    if settings.border_close_kernel_size > 0:
        kernel = np.ones((settings.border_close_kernel_size, settings.border_close_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_fraction = (w * h) / frame_area
        if settings.border_min_area_fraction <= area_fraction <= settings.border_max_area_fraction:
            candidates.append((w * h, (x, y, w, h)))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def crop_name_label(
    frame: np.ndarray,
    tile_bbox: tuple[int, int, int, int],
    settings: ZoomLayoutSettings | None = None,
) -> np.ndarray:
    """Crop the bottom-left corner of a speaker tile, where Zoom overlays the participant's name pill."""
    settings = settings or ZoomLayoutSettings()
    x, y, w, h = tile_bbox
    label_h = max(1, int(h * settings.name_label_height_fraction))
    label_w = max(1, int(w * settings.name_label_width_fraction))
    y0 = max(0, y + h - label_h)
    return frame[y0 : y + h, x : x + label_w]


def presentation_badge_bbox(
    frame_shape: tuple[int, ...],
    settings: ZoomLayoutSettings | None = None,
) -> tuple[int, int, int, int]:
    """Fixed top-right crop region for the presentation-mode speaker badge."""
    settings = settings or ZoomLayoutSettings()
    height, width = frame_shape[0], frame_shape[1]
    w = int(width * settings.badge_right_fraction)
    h = int(height * settings.badge_top_fraction)
    return (width - w, 0, w, h)


def crop_badge_name_label(
    frame: np.ndarray,
    badge_bbox: tuple[int, int, int, int],
    settings: ZoomLayoutSettings | None = None,
) -> np.ndarray:
    """Narrow the presentation badge crop to just its bottom strip, where the name sits below the video thumbnail."""
    settings = settings or ZoomLayoutSettings()
    x, y, w, h = badge_bbox
    label_h = max(1, int(h * settings.badge_name_height_fraction))
    y0 = y + h - label_h
    return frame[y0 : y + h, x : x + w]


def crop_region(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w]
