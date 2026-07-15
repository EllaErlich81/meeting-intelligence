"""Runtime configuration for every pipeline stage.

All settings are environment-variable driven (see `.env.example`) so that
each stage can be invoked independently -- via the CLI or as a library --
without needing to thread configuration objects through call sites by hand.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .zoom_layout import ZoomLayoutSettings


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- General ----
    log_level: str = "INFO"

    # ---- Speech (Part A) ----
    whisper_model_size: str = "large-v3"
    whisper_device: str = "cpu"
    use_openai_whisper_api: bool = False
    huggingface_token: str | None = None
    # Caps how long (in seconds) a single Transcript merged-same-speaker
    # segment can span before a new segment is started, even mid-speaker
    # -- see transcript.build_transcript's docstring.
    transcript_max_segment_duration_sec: float = Field(default=120.0, gt=0)

    # ---- Vision (Part B) ----
    scene_sample_fps: float = Field(default=2.0, gt=0) #default=1.0
    scene_diff_threshold: float = Field(default=0.06, ge=0, le=1) #default=0.12
    ocr_gpt4o_fallback: bool = True

    # ---- Vision (Part B) / Zoom speaker identification ----
    # Gallery view: Zoom draws a colored border around the active speaker's
    # tile. Default color is green (measured off real footage, hue ~42-50,
    # widened to 35-65 for margin); recalibrate against your own footage if
    # your Zoom theme/version highlights a different color, e.g. the older
    # yellow border some versions use (roughly hue 20-35).
    zoom_active_speaker_detection: bool = True
    zoom_border_hue_min: int = Field(default=35, ge=0, le=179)
    zoom_border_hue_max: int = Field(default=65, ge=0, le=179)
    zoom_border_sat_min: int = Field(default=100, ge=0, le=255)
    zoom_border_val_min: int = Field(default=100, ge=0, le=255)
    # Bridges gaps in a broken/anti-aliased border outline before contour
    # detection; 0 disables. See zoom_layout.ZoomLayoutSettings.
    zoom_border_close_kernel_size: int = Field(default=5, ge=0)
    # Bottom-left corner of the tile, where Zoom anchors the name pill
    # (not the full width/height, which can include status icons).
    zoom_name_label_height_fraction: float = Field(default=0.18, gt=0, le=1)
    zoom_name_label_width_fraction: float = Field(default=0.65, gt=0, le=1)
    # Presentation/screen-share mode: fixed top-right speaker badge crop.
    zoom_presentation_badge_detection: bool = True
    zoom_badge_top_fraction: float = Field(default=0.22, gt=0, le=1)
    zoom_badge_right_fraction: float = Field(default=0.22, gt=0, le=1)
    # Within that badge, the bottom fraction where the name sits below the
    # speaker's video thumbnail.
    zoom_badge_name_height_fraction: float = Field(default=0.4, gt=0, le=1)

    # ---- LLM enrichment (Part C) ----
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"

    # ---- Speaker Naming (Part C) ----
    # A visual speaker event only resolves to a speaker if the share of its
    # name's total temporal-overlap-weighted evidence pointing at that one
    # speaker clears this confidence threshold. See
    # speaker_naming.build_speaker_name_map's docstring.
    speaker_naming_min_confidence: float = Field(default=0.6, ge=0, le=1)
    # OCR-noise name variants (e.g. "Maria Alvarez" / "Marla Alvarez")
    # within this string-similarity ratio of each other are treated as the
    # same person before alignment. See speaker_naming._canonicalize_names's
    # docstring for how the default was picked.
    speaker_naming_name_similarity_threshold: float = Field(default=0.65, ge=0, le=1)


    def zoom_layout_settings(self) -> ZoomLayoutSettings:
        return ZoomLayoutSettings(
            enable_active_speaker_border=self.zoom_active_speaker_detection,
            enable_presentation_badge=self.zoom_presentation_badge_detection,
            border_hue_min=self.zoom_border_hue_min,
            border_hue_max=self.zoom_border_hue_max,
            border_sat_min=self.zoom_border_sat_min,
            border_val_min=self.zoom_border_val_min,
            border_close_kernel_size=self.zoom_border_close_kernel_size,
            name_label_height_fraction=self.zoom_name_label_height_fraction,
            name_label_width_fraction=self.zoom_name_label_width_fraction,
            badge_top_fraction=self.zoom_badge_top_fraction,
            badge_right_fraction=self.zoom_badge_right_fraction,
            badge_name_height_fraction=self.zoom_badge_name_height_fraction,
        )


def get_settings() -> Settings:
    """Return a fresh Settings instance, re-reading the environment.

    Deliberately not cached: tests and independently-run CLI stages
    frequently mutate `os.environ` between calls and expect it to take
    effect immediately.
    """
    return Settings()
