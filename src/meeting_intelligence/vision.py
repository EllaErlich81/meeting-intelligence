"""Part B: Computer Vision (Visual Track) orchestrator.

Chains Scene Detection -> OCR & Name Scan into a single `VisionTrackOutput`.
`scene_detection.py` and `ocr.py` are independently usable; this module
exists for convenience and for the `vision` CLI subcommand.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .models import VisionTrackOutput
from .ocr import run_ocr
from .scene_detection import detect_scenes

logger = logging.getLogger(__name__)


def run_vision_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 1.0,
    diff_threshold: float = 0.12,
    use_gpt4o_fallback: bool = True,
    openai_api_key: str | None = None,
    gpt4o_model: str = "gpt-4o",
) -> VisionTrackOutput:
    """Run Scene Detection -> OCR & Name Scan end to end."""
    scenes = detect_scenes(video_path, output_dir, sample_fps=sample_fps, diff_threshold=diff_threshold)
    return run_ocr(
        scenes,
        use_gpt4o_fallback=use_gpt4o_fallback,
        openai_api_key=openai_api_key,
        gpt4o_model=gpt4o_model,
    )


async def run_vision_pipeline_async(*args, **kwargs) -> VisionTrackOutput:
    """Async wrapper so the orchestrator can run Vision alongside Speech."""
    return await asyncio.to_thread(run_vision_pipeline, *args, **kwargs)
