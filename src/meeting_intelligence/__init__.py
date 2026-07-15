"""Multimodal Meeting Intelligence Pipeline.

Turns a raw meeting recording (.mp4) into structured conversation
intelligence (summary, topics, action items) by fusing a speech track
(VAD -> diarization -> ASR) with a visual track (scene detection -> OCR)
and enriching the resulting timeline with an LLM.

See docs/architecture.md and README.md for the stage-by-stage design.
"""

__version__ = "0.1.0"
