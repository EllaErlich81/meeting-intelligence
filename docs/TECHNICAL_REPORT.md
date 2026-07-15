# Multimodal Meeting Intelligence Pipeline

**Technical Report**

**Author:** Ella Erlich

---

# 1. Introduction

This project implements an end-to-end **Meeting Intelligence Pipeline** that processes a recorded Zoom meeting and produces structured meeting insights by combining speech processing, computer vision, and large language models.

The objective was not to maximize the accuracy of any individual model, but to design a modular, production-oriented system that demonstrates sound engineering judgment, effective model integration, and clear reasoning under realistic time constraints.

The pipeline transforms a meeting recording into a structured JSON containing:

- Speaker-attributed transcript
- Slide-aware visual context
- Speaker identification
- Meeting summary
- Key discussion topics
- Action items

A major design goal was **modularity**. Each stage exposes a well-defined JSON interface, allowing components to be replaced or evaluated independently without affecting the rest of the pipeline.

---

# 2. System Architecture

The system consists of four independent stages connected through explicit data contracts.

![Pipeline architecture](architecture.png)

The architecture follows a production-style pipeline in which each component is responsible for a single task and communicates only through structured artifacts.

This separation simplifies debugging, enables partial re-execution, and makes it straightforward to replace individual models without changing downstream stages.

---

# 3. Design Decisions

Several design decisions were made to balance accuracy, robustness, simplicity, and computational cost.

## Speech Processing

Speech processing combines **Silero VAD**, **Pyannote diarization**, and **Faster-Whisper transcription**.

Silero VAD removes silence before transcription, reducing unnecessary computation.

Pyannote was selected because it provides one of the strongest open-source diarization solutions while integrating naturally into the pipeline.

Faster-Whisper was chosen as the default ASR engine because it offers high transcription quality while running locally, avoiding API costs and keeping audio on-device.

---

## Visual Context

Rather than processing every video frame, the pipeline first detects slide changes and performs OCR only on representative frames.

This significantly reduces computation while preserving nearly all information that is useful for downstream meeting understanding.

PaddleOCR performs the primary OCR locally, with GPT-4o Vision available as an optional fallback when OCR confidence is low.

---

## Speaker Identification

Instead of face recognition, the system identifies speakers using Zoom's own user interface elements, including active-speaker borders and presentation badges.

This approach offers two advantages:

- Avoids biometric processing and associated privacy concerns.
- Relies on interface elements that are more stable than facial appearance under different lighting or camera conditions.

Visual speaker evidence is combined with diarization through temporal alignment, producing confidence-weighted speaker identities.

---

## LLM-Based Meeting Intelligence

The final stage combines transcript segments, speaker information, and slide context into a shared timeline before sending them to the LLM.

Structured JSON output is enforced through schema validation, improving consistency and reducing hallucinations.

Grounding each summary item with timestamps, speakers, and visual context increases traceability and allows users to verify generated insights.

---

# 4. Model Selection

| Component | Selected Model | Rationale |
|------------|----------------|-----------|
| Voice Activity Detection | Silero VAD | Lightweight, efficient speech segmentation |
| Speaker Diarization | pyannote.audio | Strong open-source diarization performance |
| Speech Recognition | Faster-Whisper | High-quality multilingual ASR with local execution |
| OCR | PaddleOCR | Accurate OCR for presentation slides with local inference |
| Vision Fallback | GPT-4o Vision | Used only for difficult OCR cases |
| LLM | GPT-4o / Gemini Flash | Structured JSON generation and meeting summarization |

The overall philosophy was to prefer local inference whenever possible while allowing optional hosted models when they provide clear value.

---

# 5. Evaluation

Evaluation focused on validating the complete end-to-end pipeline rather than optimizing individual models in isolation.

| Component | Evaluation Method |
|------------|-------------------|
| ASR | Manual transcript inspection |
| Speaker Attribution | Manual verification against the recording |
| OCR | Text completeness on representative slides |
| Speaker Identification | Correctness of assigned display names |
| LLM Output | Human evaluation of summary quality and grounding |

Testing on the provided recording revealed several practical issues that were addressed during development:

- Incorrect Zoom border calibration reduced speaker detection accuracy.
- Very small OCR crops required adaptive upscaling.
- Inconsistent OCR spelling required fuzzy name canonicalization.
- Long transcript segments were split to better match LLM context windows.

These observations reinforced that integration quality and empirical validation are often more important than maximizing the performance of individual models.

### Failure Analysis

Despite the improvements, several limitations remain:

- Speaker identification depends on Zoom interface visibility.
- OCR quality decreases on very small or blurred labels.
- ASR errors can propagate into downstream LLM reasoning.
- Hallucination risk increases when transcript quality degrades.

These limitations are documented explicitly to provide a realistic assessment of the system.

---

# 6. Production Considerations

The pipeline was designed with production deployment in mind.

## Scalability

Each stage operates independently, allowing audio and video processing to run in parallel or be distributed across multiple workers.

## Cost

Most processing is performed locally. External APIs are invoked only when confidence-based heuristics indicate that additional quality justifies the cost.

## Reliability

Graceful degradation is supported throughout the pipeline. If an optional stage fails, downstream components continue using the available information rather than terminating the entire workflow.

## Monitoring

Every stage generates structured artifacts and logging information, enabling debugging, quality monitoring, and regression analysis.

## Extensibility

Because all interfaces are defined through Pydantic models, replacing an ASR model, OCR engine, or LLM requires minimal changes to the remaining pipeline.

---

# 7. Future Work

Potential improvements include:

- Improve handling of ASR repetition and hallucination artifacts.
- Support streaming inference for real-time meetings.
- Introduce dependency-aware artifact caching.
- Improve speaker identification under limited visual evidence.
- Enhance OCR using slide-specific layout understanding.
- Expand evaluation using manually annotated benchmark recordings.

---

# 8. AI Assistance

AI assistants (Claude Code and ChatGPT) were used throughout development for brainstorming, implementation support, prompt refinement, and code review.

Final architectural decisions, model selection, evaluation methodology, debugging, and engineering trade-offs were made by the author after validating the system on the provided recording.

---

# 9. Conclusions

This project demonstrates that building a practical Meeting Intelligence system is primarily an **integration challenge** rather than a modeling challenge.

Modern ASR, OCR, computer vision, and LLM models already provide strong individual capabilities. The primary engineering challenge is orchestrating these components into a reliable pipeline with explicit interfaces, confidence-aware fusion, and systematic evaluation.

The resulting system emphasizes modularity, explainability, and production-oriented design while remaining flexible enough to incorporate future improvements and alternative models.
