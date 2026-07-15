from __future__ import annotations

from meeting_intelligence.config import LLMProvider, Settings, get_settings


def test_defaults_when_env_empty(monkeypatch):
    for var in [
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "WHISPER_MODEL_SIZE",
        "WHISPER_DEVICE",
        "USE_OPENAI_WHISPER_API",
        "HUGGINGFACE_TOKEN",
        "SCENE_SAMPLE_FPS",
        "SCENE_DIFF_THRESHOLD",
        "OCR_GPT4O_FALLBACK",
    ]:
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)
    assert settings.llm_provider == LLMProvider.OPENAI
    assert settings.whisper_model_size == "large-v3"
    assert settings.scene_sample_fps == 2.0
    assert settings.scene_diff_threshold == 0.06
    assert settings.ocr_gpt4o_fallback is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "large-v3")
    monkeypatch.setenv("SCENE_SAMPLE_FPS", "3.0")

    settings = get_settings()
    assert settings.llm_provider == LLMProvider.GEMINI
    assert settings.whisper_model_size == "large-v3"
    assert settings.scene_sample_fps == 3.0
