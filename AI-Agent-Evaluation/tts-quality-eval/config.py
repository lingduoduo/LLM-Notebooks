"""Experiment 6-5: configuration and evaluation corpus for the automated TTS quality pipeline.

This module centralizes:
  - OpenAI model names and reference prices for rough cost estimates;
  - TTS configurations, each a model/voice/speed combination to compare; and
  - challenging reference samples covering numbers, heteronyms, long sentences,
    and proper nouns with emotion.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model names (all use OPENAI_API_KEY).
# ---------------------------------------------------------------------------
WHISPER_MODEL = "whisper-1"        # Speech transcription for WER and word accuracy; requires direct OpenAI access.
JUDGE_MODEL = "gpt-5.6-luna"       # LLM rubric judge; chat requests can fall back to OpenRouter.

# Optional Gemini audio judge used in the accompanying book. The default is the
# current low-cost flagship, gemini-3.5-flash, verified to accept audio input
# and listen directly to synthesized speech. Model names can expire, so runtime
# discovery through REST /models corrects them. Used only with --gemini.
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash"

# Reference prices in US dollars. They only support rough printed cost estimates
# and do not affect scoring; official values may change.
PRICE: dict[str, float] = {
    "tts-1": 15.0 / 1_000_000,          # $ / character
    "tts-1-hd": 30.0 / 1_000_000,       # $ / character
    "gpt-4o-mini-tts": 12.0 / 1_000_000,
    "whisper-1": 0.006 / 60,            # $ / second
}


@dataclass
class TTSConfig:
    """One TTS configuration to evaluate; names must be unique in the table.

    ``provider`` selects the synthesis service (openai, elevenlabs, fishaudio,
    minimax, or doubao). Each provider defines the meaning of model, voice, and
    speed. For example, ElevenLabs voice is a voice_id, while Fish Audio voice
    is a reference_id and may be empty to use the default voice.
    """

    name: str
    model: str
    voice: str
    speed: float = 1.0
    provider: str = "openai"

    def supports_speed(self) -> bool:
        # Only some provider/model combinations accept a speed parameter.
        if self.provider == "openai":
            return self.model in ("tts-1", "tts-1-hd")
        return self.provider in ("minimax", "doubao")


# ---------------------------------------------------------------------------
# Multi-provider registry, corresponding to the book's integration of OpenAI,
# ElevenLabs, Fish Audio, Minimax, and Doubao. Each provider declares needed
# environment variables and a representative configuration for comparison.
# Other providers use their public REST APIs; a missing key marks only that row
# as failed and does not stop the full table (see demo.py).
# ---------------------------------------------------------------------------
# Environment aliases let one credential be supplied under a historical or
# conventional name; any non-empty alias counts as configured.
ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "FISH_API_KEY": ("FISH_API_KEY", "FISHAUDIO_API_KEY"),
}


def env_get(name: str) -> str:
    """Return the first stripped, non-empty environment value for ``name`` or its aliases."""
    import os
    for n in ENV_ALIASES.get(name, (name,)):
        val = os.environ.get(n, "").strip()
        if val:
            return val
    return ""


@dataclass
class ProviderInfo:
    key: str                # Internal identifier for --providers.
    label: str              # Display name.
    env: tuple[str, ...]    # Environment variable names required for synthesis.
    note: str               # Brief description of voice semantics.

    def configured(self) -> bool:
        return all(env_get(e) for e in self.env)


PROVIDERS: dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        "openai", "OpenAI", ("OPENAI_API_KEY",),
        "voice=alloy/nova/…, model=tts-1/tts-1-hd/gpt-4o-mini-tts; the only provider validated end-to-end in this repository.",
    ),
    "elevenlabs": ProviderInfo(
        "elevenlabs", "ElevenLabs", ("ELEVENLABS_API_KEY",),
        "voice=voice_id; model defaults to eleven_multilingual_v2.",
    ),
    "fishaudio": ProviderInfo(
        "fishaudio", "Fish Audio", ("FISH_API_KEY",),
        "voice=reference_id (empty uses the default voice), using /v1/tts; FISHAUDIO_API_KEY is also accepted.",
    ),
    "minimax": ProviderInfo(
        "minimax", "Minimax", ("MINIMAX_API_KEY", "MINIMAX_GROUP_ID"),
        "voice=voice_id; model defaults to speech-01-turbo and requires an additional GroupId.",
    ),
    "doubao": ProviderInfo(
        "doubao", "Doubao (Volcengine)", ("DOUBAO_APP_ID", "DOUBAO_ACCESS_TOKEN"),
        "voice=voice_type, using openspeech.bytedance.com; the authorization header is 'Bearer;{token}'.",
    ),
}

# Representative configurations selected by --providers, one per provider.
# Non-OpenAI model and voice values are common defaults and may be adjusted to
# voices available on the account.
PROVIDER_CONFIGS: dict[str, TTSConfig] = {
    "openai": TTSConfig("openai-alloy", provider="openai", model="tts-1", voice="alloy"),
    "elevenlabs": TTSConfig("elevenlabs-multi", provider="elevenlabs",
                            model="eleven_multilingual_v2", voice="21m00Tcm4TlvDq8ikWAM"),
    "fishaudio": TTSConfig("fishaudio-default", provider="fishaudio",
                          model="speech-1.5", voice=""),
    "minimax": TTSConfig("minimax-turbo", provider="minimax",
                        model="speech-01-turbo", voice="male-qn-qingse"),
    "doubao": TTSConfig("doubao-tts", provider="doubao",
                       model="volcano_tts", voice="zh_female_qingxin"),
}


# Default comparison set. It varies model (tts-1 versus tts-1-hd), voice, and
# speed to make accuracy and naturalness differences easy to observe. All use
# OpenAI by default so the evaluator runs without extra provider configuration.
TTS_CONFIGS: list[TTSConfig] = [
    TTSConfig("tts1-alloy-1.0", model="tts-1", voice="alloy", speed=1.0),
    TTSConfig("tts1hd-alloy-1.0", model="tts-1-hd", voice="alloy", speed=1.0),
    TTSConfig("tts1-nova-1.0", model="tts-1", voice="nova", speed=1.0),
    TTSConfig("tts1-alloy-1.5", model="tts-1", voice="alloy", speed=1.5),
]

# Optional configuration enabled by --extra. It is excluded by default so the
# standard setup always runs.
EXTRA_CONFIGS: list[TTSConfig] = [
    TTSConfig("4omini-nova-1.0", model="gpt-4o-mini-tts", voice="nova", speed=1.0),
]


@dataclass
class Sample:
    """Reference text plus an expected emotion label for rubric context."""

    id: str
    text: str
    challenge: str      # Main challenge exercised by this sample.
    emotion: str = "neutral"


# Diverse English test corpus: numbers, heteronyms, a long sentence, and
# proper nouns with emotion.
CORPUS: list[Sample] = [
    Sample("num", "In the third quarter of 2026, revenue grew 37.5 percent, up 12 percentage points year over year.", "numbers, percentages, and dates", "neutral"),
    Sample("pronunciation", "The bass player caught a bass near the lead mine after he read the latest report.", "heteronyms and pronunciation-sensitive words", "neutral"),
    Sample("long", "According to the report, as artificial intelligence advances rapidly, more companies are applying large language models to customer service, content creation, and data analysis, significantly improving operational efficiency.", "long sentence and news style", "neutral"),
    Sample("emotion", "Fantastic! OpenAI's newly released model achieved an amazing result on the GAIA benchmark!", "proper nouns and excited delivery", "excited"),
]
