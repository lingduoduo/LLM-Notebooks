"""Provider tables must agree with each other, and Chinese hosted TTS providers
(Doubao, MiniMax) are no longer offered."""
import pytest

import config
import demo
import pipeline

REMOVED_PROVIDERS = ("doubao", "minimax")


def test_every_representative_config_has_a_provider_and_synthesizer():
    for name, cfg in config.PROVIDER_CONFIGS.items():
        assert cfg.provider == name
        assert cfg.provider in config.PROVIDERS
        assert cfg.provider in pipeline._SYNTH_DISPATCH


def test_provider_tables_list_the_same_providers():
    assert set(config.PROVIDERS) == set(config.PROVIDER_CONFIGS) == set(pipeline._SYNTH_DISPATCH)


def test_remaining_providers_are_exactly_openai_elevenlabs_fishaudio():
    assert set(config.PROVIDERS) == {"openai", "elevenlabs", "fishaudio"}


@pytest.mark.parametrize("provider", REMOVED_PROVIDERS)
def test_removed_provider_is_not_offered(provider):
    assert provider not in config.PROVIDERS
    assert provider not in config.PROVIDER_CONFIGS
    assert provider not in pipeline._SYNTH_DISPATCH
    assert not hasattr(pipeline, f"_synth_{provider}")


def test_no_provider_requires_minimax_credentials():
    envs = {e for p in config.PROVIDERS.values() for e in p.env}
    assert not any(e.startswith("MINIMAX") for e in envs)


@pytest.mark.parametrize("provider", REMOVED_PROVIDERS)
def test_selecting_removed_provider_is_rejected(provider):
    with pytest.raises(ValueError, match=f"unknown provider '{provider}'"):
        demo.select_configs(provider, False)


@pytest.mark.parametrize(
    "provider, model",
    [("doubao", "volcano_tts"), ("minimax", "speech-01-turbo")],
)
def test_synthesize_rejects_removed_provider_before_any_network_call(tmp_path, provider, model):
    cfg = config.TTSConfig(f"{provider}-tts", model=model, voice="v", provider=provider)
    with pytest.raises(RuntimeError, match=f"Unknown provider: '{provider}'"):
        pipeline.synthesize(cfg, "hello", str(tmp_path / "out.mp3"))


@pytest.mark.parametrize(
    "provider, model, expected",
    [
        ("openai", "tts-1", True),
        ("openai", "gpt-4o-mini-tts", False),
        ("elevenlabs", "eleven_multilingual_v2", False),
        ("fishaudio", "speech-1.5", False),
        ("minimax", "speech-01-turbo", False),
        ("doubao", "volcano_tts", False),
    ],
)
def test_supports_speed(provider, model, expected):
    assert config.TTSConfig("x", model=model, voice="v", provider=provider).supports_speed() is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
