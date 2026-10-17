"""Provider tables must agree with each other, and Doubao is no longer offered."""
import pytest

import config
import demo
import pipeline


def test_every_representative_config_has_a_provider_and_synthesizer():
    for name, cfg in config.PROVIDER_CONFIGS.items():
        assert cfg.provider == name
        assert cfg.provider in config.PROVIDERS
        assert cfg.provider in pipeline._SYNTH_DISPATCH


def test_provider_tables_list_the_same_providers():
    assert set(config.PROVIDERS) == set(config.PROVIDER_CONFIGS) == set(pipeline._SYNTH_DISPATCH)


def test_doubao_is_not_offered():
    assert "doubao" not in config.PROVIDERS
    assert "doubao" not in config.PROVIDER_CONFIGS
    assert "doubao" not in pipeline._SYNTH_DISPATCH
    assert not hasattr(pipeline, "_synth_doubao")


def test_selecting_doubao_is_rejected():
    with pytest.raises(ValueError, match="unknown provider 'doubao'"):
        demo.select_configs("doubao", False)


def test_synthesize_rejects_doubao_before_any_network_call(tmp_path):
    cfg = config.TTSConfig("doubao-tts", model="volcano_tts", voice="v", provider="doubao")
    with pytest.raises(RuntimeError, match="Unknown provider: 'doubao'"):
        pipeline.synthesize(cfg, "hello", str(tmp_path / "out.mp3"))


@pytest.mark.parametrize(
    "provider, model, expected",
    [
        ("openai", "tts-1", True),
        ("openai", "gpt-4o-mini-tts", False),
        ("minimax", "speech-01-turbo", True),
        ("elevenlabs", "eleven_multilingual_v2", False),
        ("doubao", "volcano_tts", False),
    ],
)
def test_supports_speed(provider, model, expected):
    assert config.TTSConfig("x", model=model, voice="v", provider=provider).supports_speed() is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
