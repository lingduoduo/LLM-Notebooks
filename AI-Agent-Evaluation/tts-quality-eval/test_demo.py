import pytest

import config
import demo


def test_select_configs_uses_defaults_and_optional_extra():
    assert demo.select_configs(None, False) == config.TTS_CONFIGS
    assert demo.select_configs(None, True) == config.TTS_CONFIGS + config.EXTRA_CONFIGS


def test_select_configs_preserves_requested_provider_order():
    configs = demo.select_configs("minimax, openai", False)
    assert [item.provider for item in configs] == ["minimax", "openai"]


def test_select_configs_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider 'missing'"):
        demo.select_configs("openai,missing", False)


def test_select_corpus_supports_custom_and_quick_modes():
    custom = demo.select_corpus("Hello world", quick=True)
    assert custom == [config.Sample("custom", "Hello world", "custom text", "neutral")]
    assert demo.select_corpus(None, quick=True) == config.CORPUS[:2]
    assert demo.select_corpus(None, quick=False) == config.CORPUS


def test_run_evaluations_preserves_grid_order_and_arguments(monkeypatch):
    configs = [config.TTSConfig("a", "m", "v"), config.TTSConfig("b", "m", "v")]
    corpus = [config.Sample("one", "One", "fixture"), config.Sample("two", "Two", "fixture")]
    calls = []

    def fake_evaluate(cfg, sample, use_gemini, fresh, judge_model):
        calls.append((cfg.name, sample.id, use_gemini, fresh, judge_model))
        return {"config": cfg.name, "sample": sample.id, "ok": True}

    monkeypatch.setattr(demo, "evaluate_one", fake_evaluate)
    monkeypatch.setattr(demo, "print_detail", lambda *_: None)

    records = demo.run_evaluations(configs, corpus, True, True, None)

    assert [(r["config"], r["sample"]) for r in records] == [
        ("a", "one"), ("a", "two"), ("b", "one"), ("b", "two")
    ]
    assert calls == [
        ("a", "one", True, True, None), ("a", "two", True, True, None),
        ("b", "one", True, True, None), ("b", "two", True, True, None),
    ]
