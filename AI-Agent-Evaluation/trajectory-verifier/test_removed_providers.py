"""Chinese hosted-model backends (Moonshot/Kimi, ByteDance Ark/Doubao) are no longer offered."""
import sys

import pytest

import evidence_client
import run_experiment_9_1

REMOVED = ("moonshot", "ark")


def test_backends_are_openai_and_openrouter_only():
    assert set(evidence_client.BACKENDS) == {"openai", "openrouter"}


def test_openai_backend_uses_official_api():
    key_env, base_url, _model = evidence_client.BACKENDS["openai"]
    assert key_env == "OPENAI_API_KEY"
    assert base_url == "https://api.openai.com/v1"


def test_no_backend_points_at_a_removed_host_or_model():
    for key_env, base_url, model in evidence_client.BACKENDS.values():
        assert key_env not in ("MOONSHOT_API_KEY", "ARK_API_KEY")
        assert "moonshot" not in base_url and "volces" not in base_url
        assert not model.startswith(("kimi", "doubao", "moonshotai/", "qwen/", "deepseek/"))


@pytest.mark.parametrize("provider", REMOVED)
def test_evidence_client_rejects_removed_provider_before_network(provider, monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "x")
    monkeypatch.setenv("ARK_API_KEY", "x")
    with pytest.raises(ValueError, match=f"unsupported provider: {provider}"):
        evidence_client.EvidenceChatClient(provider)


@pytest.mark.parametrize("provider", REMOVED)
def test_cli_rejects_removed_provider(provider, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_experiment_9_1.py", "--provider", provider])
    with pytest.raises(SystemExit) as exc:
        run_experiment_9_1.main()
    assert exc.value.code == 2
