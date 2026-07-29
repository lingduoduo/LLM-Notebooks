"""Regression tests for SMTP encryption mechanism selection.

SMTP has two mutually exclusive encryption mechanisms, and SMTP_USE_TLS only
says "encrypt", not which one:

    implicit TLS (SMTPS, port 465) -- wrap the socket before any SMTP dialogue
    STARTTLS     (port 587 and 25) -- connect plaintext, then upgrade

`smtp_use_tls` was passed straight through as aiosmtplib's `use_tls`, so the
default Gmail config (smtp.gmail.com:587, SMTP_USE_TLS=true) offered a TLS
handshake to a server still speaking plaintext SMTP and died with:

    [SSL: WRONG_VERSION_NUMBER] wrong version number

which reads like a credentials or firewall problem but is purely the wrong
mechanism for the port.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import notification_tools as nt  # noqa: E402


@pytest.fixture
def sent(monkeypatch):
    """Capture the kwargs handed to aiosmtplib.send without touching a network."""
    calls = []

    class _FakeAiosmtplib:
        @staticmethod
        async def send(msg, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "aiosmtplib", _FakeAiosmtplib)

    from config import config

    monkeypatch.setattr(config.email, "smtp_username", "user@example.com")
    monkeypatch.setattr(config.email, "smtp_password", "secret")
    monkeypatch.setattr(config.email, "smtp_from_email", "user@example.com")
    monkeypatch.setattr(config.email, "sendgrid_api_key", None)
    monkeypatch.setattr(config.email, "smtp_host", "smtp.example.com")
    return calls


def _send(port, use_tls, sent):
    from config import config

    config.email.smtp_port = port
    config.email.smtp_use_tls = use_tls
    result = asyncio.run(nt.send_email("to@example.com", "s", "b"))
    assert result["success"] is True
    return sent[0]


class TestMechanismFollowsPort:
    def test_port_587_uses_starttls_not_implicit_tls(self, sent):
        """The exact bug: 587 + use_tls=True produced WRONG_VERSION_NUMBER."""
        kwargs = _send(587, True, sent)
        assert kwargs["start_tls"] is True
        assert kwargs["use_tls"] is False

    def test_port_25_uses_starttls(self, sent):
        kwargs = _send(25, True, sent)
        assert kwargs["start_tls"] is True
        assert kwargs["use_tls"] is False

    def test_port_465_uses_implicit_tls(self, sent):
        kwargs = _send(465, True, sent)
        assert kwargs["use_tls"] is True
        assert kwargs["start_tls"] is False


class TestEncryptionCanBeDisabled:
    @pytest.mark.parametrize("port", [25, 587, 465])
    def test_use_tls_false_disables_both_mechanisms(self, sent, port):
        kwargs = _send(port, False, sent)
        assert kwargs["use_tls"] is False
        assert kwargs["start_tls"] is False


def test_credentials_and_host_are_forwarded(sent):
    kwargs = _send(587, True, sent)
    assert kwargs["hostname"] == "smtp.example.com"
    assert kwargs["port"] == 587
    assert kwargs["username"] == "user@example.com"
    assert kwargs["password"] == "secret"
