"""Regression tests for email attachments and the SendGrid sender address.

Three defects, all reachable through the documented API:

  * A missing attachment path was skipped inside `if path.exists():`, so the
    send reported success with the file silently absent -- undetectable by the
    caller until the recipient asks where it is.
  * The Content-Disposition filename was interpolated with an f-string, giving
    the unquoted `filename=Q3 refund report.pdf`. That is malformed per RFC 2183;
    strict parsers truncate at the first space and the file arrives as "Q3".
  * SendGrid read its sender from SMTP_FROM_EMAIL only, so the documented
    "SendGrid as an alternative to SMTP" setup passed from_email=None.
"""

import asyncio
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import notification_tools as nt  # noqa: E402


@pytest.fixture
def smtp_capture(monkeypatch):
    """Capture the MIME message instead of sending it."""
    box = {}

    fake = types.ModuleType("aiosmtplib")

    async def _send(msg, **kwargs):
        box["msg"] = msg
        box["kwargs"] = kwargs

    fake.send = _send
    monkeypatch.setitem(sys.modules, "aiosmtplib", fake)

    from config import config

    monkeypatch.setattr(config.email, "sendgrid_api_key", None)
    monkeypatch.setattr(config.email, "smtp_username", "me@example.com")
    monkeypatch.setattr(config.email, "smtp_password", "pw")
    monkeypatch.setattr(config.email, "smtp_from_email", "me@example.com")
    monkeypatch.setattr(config.email, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(config.email, "smtp_port", 587)
    monkeypatch.setattr(config.email, "smtp_use_tls", True)
    return box


class TestMissingAttachments:
    def test_missing_attachment_fails_the_send(self, smtp_capture, tmp_path):
        result = asyncio.run(
            nt.send_email("to@example.com", "s", "b",
                          attachments=[str(tmp_path / "nope.pdf")])
        )
        assert result["success"] is False
        assert "not found" in result["error"]
        assert "msg" not in smtp_capture, "must not connect when an attachment is missing"

    def test_error_names_the_missing_path(self, smtp_capture, tmp_path):
        missing = str(tmp_path / "quarterly.pdf")
        result = asyncio.run(nt.send_email("to@example.com", "s", "b", attachments=[missing]))
        assert missing in result["error"]

    def test_a_directory_is_not_a_valid_attachment(self, smtp_capture, tmp_path):
        result = asyncio.run(nt.send_email("to@example.com", "s", "b", attachments=[str(tmp_path)]))
        assert result["success"] is False

    def test_present_attachment_still_sends(self, smtp_capture, tmp_path):
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4")
        result = asyncio.run(nt.send_email("to@example.com", "s", "b", attachments=[str(f)]))
        assert result["success"] is True
        assert len(smtp_capture["msg"].get_payload()) == 2  # body + attachment


class TestFilenameQuoting:
    @pytest.mark.parametrize("name", ["Q3 refund report.pdf", "report (final).pdf"])
    def test_filenames_with_spaces_are_quoted(self, smtp_capture, tmp_path, name):
        f = tmp_path / name
        f.write_bytes(b"data")
        asyncio.run(nt.send_email("to@example.com", "s", "b", attachments=[str(f)]))

        parts = [p for p in smtp_capture["msg"].get_payload() if p.get("Content-Disposition")]
        assert len(parts) == 1
        disposition = parts[0]["Content-Disposition"]
        assert f'filename="{name}"' in disposition, f"unquoted filename: {disposition}"
        assert parts[0].get_filename() == name


class TestSendGridSender:
    def test_sendgrid_only_setup_reports_missing_sender_clearly(self, monkeypatch):
        from config import config

        monkeypatch.setattr(config.email, "sendgrid_api_key", "SG.looks-real")
        monkeypatch.setattr(config.email, "smtp_from_email", None)
        monkeypatch.setattr(config.email, "smtp_username", None)

        result = asyncio.run(nt.send_email("to@example.com", "s", "b"))
        assert result["success"] is False
        assert "SMTP_FROM_EMAIL" in result["message"]

    def test_sendgrid_falls_back_to_smtp_username(self, monkeypatch):
        """A SendGrid-only setup with just SMTP_USERNAME set should still resolve a sender."""
        from config import config

        monkeypatch.setattr(config.email, "sendgrid_api_key", "SG.looks-real")
        monkeypatch.setattr(config.email, "smtp_from_email", None)
        monkeypatch.setattr(config.email, "smtp_username", "me@example.com")

        # Patch only the transport on the real sendgrid package, so the Mail
        # object is built by the genuine library -- that is what caught
        # add_plain_text_content() not existing in sendgrid 6.x.
        import sendgrid

        sent = {}

        class _FakeClient:
            def __init__(self, key):
                pass

            def send(self, message):
                sent["message"] = message
                return types.SimpleNamespace(status_code=202)

        monkeypatch.setattr(sendgrid, "SendGridAPIClient", _FakeClient)

        result = asyncio.run(nt.send_email("to@example.com", "s", "b"))
        assert result["success"] is True
        assert sent["message"].from_email.email == "me@example.com"


class TestSendGridBodyApi:
    """sendgrid 6.x Mail has no add_plain_text_content/add_html_content."""

    @pytest.fixture
    def sendgrid_capture(self, monkeypatch):
        import sendgrid
        from config import config

        monkeypatch.setattr(config.email, "sendgrid_api_key", "SG.looks-real")
        monkeypatch.setattr(config.email, "smtp_from_email", "me@example.com")

        sent = {}

        class _FakeClient:
            def __init__(self, key):
                pass

            def send(self, message):
                sent["message"] = message
                return types.SimpleNamespace(status_code=202)

        monkeypatch.setattr(sendgrid, "SendGridAPIClient", _FakeClient)
        return sent

    def test_plain_text_body_is_attached(self, sendgrid_capture):
        result = asyncio.run(nt.send_email("to@example.com", "s", "hello body"))
        assert result["success"] is True
        contents = sendgrid_capture["message"].contents
        assert any(c.mime_type == "text/plain" and c.content == "hello body" for c in contents)

    def test_html_body_uses_html_mime_type(self, sendgrid_capture):
        result = asyncio.run(nt.send_email("to@example.com", "s", "<b>hi</b>", html=True))
        assert result["success"] is True
        contents = sendgrid_capture["message"].contents
        assert any(c.mime_type == "text/html" for c in contents)
