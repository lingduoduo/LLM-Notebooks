"""Notification tools for email and instant messaging."""

import asyncio
import json
from typing import Optional, Dict, Any, List
import logging
import httpx

logger = logging.getLogger(__name__)


class AttachmentError(Exception):
    """A requested attachment could not be read."""


def _resolve_attachments(attachments: Optional[List[str]]):
    """Resolve attachment paths, raising rather than skipping unreadable ones.

    Both send paths used to wrap the attach step in `if path.exists():` and do
    nothing otherwise, so a typo'd or moved path produced a "successfully sent"
    result with the attachment silently missing -- the one failure mode the
    caller cannot detect from the outside and will not discover until the
    recipient asks where the file is.
    """
    from pathlib import Path

    resolved = []
    missing = []
    for filepath in attachments or []:
        path = Path(filepath).expanduser()
        if path.is_file():
            resolved.append(path)
        else:
            missing.append(str(filepath))

    if missing:
        raise AttachmentError(
            f"Attachment(s) not found: {', '.join(missing)}"
        )
    return resolved


async def send_email(
    to_email: str,
    subject: str,
    body: str,
    html: bool = False,
    cc: Optional[List[str]] = None,
    attachments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Send an email notification.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body content
        html: Whether body is HTML formatted
        cc: Optional list of CC recipients
        attachments: Optional list of file paths to attach
        
    Returns:
        Dictionary with send status
    """
    try:
        from config import config

        # Validate attachments before opening any connection, so a typo'd path
        # is reported as itself rather than surfacing later as an SMTP error.
        try:
            _resolve_attachments(attachments)
        except AttachmentError as exc:
            logger.error("Refusing to send: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "message": f"Email to {to_email} was not sent because an attachment is missing"
            }

        # Check if SendGrid is configured (preferred)
        if config.email.sendgrid_api_key:
            return await _send_email_sendgrid(
                to_email, subject, body, html, cc, attachments
            )
        # Fall back to SMTP
        elif config.email.smtp_username and config.email.smtp_password:
            return await _send_email_smtp(
                to_email, subject, body, html, cc, attachments
            )
        else:
            return {
                "success": False,
                "error": "No email service configured",
                "message": "Please configure SendGrid API key or SMTP credentials"
            }
            
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to send email to {to_email}"
        }


async def _send_email_smtp(
    to_email: str,
    subject: str,
    body: str,
    html: bool,
    cc: Optional[List[str]],
    attachments: Optional[List[str]]
) -> Dict[str, Any]:
    """Send email using SMTP."""
    try:
        import aiosmtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from email.mime.base import MIMEBase
        from email import encoders
        from pathlib import Path
        from config import config
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config.email.smtp_from_email or config.email.smtp_username
        msg['To'] = to_email
        msg['Subject'] = subject
        
        if cc:
            msg['Cc'] = ', '.join(cc)
        
        # Add body
        mime_type = 'html' if html else 'plain'
        msg.attach(MIMEText(body, mime_type))
        
        # Add attachments. Missing paths raise instead of being skipped.
        for path in _resolve_attachments(attachments):
            with open(path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                # Pass the filename as a parameter rather than interpolating it
                # into the header value: an f-string produced the unquoted
                # `filename=Q3 refund report.pdf`, which is malformed per
                # RFC 2183. Lenient parsers recover it, strict ones truncate at
                # the first space and the file arrives named "Q3".
                part.add_header('Content-Disposition', 'attachment', filename=path.name)
                msg.attach(part)
        
        # SMTP has two different, mutually exclusive encryption mechanisms, and
        # SMTP_USE_TLS only says "encrypt", not which one:
        #
        #   implicit TLS (SMTPS, port 465) -- wrap the socket before any SMTP
        #       dialogue. This is aiosmtplib's `use_tls`.
        #   STARTTLS (port 587, and 25)    -- connect in plaintext, then upgrade
        #       via the STARTTLS command. This is aiosmtplib's `start_tls`.
        #
        # Passing use_tls=True on 587 makes the client offer a TLS handshake to
        # a server still speaking plaintext SMTP, which fails as:
        #   [SSL: WRONG_VERSION_NUMBER] wrong version number
        # That looked like a credentials or firewall problem but is purely the
        # wrong mechanism. Pick it from the port, which is what actually
        # determines it. SMTP_USE_TLS=false disables encryption entirely.
        port = config.email.smtp_port
        encrypt = config.email.smtp_use_tls
        implicit_tls = encrypt and port == 465
        starttls = encrypt and port != 465

        await aiosmtplib.send(
            msg,
            hostname=config.email.smtp_host,
            port=port,
            username=config.email.smtp_username,
            password=config.email.smtp_password,
            use_tls=implicit_tls,
            start_tls=starttls
        )
        
        return {
            "success": True,
            "to": to_email,
            "subject": subject,
            "method": "SMTP",
            "message": f"Email sent successfully to {to_email}"
        }
        
    except Exception as e:
        logger.error(f"SMTP send failed: {e}")
        raise


async def _send_email_sendgrid(
    to_email: str,
    subject: str,
    body: str,
    html: bool,
    cc: Optional[List[str]],
    attachments: Optional[List[str]]
) -> Dict[str, Any]:
    """Send email using SendGrid API."""
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import (
            Mail, Attachment, FileContent, FileName, FileType, Disposition, Content
        )
        import base64
        from pathlib import Path
        from config import config

        # SendGrid is usable on its own, but the from address was read only from
        # SMTP_FROM_EMAIL -- so a SendGrid-only setup (the documented
        # "alternative to SMTP") passed from_email=None and failed inside the
        # SendGrid client with an error that named neither variable.
        from_email = config.email.smtp_from_email or config.email.smtp_username
        if not from_email:
            return {
                "success": False,
                "error": "No sender address configured",
                "message": "Set SMTP_FROM_EMAIL (used as the SendGrid sender address)"
            }

        # Create message
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject
        )
        
        # Add body.
        #
        # `add_html_content` / `add_plain_text_content` do not exist on Mail in
        # sendgrid 6.x -- the version this project pins (sendgrid>=6.11.0). Every
        # SendGrid send raised:
        #     'Mail' object has no attribute 'add_plain_text_content'
        # so this path could never have delivered a message. The supported API is
        # add_content() with an explicit MIME type.
        message.add_content(Content("text/html" if html else "text/plain", body))
        
        # Add CC
        if cc:
            for cc_email in cc:
                message.add_cc(cc_email)
        
        # Add attachments. Missing paths raise instead of being skipped.
        for path in _resolve_attachments(attachments):
            with open(path, 'rb') as f:
                data = f.read()
                encoded = base64.b64encode(data).decode()
                attachment = Attachment(
                    FileContent(encoded),
                    FileName(path.name),
                    FileType('application/octet-stream'),
                    Disposition('attachment')
                )
                message.add_attachment(attachment)
        
        # Send
        sg = SendGridAPIClient(config.email.sendgrid_api_key)
        response = sg.send(message)
        
        return {
            "success": True,
            "to": to_email,
            "subject": subject,
            "method": "SendGrid",
            "status_code": response.status_code,
            "message": f"Email sent successfully to {to_email}"
        }
        
    except Exception as e:
        logger.error(f"SendGrid send failed: {e}")
        raise


async def send_telegram_message(
    message: str,
    chat_id: Optional[str] = None,
    parse_mode: Optional[str] = None
) -> Dict[str, Any]:
    """Send a Telegram message.

    Args:
        message: Message text to send
        chat_id: Optional Telegram chat ID (uses default if not provided)
        parse_mode: Message parse mode ("HTML", "Markdown", or None for plain
            text). Defaults to None: with a markup mode set, Telegram rejects
            the whole message if the body contains an unescaped '<' or '&',
            which is easy to hit with arbitrary notification text. Pass "HTML"
            only when the caller has escaped the body itself.

    Returns:
        Dictionary with send status
    """
    try:
        from config import config
        
        if not config.im.telegram_bot_token:
            return {
                "success": False,
                "error": "Telegram bot token not configured",
                "message": "Please set TELEGRAM_BOT_TOKEN in environment"
            }
        
        target_chat_id = chat_id or config.im.telegram_default_chat_id
        if not target_chat_id:
            return {
                "success": False,
                "error": "No chat ID provided",
                "message": "Please provide chat_id parameter or set TELEGRAM_DEFAULT_CHAT_ID"
            }
        
        # Send message via Telegram Bot API
        url = f"https://api.telegram.org/bot{config.im.telegram_bot_token}/sendMessage"
        
        payload = {
            "chat_id": target_chat_id,
            "text": message
        }
        
        if parse_mode:
            payload["parse_mode"] = parse_mode
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
        
        return {
            "success": True,
            "chat_id": target_chat_id,
            "message_id": result.get("result", {}).get("message_id"),
            "message": "Telegram message sent successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to send Telegram message"
        }


async def send_slack_message(
    message: str,
    webhook_url: Optional[str] = None,
    channel: Optional[str] = None,
    username: str = "Collaboration Agent"
) -> Dict[str, Any]:
    """Send a Slack message via webhook.
    
    Args:
        message: Message text to send
        webhook_url: Optional Slack webhook URL (uses default if not provided)
        channel: Optional channel to post to
        username: Bot username to display
        
    Returns:
        Dictionary with send status
    """
    try:
        from config import config
        
        target_webhook = webhook_url or config.im.slack_webhook_url
        if not target_webhook:
            return {
                "success": False,
                "error": "Slack webhook URL not configured",
                "message": "Please provide webhook_url or set SLACK_WEBHOOK_URL"
            }
        
        payload = {
            "text": message,
            "username": username
        }
        
        if channel:
            payload["channel"] = channel
        
        async with httpx.AsyncClient() as client:
            response = await client.post(target_webhook, json=payload)
            response.raise_for_status()
        
        return {
            "success": True,
            "channel": channel or "default",
            "message": "Slack message sent successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to send Slack message: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to send Slack message"
        }


async def send_discord_message(
    message: str,
    webhook_url: Optional[str] = None,
    username: str = "Collaboration Agent"
) -> Dict[str, Any]:
    """Send a Discord message via webhook.
    
    Args:
        message: Message text to send
        webhook_url: Optional Discord webhook URL (uses default if not provided)
        username: Bot username to display
        
    Returns:
        Dictionary with send status
    """
    try:
        from config import config
        
        target_webhook = webhook_url or config.im.discord_webhook_url
        if not target_webhook:
            return {
                "success": False,
                "error": "Discord webhook URL not configured",
                "message": "Please provide webhook_url or set DISCORD_WEBHOOK_URL"
            }
        
        payload = {
            "content": message,
            "username": username
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(target_webhook, json=payload)
            response.raise_for_status()
        
        return {
            "success": True,
            "message": "Discord message sent successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to send Discord message"
        }
