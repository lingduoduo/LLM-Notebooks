#!/usr/bin/env python3
"""Collaboration Tools -- unified command line entry point.

The collaboration tools fall into three categories (matching the book's
"collaboration tools" section):
  1. Sub-agent management: spawn_subagent / send_message_to_subagent / cancel_subagent
     (supports both sync and async modes, and the minimal / llm_generated
     context-passing strategies)
  2. Human-in-the-loop (HITL): request_admin_approval / request_admin_input
     (with timeouts and default behaviour)
  3. Multi-channel notifications: email / slack / telegram / discord

Examples:
  python main.py list                      # list every collaboration tool
  python main.py demo                      # run the offline end-to-end demo (no API key needed)
  python main.py subagent compare          # compare the two context-passing strategies
  python main.py subagent spawn --task "Look up the status of order A12345" --strategy minimal
  python main.py hitl approve --message "Delete 1000 records?" --timeout 5 --auto-approve
  python main.py notify slack --message "Deployment complete"

Notes:
  - Sub-agent execution and the llm_generated context strategy require
    OPENAI_API_KEY. Without it, they fall back to a deterministic offline
    simulation (results are explicitly labelled "no LLM call").
  - Really sending notifications / email requires the matching channel
    credentials in .env. When they are missing the tools return a "not
    configured" explanation, and the command itself still parses and runs.
"""

import argparse
import asyncio
import json
import os
import sys

# Modules under src/ use bare imports (consistent with quickstart.py / subagent_comparison.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import subagent_tools as sa  # noqa: E402
import hitl_tools as hitl  # noqa: E402
import notification_tools as notify  # noqa: E402


def _print(obj) -> None:
    """Print a tool's return value as indented JSON."""
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _parse_json_arg(value):
    """Try to parse as JSON; return the raw string if it is not valid JSON (used as-is by the sub-agent)."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------

COLLAB_TOOLS = {
    "Sub-agent management": [
        ("spawn_subagent", "Create a sub-agent (sync/async, minimal/llm_generated context strategy)"),
        ("send_message_to_subagent", "Send a follow-up message to a sub-agent and get its reply"),
        ("cancel_subagent", "Cancel a sub-agent (async tasks abort their background coroutine)"),
        ("get_subagent_status", "Inspect a sub-agent's status and result (used for async)"),
    ],
    "Human-in-the-loop (HITL)": [
        ("request_admin_approval", "Ask an admin to approve a critical decision (with timeout and default behaviour)"),
        ("request_admin_input", "Ask an admin for additional input"),
        ("respond_to_request", "Admin approves/rejects a pending request"),
        ("list_pending_requests", "List every pending approval request"),
    ],
    "Multi-channel notifications": [
        ("send_email", "Send an email notification (SMTP / SendGrid)"),
        ("send_slack_message", "Send a Slack message via webhook"),
        ("send_telegram_message", "Send a Telegram message"),
        ("send_discord_message", "Send a Discord message via webhook"),
    ],
}


def cmd_list(args) -> None:
    print("Collaboration tool catalog (Experiment 4-3)\n" + "=" * 60)
    for category, tools in COLLAB_TOOLS.items():
        print(f"\n[{category}]")
        for name, desc in tools:
            print(f"  - {name:<28} {desc}")
    print("\nTip: run `python main.py <subcommand> -h` to see each tool's arguments.")


# ---------------------------------------------------------------------------
# Sub-agent subcommands
# ---------------------------------------------------------------------------

async def _subagent_dispatch(args) -> None:
    if args.sub_action == "spawn":
        res = await sa.spawn_subagent(
            task=args.task,
            context_strategy=args.strategy,
            mode=args.mode,
            parent_context=_parse_json_arg(args.parent_context),
            role=args.role,
            minimal_slice=_parse_json_arg(args.minimal_slice),
            business_rules=args.business_rules,
        )
        _print(res)
    elif args.sub_action == "send":
        _print(await sa.send_message_to_subagent(args.id, args.message))
    elif args.sub_action == "cancel":
        _print(await sa.cancel_subagent(args.id))
    elif args.sub_action == "status":
        _print(await sa.get_subagent_status(args.id))
    elif args.sub_action == "compare":
        await sa.run_context_strategy_comparison(task=args.task)


def cmd_subagent(args) -> None:
    asyncio.run(_subagent_dispatch(args))


# ---------------------------------------------------------------------------
# HITL subcommands
# ---------------------------------------------------------------------------

async def _auto_responder(approve: bool, notes: str, delay: float = 1.0) -> None:
    """Simulate an admin: poll pending requests and answer them, for the offline HITL demo."""
    await asyncio.sleep(delay)
    pending = await hitl.list_pending_requests()
    for req in pending.get("requests", []):
        await hitl.respond_to_request(req["request_id"], approve, notes)


async def _hitl_dispatch(args) -> None:
    if args.hitl_action == "approve":
        coro = hitl.request_admin_approval(
            request_message=args.message,
            timeout_seconds=args.timeout,
            urgent=args.urgent,
        )
        if args.auto_approve or args.auto_reject:
            responder = _auto_responder(
                approve=not args.auto_reject,
                notes=args.notes or ("Simulated auto-approval" if not args.auto_reject else "Simulated auto-rejection"),
            )
            res, _ = await asyncio.gather(coro, responder)
        else:
            res = await coro
        _print(res)
    elif args.hitl_action == "input":
        coro = hitl.request_admin_input(prompt=args.prompt, timeout_seconds=args.timeout)
        if args.auto_answer is not None:
            responder = _auto_responder(approve=True, notes=args.auto_answer)
            res, _ = await asyncio.gather(coro, responder)
        else:
            res = await coro
        _print(res)
    elif args.hitl_action == "respond":
        _print(await hitl.respond_to_request(args.id, args.approve, args.notes))
    elif args.hitl_action == "list":
        _print(await hitl.list_pending_requests())


def cmd_hitl(args) -> None:
    asyncio.run(_hitl_dispatch(args))


# ---------------------------------------------------------------------------
# Notification subcommands
# ---------------------------------------------------------------------------

async def _notify_dispatch(args) -> None:
    if args.channel == "email":
        _print(await notify.send_email(args.to, args.subject, args.body))
    elif args.channel == "slack":
        _print(await notify.send_slack_message(args.message, webhook_url=args.webhook))
    elif args.channel == "telegram":
        _print(await notify.send_telegram_message(args.message, chat_id=args.chat_id))
    elif args.channel == "discord":
        _print(await notify.send_discord_message(args.message, webhook_url=args.webhook))


def cmd_notify(args) -> None:
    asyncio.run(_notify_dispatch(args))


# ---------------------------------------------------------------------------
# End-to-end demo: a support coordinator agent handling one refund
# ---------------------------------------------------------------------------

def _neutralize_network_creds() -> list:
    """Blank every notification credential for the duration of the demo.

    The demo is meant to be runnable offline and repeatedly, so it must never
    send real mail or POST to a real webhook even when .env is fully configured.

    Returns the names of the channels that were actually suppressed, so the demo
    can distinguish "you have not configured this" from "you configured this and
    the demo is deliberately not using it" -- reporting the latter as the former
    reads as a broken setup.
    """
    from config import config

    suppressed = []
    if config.email.smtp_username or config.email.sendgrid_api_key:
        suppressed.append("email")
    if config.im.slack_webhook_url:
        suppressed.append("slack")
    if config.im.telegram_bot_token:
        suppressed.append("telegram")
    if config.im.discord_webhook_url:
        suppressed.append("discord")

    config.email.smtp_username = None
    config.email.smtp_password = None
    config.email.sendgrid_api_key = None
    config.im.telegram_bot_token = None
    config.im.slack_webhook_url = None
    config.im.discord_webhook_url = None
    config.hitl.webhook_url = None
    config.hitl.admin_email = None

    return suppressed


def _demo_recipient() -> str:
    """Who step 3 mails in --live mode.

    Never the hardcoded admin@example.com the suppressed demo "sends" to: that
    is a real reserved domain, so a live run would emit mail that just bounces.
    Prefer the configured admin, else the sender's own address (a self-test).
    """
    from config import config

    return (
        config.hitl.admin_email
        or config.email.smtp_from_email
        or config.email.smtp_username
        or "admin@example.com"
    )


async def _demo(live: bool = False) -> None:
    if live:
        suppressed = []
        recipient = _demo_recipient()
    else:
        suppressed = _neutralize_network_creds()
        recipient = "admin@example.com"
    online = bool(os.getenv("OPENAI_API_KEY"))

    print("=" * 74)
    print("End-to-end collaboration demo: a support coordinator agent handling a refund")
    print(f"(Sub-agent execution mode: {'online LLM' if online else 'offline simulation (OPENAI_API_KEY not set)'})")
    print("=" * 74)

    print("\n[Step 1/3] Delegate refund approval to a sub-agent and compare the two context strategies")
    print("-" * 74)
    if not online:
        print("(Note: OPENAI_API_KEY is not set, so sub-agent execution and the llm_generated")
        print("  strategy return errors. This only demonstrates the interface and context")
        print("  construction; set the key to see real results.)")
    await sa.run_context_strategy_comparison()

    print("\n[Step 2/3] A high-value operation triggers HITL: ask an admin to approve (with timeout and default behaviour)")
    print("-" * 74)
    print("-> Scenario A: the admin approves before the timeout (simulated responder in the background)")
    approval, _ = await asyncio.gather(
        hitl.request_admin_approval(
            request_message="Refund amount is 8888 CNY, above the auto-approval threshold. Please confirm manually.",
            timeout_seconds=10,
            urgent=True,
        ),
        _auto_responder(approve=True, notes="Verified, refund approved", delay=1.0),
    )
    _print(approval)

    print("\n-> Scenario B: the admin does not respond in time, triggering the timeout and the conservative default (not approved)")
    timeout_res = await hitl.request_admin_approval(
        request_message="Refund amount is 8888 CNY. Please confirm manually.",
        timeout_seconds=2,
    )
    _print(timeout_res)

    print("\n[Step 3/3] Notify collaborators of the outcome across multiple channels")
    print("-" * 74)
    if suppressed:
        print(f"(Note: {', '.join(suppressed)} IS configured in your .env, but the demo")
        print("  deliberately suppresses real sending so it stays repeatable. Re-run with")
        print("  `python main.py demo --live` to actually send through every configured channel.)")
    elif live:
        print(f"(--live: sending for real through every configured channel. Email -> {recipient})")
    summary = "Refund ticket A12345: approved by the sub-agent, confirmed by the admin, payment released."
    for channel, coro in (
        ("email", notify.send_email(recipient, "Refund processed", summary)),
        ("slack", notify.send_slack_message(summary)),
        ("telegram", notify.send_telegram_message(summary)),
    ):
        res = await coro
        if res.get("success"):
            status = "sent"
        elif channel in suppressed:
            status = "suppressed for the demo (configured, not sent)"
        else:
            status = f"not sent ({res.get('error')})"
        print(f"  [{channel:<8}] {status}: {summary}")

    print("\n" + "=" * 74)
    print("Demo finished. Really sending notifications/email requires the matching channel")
    print("credentials in .env; real LLM sub-agent execution and the llm_generated strategy")
    print("require OPENAI_API_KEY.")
    print("=" * 74)


def cmd_demo(args) -> None:
    asyncio.run(_demo(live=args.live))


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Collaboration tools CLI (Experiment 4-3): sub-agent management / human-in-the-loop / multi-channel notifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py list\n"
            "  python main.py demo\n"
            "  python main.py subagent compare\n"
            "  python main.py subagent spawn --task 'Look up the status of order A12345' --strategy minimal\n"
            "  python main.py hitl approve --message 'Delete 1000 records?' --timeout 5 --auto-approve\n"
            "  python main.py notify slack --message 'Deployment complete'\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    sub.add_parser("list", help="List every collaboration tool").set_defaults(func=cmd_list)

    p_demo = sub.add_parser("demo", help="Run the offline end-to-end collaboration demo (no API key needed)")
    p_demo.add_argument("--live", action="store_true",
                        help="Actually send through every configured channel instead of "
                             "suppressing sends. Emails the configured HITL admin (or your "
                             "own address), not the placeholder admin@example.com.")
    p_demo.set_defaults(func=cmd_demo)

    # subagent
    p_sa = sub.add_parser("subagent", help="Sub-agent management tools")
    sa_sub = p_sa.add_subparsers(dest="sub_action", required=True, metavar="<action>")

    p_spawn = sa_sub.add_parser("spawn", help="Create a sub-agent")
    p_spawn.add_argument("--task", required=True, help="Sub-task delegated to the sub-agent")
    p_spawn.add_argument("--strategy", default="minimal",
                         choices=["minimal", "llm_generated"], help="Context-passing strategy")
    p_spawn.add_argument("--mode", default="sync", choices=["sync", "async"],
                         help="sync waits for the result; async returns a task_id")
    p_spawn.add_argument("--role", default=None, help="The sub-agent's role (used in the system prompt)")
    p_spawn.add_argument("--parent-context", default=None,
                         help="Main agent trajectory/state (JSON string)")
    p_spawn.add_argument("--minimal-slice", default=None,
                         help="Hand-picked information for the minimal strategy (string or JSON)")
    p_spawn.add_argument("--business-rules", default=None,
                         help="Privacy/compression rules for the llm_generated strategy")

    p_send = sa_sub.add_parser("send", help="Send a follow-up message to a sub-agent")
    p_send.add_argument("--id", required=True, help="Sub-agent ID")
    p_send.add_argument("--message", required=True, help="Message content")

    p_cancel = sa_sub.add_parser("cancel", help="Cancel a sub-agent")
    p_cancel.add_argument("--id", required=True, help="Sub-agent ID")

    p_status = sa_sub.add_parser("status", help="Inspect a sub-agent's status/result")
    p_status.add_argument("--id", required=True, help="Sub-agent ID")

    p_cmp = sa_sub.add_parser("compare", help="Compare the minimal and llm_generated strategies")
    p_cmp.add_argument("--task", default=None, help="Shared sub-task used for the comparison")
    p_sa.set_defaults(func=cmd_subagent)

    # hitl
    p_hitl = sub.add_parser("hitl", help="Human-in-the-loop (HITL) tools")
    hitl_sub = p_hitl.add_subparsers(dest="hitl_action", required=True, metavar="<action>")

    p_appr = hitl_sub.add_parser("approve", help="Request admin approval")
    p_appr.add_argument("--message", required=True, help="What needs to be approved")
    p_appr.add_argument("--timeout", type=int, default=None, help="Seconds to wait (default behaviour applies on timeout)")
    p_appr.add_argument("--urgent", action="store_true", help="Mark as urgent")
    p_appr.add_argument("--auto-approve", action="store_true", help="Simulate an admin approval in the background (for offline demos)")
    p_appr.add_argument("--auto-reject", action="store_true", help="Simulate an admin rejection in the background (for offline demos)")
    p_appr.add_argument("--notes", default=None, help="Admin notes")

    p_inp = hitl_sub.add_parser("input", help="Request input from an admin")
    p_inp.add_argument("--prompt", required=True, help="Question/prompt")
    p_inp.add_argument("--timeout", type=int, default=None, help="Seconds to wait")
    p_inp.add_argument("--auto-answer", default=None, help="Simulate an admin answer in the background (for offline demos)")

    p_resp = hitl_sub.add_parser("respond", help="Admin answers a request")
    p_resp.add_argument("--id", required=True, help="Request ID")
    grp = p_resp.add_mutually_exclusive_group(required=True)
    grp.add_argument("--approve", dest="approve", action="store_true", help="Approve")
    grp.add_argument("--reject", dest="approve", action="store_false", help="Reject")
    p_resp.add_argument("--notes", default=None, help="Notes")

    hitl_sub.add_parser("list", help="List pending requests")
    p_hitl.set_defaults(func=cmd_hitl)

    # notify
    p_notify = sub.add_parser("notify", help="Multi-channel notification tools")
    notify_sub = p_notify.add_subparsers(dest="channel", required=True, metavar="<channel>")

    p_email = notify_sub.add_parser("email", help="Send an email")
    p_email.add_argument("--to", required=True, help="Recipient")
    p_email.add_argument("--subject", required=True, help="Subject")
    p_email.add_argument("--body", required=True, help="Body")

    p_slack = notify_sub.add_parser("slack", help="Send a Slack message")
    p_slack.add_argument("--message", required=True, help="Message content")
    p_slack.add_argument("--webhook", default=None, help="Slack webhook URL (defaults to .env)")

    p_tg = notify_sub.add_parser("telegram", help="Send a Telegram message")
    p_tg.add_argument("--message", required=True, help="Message content")
    p_tg.add_argument("--chat-id", default=None, help="Telegram chat id (defaults to .env)")

    p_dc = notify_sub.add_parser("discord", help="Send a Discord message")
    p_dc.add_argument("--message", required=True, help="Message content")
    p_dc.add_argument("--webhook", default=None, help="Discord webhook URL (defaults to .env)")
    p_notify.set_defaults(func=cmd_notify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
