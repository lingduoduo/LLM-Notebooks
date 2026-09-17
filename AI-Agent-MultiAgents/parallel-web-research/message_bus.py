"""
In-process asynchronous message bus
===================================

This mimics Redis Pub/Sub semantics while running entirely inside a single
process's asyncio event loop, with no Redis deployment required. It is the
communication substrate for the "central coordination" in Experiment 10-4:

- every message is wrapped in an ``Envelope`` carrying sender_id / target /
  type / payload;
- an Agent calls ``subscribe()`` to obtain a subscription handle and receives
  messages by type;
- an Agent calls ``publish()`` to deliver a message to a specific target or to
  broadcast it to everyone;
- the bus itself makes no business decisions; it only "delivers envelopes to
  subscribers reliably".

Design notes:
- each subscriber's inbox is an ``asyncio.Queue``, which is naturally safe
  across threads and coroutines;
- when ``target`` is ``BROADCAST`` the envelope goes to everyone subscribed to
  that type, except the sender;
- timestamped event logs are printed so the publish/subscribe message flow is
  visible during a demo.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Broadcast target constant: deliver to every subscriber
BROADCAST = "*"

# Globally monotonic message sequence number, for tracing order in the logs
_seq_counter = itertools.count(1)

# Demo start time, used to print relative timestamps (easier to read)
_START_TIME = time.monotonic()


def _now() -> float:
    """Return seconds since the demo started (a relative timestamp)."""
    return time.monotonic() - _START_TIME


@dataclass
class Envelope:
    """Message envelope: the smallest unit flowing through the bus."""

    sender_id: str            # sender ID
    target: str               # target Agent ID, or BROADCAST for a broadcast
    type: str                 # message type: task_assigned / status_update / result / terminate / ack ...
    payload: Dict[str, Any] = field(default_factory=dict)  # JSON payload
    seq: int = field(default_factory=lambda: next(_seq_counter))  # global sequence number
    ts: float = field(default_factory=_now)                       # relative timestamp

    def short(self) -> str:
        """Compact single-line representation for the logs."""
        tgt = "ALL" if self.target == BROADCAST else self.target
        try:
            body = json.dumps(self.payload, ensure_ascii=False, default=str)
        except Exception:
            body = str(self.payload)
        if len(body) > 80:
            body = body[:77] + "..."
        return (
            f"[t={self.ts:6.2f}s #{self.seq:<3}] "
            f"{self.sender_id:>11} -> {tgt:<11} | {self.type:<14} | {body}"
        )


class Subscription:
    """Subscription handle: an inbox queue plus the set of message types of interest."""

    def __init__(self, owner_id: str, types: Optional[List[str]]):
        self.owner_id = owner_id
        # types=None means subscribe to every type
        self.types = set(types) if types is not None else None
        self.inbox: "asyncio.Queue[Envelope]" = asyncio.Queue()

    def accepts(self, env: Envelope) -> bool:
        return self.types is None or env.type in self.types

    async def get(self) -> Envelope:
        return await self.inbox.get()

    async def get_nowait_or_wait(self, timeout: float) -> Optional[Envelope]:
        """Take one message with a timeout; return None on timeout.

        This lets a sub-Agent poll for the termination signal inside a loop.
        """
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class MessageBus:
    """Asynchronous message bus: register subscribers, deliver envelopes, log the flow."""

    def __init__(self, verbose: bool = True):
        # owner_id -> that owner's list of subscriptions
        self._subs: Dict[str, List[Subscription]] = {}
        self.verbose = verbose
        # Record every envelope that crosses the bus, for later counting/assertions
        self.history: List[Envelope] = []

    def subscribe(self, owner_id: str, types: Optional[List[str]] = None) -> Subscription:
        """Register a subscriber and return its handle. types=None receives every type."""
        sub = Subscription(owner_id, types)
        self._subs.setdefault(owner_id, []).append(sub)
        return sub

    async def publish(self, env: Envelope) -> None:
        """Deliver an envelope to the bus: broadcast or point to point."""
        self.history.append(env)
        if self.verbose:
            print("  BUS " + env.short())

        delivered = 0
        for owner_id, sub_list in self._subs.items():
            # Point to point: deliver only to the named target
            if env.target != BROADCAST and owner_id != env.target:
                continue
            # On a broadcast, do not deliver back to the sender
            if env.target == BROADCAST and owner_id == env.sender_id:
                continue
            for sub in sub_list:
                if sub.accepts(env):
                    await sub.inbox.put(env)
                    delivered += 1

        # Yield to the event loop so the peer picks the message up promptly
        # (closer to real push timing)
        await asyncio.sleep(0)

    # ---- Convenience: build and publish ----
    async def send(
        self,
        sender_id: str,
        target: str,
        type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Envelope:
        env = Envelope(sender_id=sender_id, target=target, type=type, payload=payload or {})
        await self.publish(env)
        return env
