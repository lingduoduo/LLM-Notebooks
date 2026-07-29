"""
event_loop_demo.py — end-to-end demo of an event-driven Agent (single process,
runs offline).

The "asynchronous, event-driven Agent" idea is that genuine proactive service
needs more than an Agent that polls the world on a schedule: the world must be
able to notify the Agent. This script gets that running with as little code as
possible:

  1. Register a few "trigger sources". Each one runs on a background thread and
     pushes a structured Event onto a shared queue the moment something happens:
       - OneShotTimer     -- the one-shot flavour of set_timer
       - RecurringTimer   -- the recurring flavour of set_timer
       - FileWatchTrigger -- a file-change trigger, like the ones n8n offers
  2. The EventLoop pulls events off the queue one at a time and wakes the Agent
     to handle each one. That closes the loop of "the Agent registers interest,
     an external event fires and wakes it asynchronously".

Unlike server.py / client.py, which need an HTTP server, this script plays both
"the outside world" and "the Agent" inside a single process, which makes it a
good way to watch the event-driven behaviour directly.

Offline mode (--mock): no LLM calls. A canned "simulated action" prints what the
Agent does once it is woken, so you can watch the whole trigger -> wake ->
handle cycle without an API key.
Live mode (default): wires up EventTriggeredAgent so a real OpenAI model handles
every event.

Usage:
    python event_loop_demo.py --mock                       # offline, all triggers
    python event_loop_demo.py --mock --trigger timer       # one-shot timer only
    python event_loop_demo.py --mock --trigger recurring --interval 3 --duration 12
    python event_loop_demo.py --trigger file --watch-dir ./watched   # real Agent
"""

import os
import sys
import time
import queue
import logging
import argparse
import threading
from datetime import datetime
from typing import Optional, Callable

from event_types import Event, EventType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("event_loop")


# ============================================================================
# Trigger sources
# ============================================================================

class TriggerSource(threading.Thread):
    """Base class for trigger sources: runs on a background thread and pushes
    events onto the shared event queue.

    Registration happens when the instance is created and start()ed; firing
    happens inside run(), which calls self.emit(event) once its condition is
    met. Those are the two distinct moments: the Agent registers interest up
    front, and an external event calls back into it asynchronously later.
    """

    def __init__(self, name: str, event_queue: "queue.Queue[Event]"):
        super().__init__(name=name, daemon=True)
        self.event_queue = event_queue
        self._stop = threading.Event()

    def emit(self, event: Event):
        """Fire: push the event onto the queue, waking the event loop."""
        logger.info(f"⚡ [{self.name}] fired event -> {event.event_type.value}: {event.content}")
        self.event_queue.put(event)

    def stop(self):
        self._stop.set()


class OneShotTimer(TriggerSource):
    """One-shot timer: fires a single timer_trigger event after `delay` seconds.

    This models a task with one specific deadline, e.g. "the user asked me to
    call the DMV, it is Saturday, so schedule 'call the DMV at 10:00 on Monday'".
    """

    def __init__(self, event_queue, delay: float, content: str, timer_id: str = "oneshot"):
        super().__init__(name=f"OneShotTimer({timer_id})", event_queue=event_queue)
        self.delay = delay
        self.content = content
        self.timer_id = timer_id

    def run(self):
        logger.info(f"⏱️  [{self.name}] registered: fires in {self.delay:.0f}s")
        if self._stop.wait(self.delay):
            return
        self.emit(Event(
            event_type=EventType.TIMER_TRIGGER,
            content=self.content,
            metadata={"timer_id": self.timer_id, "kind": "one_shot",
                      "scheduled_delay_seconds": self.delay},
        ))


class RecurringTimer(TriggerSource):
    """Recurring timer: fires a timer_trigger event every `interval` seconds.

    This models "check server health every hour" or "send a progress report
    every Friday", as well as heartbeat-style polling.
    """

    def __init__(self, event_queue, interval: float, content: str, timer_id: str = "recurring"):
        super().__init__(name=f"RecurringTimer({timer_id})", event_queue=event_queue)
        self.interval = interval
        self.content = content
        self.timer_id = timer_id

    def run(self):
        logger.info(f"🔁 [{self.name}] registered: fires every {self.interval:.0f}s")
        tick = 0
        while not self._stop.wait(self.interval):
            tick += 1
            self.emit(Event(
                event_type=EventType.TIMER_TRIGGER,
                content=f"{self.content} (tick #{tick})",
                metadata={"timer_id": self.timer_id, "kind": "recurring",
                          "interval_seconds": self.interval, "tick": tick},
            ))


class FileWatchTrigger(TriggerSource):
    """File watch: polls a directory and fires a file_change event whenever a
    file is created or modified.

    This stands in for the trigger ecosystem of workflow platforms like n8n:
    webhooks, timers, email, database changes, file watches. Polling keeps it
    dependency-free and portable, so it runs offline on any platform.
    """

    def __init__(self, event_queue, watch_dir: str, poll_interval: float = 1.0):
        super().__init__(name=f"FileWatch({watch_dir})", event_queue=event_queue)
        self.watch_dir = watch_dir
        self.poll_interval = poll_interval
        self._snapshot = {}

    def _scan(self):
        snapshot = {}
        try:
            for entry in os.scandir(self.watch_dir):
                if entry.is_file():
                    snapshot[entry.name] = entry.stat().st_mtime
        except FileNotFoundError:
            pass
        return snapshot

    def run(self):
        os.makedirs(self.watch_dir, exist_ok=True)
        self._snapshot = self._scan()
        logger.info(f"👀 [{self.name}] registered: polling every {self.poll_interval:.0f}s "
                    f"({len(self._snapshot)} file(s) already present)")
        while not self._stop.wait(self.poll_interval):
            current = self._scan()
            for name, mtime in current.items():
                if name not in self._snapshot:
                    change = "created"
                elif mtime != self._snapshot[name]:
                    change = "modified"
                else:
                    continue
                self.emit(Event(
                    event_type=EventType.FILE_CHANGE,
                    content=f"A file was {change}. Read its contents and give a "
                            f"short recommendation on how to handle it.",
                    metadata={"path": os.path.join(self.watch_dir, name), "change": change},
                ))
            self._snapshot = current


# ============================================================================
# Event loop
# ============================================================================

class SimulatedExternalWriter(threading.Thread):
    """Writes a file into the watched directory a few seconds in.

    The file-watch demo is only interesting if something actually changes a
    file while it runs, and expecting the reader to race to a second terminal
    inside the run window makes the demo look broken when nothing happens.
    This thread plays the part of that external writer. Pass --no-auto-write
    to turn it off and drive the directory by hand instead.
    """

    def __init__(self, watch_dir: str, delay: float = 3.0,
                 filename: str = "demo_note.txt"):
        super().__init__(name="SimulatedExternalWriter", daemon=True)
        self.watch_dir = watch_dir
        self.delay = delay
        self.filename = filename
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        if self._stop.wait(self.delay):
            return
        os.makedirs(self.watch_dir, exist_ok=True)
        path = os.path.join(self.watch_dir, self.filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("Deployment finished at 03:14. 2 warnings, 0 errors.\n")
            logger.info(f"✍️  [simulated external writer] wrote {path}")
        except OSError as e:
            logger.warning(f"could not write {path}: {e}")


class EventLoop:
    """A single event queue plus single-threaded dispatch.

    Every trigger pushes its heterogeneous events onto one queue; the event loop
    pulls them off in arrival order, and each event wakes the Agent once. This
    is the smallest possible version of "model all input as an event stream and
    let the event loop drive the Agent's thinking and acting".
    """

    def __init__(self, dispatch: Callable[[Event], None]):
        self.event_queue: "queue.Queue[Event]" = queue.Queue()
        self.dispatch = dispatch
        self.triggers = []
        self.processed = 0

    def add_trigger(self, trigger: TriggerSource):
        self.triggers.append(trigger)

    def run(self, duration: float):
        """Start every trigger, then run for `duration` seconds and stop."""
        deadline = time.monotonic() + duration
        for t in self.triggers:
            t.start()

        logger.info(f"🟢 Event loop started; running for {duration:.0f}s, "
                    f"waiting for events to wake the Agent...\n")
        while time.monotonic() < deadline:
            try:
                event = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.processed += 1
            logger.info(f"\n{'='*80}\n📥 Event loop dequeued event #{self.processed}"
                        f" -> waking the Agent\n{'='*80}")
            try:
                self.dispatch(event)
            except Exception as e:  # noqa: BLE001 - one bad event must not kill the demo loop
                logger.error(f"❌ Error while handling event: {e}")

        for t in self.triggers:
            t.stop()
        logger.info(f"\n🔴 Event loop finished; handled {self.processed} event(s).")


# ============================================================================
# Dispatch handlers: simulated action or a real Agent
# ============================================================================

def make_mock_dispatch() -> Callable[[Event], None]:
    """Offline handler: no LLM call, just prints what the woken Agent would do."""

    def dispatch(event: Event):
        logger.info(f"🤖 Agent woken, received message: {event.to_user_message()}")
        # A deterministic "simulated action" stands in for the LLM + tool calls
        if event.event_type == EventType.TIMER_TRIGGER:
            action = "load the scheduled-task context -> run the routine check -> report back"
        elif event.event_type == EventType.FILE_CHANGE:
            path = event.metadata.get("path", "")
            preview = ""
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    preview = f.read(120).replace("\n", " ")
            except OSError:
                preview = "(could not read the file)"
            action = (f"read {os.path.basename(path)} -> content preview: {preview!r} "
                      f"-> draft a recommendation")
        else:
            action = "parse the event -> call the relevant tools -> produce a result"
        logger.info(f"🛠️  [simulated action] {action}")
        logger.info(f"✅ Agent finished: responded to the {event.event_type.value} event")

    return dispatch


def make_agent_dispatch(model: Optional[str], max_iterations: int) -> Callable[[Event], None]:
    """Live handler: wires up EventTriggeredAgent so a real model handles events."""
    from agent import EventTriggeredAgent, SystemHintConfig, resolve_api_key, DEFAULT_MODEL

    api_key = resolve_api_key()
    if not api_key:
        print("❌ OPENAI_API_KEY is not set (checked the environment and the .env file).")
        print("   Set it first, or run the offline demo: python event_loop_demo.py --mock")
        sys.exit(1)

    config = SystemHintConfig(
        enable_timestamps=True,
        enable_tool_counter=True,
        enable_todo_list=True,
        enable_detailed_errors=True,
        enable_system_state=True,
        save_trajectory=True,
        trajectory_file="event_loop_trajectory.json",
        temperature=0.7,
        use_mcp_servers=False,  # built-in tools only here, to avoid the MCP dependency
    )
    agent = EventTriggeredAgent(api_key=api_key, model=model, config=config, verbose=True)
    logger.info(f"✅ Live Agent ready (model={agent.model})")

    def dispatch(event: Event):
        result = agent.handle_event(event, max_iterations=max_iterations)
        logger.info(f"✅ Agent finished: success={result['success']}, "
                    f"iterations={result['iterations']}, "
                    f"tool_calls={len(result['tool_calls'])}")

    return dispatch


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end event-driven Agent demo: register triggers and let "
                    "external events wake the Agent asynchronously.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python event_loop_demo.py --mock
      Offline demo of every trigger (one-shot + recurring + file watch); no API key needed
  python event_loop_demo.py --mock --trigger timer
      One-shot timer only
  python event_loop_demo.py --mock --trigger recurring --interval 3 --duration 12
      Recurring timer firing every 3s for a 12s run
  python event_loop_demo.py --mock --trigger file --watch-dir ./watched
      Watch ./watched; writing a file in there fires an event
  python event_loop_demo.py --trigger timer
      Handle the one-shot timer event with a real OpenAI model (needs OPENAI_API_KEY)
""",
    )
    parser.add_argument(
        "--trigger", choices=["timer", "recurring", "file", "all"], default="all",
        help="Which trigger to demo: timer=one-shot, recurring=recurring timer, "
             "file=file watch, all=every one (default: all)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Offline mode: no LLM calls, uses simulated actions to show the "
             "trigger -> wake -> handle cycle (no API key needed)",
    )
    parser.add_argument(
        "--duration", type=float, default=12.0,
        help="How long the event loop runs, in seconds; all triggers stop afterwards (default: 12)",
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Delay before the one-shot timer fires, in seconds (default: 3)",
    )
    parser.add_argument(
        "--interval", type=float, default=4.0,
        help="Interval between recurring-timer firings, in seconds (default: 4)",
    )
    parser.add_argument(
        "--watch-dir", default="watched_dir",
        help="Directory the file-watch trigger monitors; created if missing (default: watched_dir)",
    )
    parser.add_argument(
        "--no-auto-write", dest="auto_write", action="store_false",
        help="Do not write a file into the watched directory automatically; "
             "drive the file-watch trigger by hand from another terminal instead",
    )
    parser.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL"),
        help="OpenAI model to use in live mode (default: OPENAI_MODEL env var, else gpt-5.2)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum tool-call rounds per event in live mode (default: 10)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    print("\n" + "=" * 80)
    print("🚀 EVENT-DRIVEN AGENT DEMO")
    print("=" * 80)
    print(f"Trigger: {args.trigger} | Mode: {'offline mock' if args.mock else 'live Agent'} | "
          f"Duration: {args.duration:.0f}s")
    print("=" * 80 + "\n")
    sys.stdout.flush()

    if args.mock:
        dispatch = make_mock_dispatch()
    else:
        dispatch = make_agent_dispatch(args.model, args.max_iterations)

    loop = EventLoop(dispatch)

    if args.trigger in ("timer", "all"):
        loop.add_trigger(OneShotTimer(
            loop.event_queue, delay=args.delay, timer_id="daily_backup_check",
            content="One-shot timer fired: check whether the daily backup has finished.",
        ))
    if args.trigger in ("recurring", "all"):
        loop.add_trigger(RecurringTimer(
            loop.event_queue, interval=args.interval, timer_id="health_check",
            content="Recurring timer fired: check the health of the server.",
        ))
    writer = None
    if args.trigger in ("file", "all"):
        loop.add_trigger(FileWatchTrigger(loop.event_queue, watch_dir=args.watch_dir))
        print(f"💡 Tip: write or modify a file under {args.watch_dir}/ to fire a file_change event.")
        print(f"   For example, in another terminal: echo hello > {args.watch_dir}/note.txt")
        if args.auto_write:
            # Give the watcher a moment to take its initial snapshot first,
            # otherwise the file is already there and counts as pre-existing.
            writer = SimulatedExternalWriter(args.watch_dir, delay=min(3.0, args.duration / 3))
            print(f"   (a simulated external writer will also drop a file in "
                  f"automatically; pass --no-auto-write to disable)")
        print()
    sys.stdout.flush()

    if not loop.triggers:
        print("❌ No triggers to run.")
        sys.exit(1)

    try:
        if writer:
            writer.start()
        loop.run(duration=args.duration)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted, shutting down...")
        for t in loop.triggers:
            t.stop()
    finally:
        if writer:
            writer.stop()

    print("\n" + "=" * 80)
    print(f"📊 Demo finished: handled {loop.processed} event(s).")
    if loop.processed == 0:
        # A silent zero looks like a broken demo; say why nothing happened.
        print()
        print("   No events fired during this run. That is expected if nothing")
        print("   triggered them — for the file watcher, a file has to be created or")
        print("   modified inside the watched directory while the loop is running.")
        print(f"   Try:  python event_loop_demo.py --mock --trigger file "
              f"--watch-dir {args.watch_dir}")
        print(f"   or a longer window:  --duration 30")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
