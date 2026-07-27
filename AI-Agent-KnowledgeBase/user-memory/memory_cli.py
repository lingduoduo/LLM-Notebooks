#!/usr/bin/env python3
"""
Offline user-memory command line tool (memory_cli)

An **offline** memory-operations CLI that works directly against
memory_manager's persistent store. It needs no LLM API and still walks
through the full lifecycle of the user memory system:
extraction (manual write) -> storage -> update -> dedup / versioned conflict
resolution -> cross-session recall.

How it divides up with main.py:
  * main.py  -- the full conversation / background memory processing /
    evaluation flow, requires an LLM API.
  * memory_cli.py -- create, read, update and consolidate individual
    memories, runs entirely locally so you can inspect storage, dedup and
    conflict-resolution behavior without an API key.

Subcommands:
  add          Write one memory (simulates a fact extracted from a session)
  query        Search memories by keyword (cross-session recall)
  update       Update an existing memory by ID
  consolidate  Dedup memories and resolve conflicts by version (no API)
  show         Print all memories currently stored for a user
  demo         Run a multi-session offline example showing memories being
               reused in later sessions
  extract      Extract memories from a conversation automatically (needs an
               LLM API)

Examples:
  python memory_cli.py demo
  python memory_cli.py add --user alice --session s1 \
      --content "Prefers a window seat" --tags seat_preference
  python memory_cli.py query --user alice --query seat
  python memory_cli.py consolidate --user alice
"""

import argparse
import sys

from config import Config, MemoryMode
from memory_manager import create_memory_manager


# Memory mode string -> enum, shared by all subcommands
MODE_MAP = {
    "notes": MemoryMode.NOTES,
    "enhanced_notes": MemoryMode.ENHANCED_NOTES,
    "json_cards": MemoryMode.JSON_CARDS,
    "advanced_json_cards": MemoryMode.ADVANCED_JSON_CARDS,
}


def _apply_store_path(store_path):
    """Redirect the memory storage directory if --store-path was given (leaves the default data alone)."""
    if store_path:
        Config.MEMORY_STORAGE_DIR = store_path
    Config.create_directories()


def _build_manager(args):
    """Build the memory manager from CLI args (set the storage dir before instantiating)."""
    _apply_store_path(getattr(args, "store_path", None))
    mode = MODE_MAP[args.memory_mode] if getattr(args, "memory_mode", None) else Config.MEMORY_MODE
    manager = create_memory_manager(args.user, mode)
    manager.verbose = True
    return manager, mode


def cmd_add(args):
    """Write one memory. Only notes / enhanced_notes modes accept free-text writes."""
    manager, mode = _build_manager(args)
    if mode not in (MemoryMode.NOTES, MemoryMode.ENHANCED_NOTES):
        print("❌ The add subcommand only supports notes / enhanced_notes modes "
              "(for JSON cards, generate them through main.py's conversation flow)")
        return 1
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    note_id = manager.add_memory(args.content, args.session, tags=tags)
    print(f"✅ Memory written, ID={note_id}")
    return 0


def cmd_query(args):
    """Search memories by keyword -- demonstrates "recalling user info in a later session"."""
    manager, _ = _build_manager(args)
    results = manager.search_memories(args.query)
    if not results:
        print(f"🔍 No memories found related to \"{args.query}\"")
        return 0
    print(f"🔍 Found {len(results)} memories related to \"{args.query}\":")
    for item in results:
        if hasattr(item, "content"):  # MemoryNote
            tags = f" [tags: {', '.join(item.tags)}]" if item.tags else ""
            print(f"  - ({item.note_id[:8]}) {item.content}{tags}")
        else:  # (memory_path, data) tuple from JSON managers
            path, data = item
            print(f"  - {path}: {data}")
    return 0


def cmd_update(args):
    """Update an existing memory by ID (simulates the user supplying updated information)."""
    manager, _ = _build_manager(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    ok = manager.update_memory(args.id, args.content, args.session, tags=tags)
    print("✅ Update succeeded" if ok else "⚠️  No memory found with that ID, update failed")
    return 0 if ok else 1


def cmd_consolidate(args):
    """Dedup + versioned conflict resolution (fully offline, no API needed)."""
    manager, _ = _build_manager(args)
    if not hasattr(manager, "consolidate_memories"):
        print("ℹ️  In this memory mode consolidation happens automatically via key "
              "overwrite on write, so no explicit consolidate is needed.")
        return 0
    report = manager.consolidate_memories(resolve_conflicts=not args.no_conflict)
    print("\n===== Memory Consolidation Report =====")
    print(f"Count before consolidation: {report['initial_count']}")
    print(f"Duplicates removed: {report['duplicates_removed']}")
    print(f"Conflicts resolved: {len(report['conflicts_resolved'])}")
    for c in report["conflicts_resolved"]:
        print(f"  ⚔️  Attribute \"{c['attribute']}\": kept \"{c['kept']}\", "
              f"superseded {c['superseded']}")
    print(f"Count after consolidation: {report['final_count']}")
    return 0


def cmd_show(args):
    """Print all memories currently stored for a user (the string injected into the model context)."""
    manager, mode = _build_manager(args)
    print(f"\n===== Memories for user {args.user} (mode: {mode.value}) =====")
    print(manager.get_context_string())
    return 0


def cmd_demo(args):
    """Multi-session offline example: write -> conflict/duplicate -> consolidate -> later-session recall.

    Uses a dedicated user_id and a temporary storage directory, so it never
    touches real user data under data/.
    """
    import tempfile

    Config.MEMORY_STORAGE_DIR = args.store_path or tempfile.mkdtemp(prefix="memcli_demo_")
    Config.create_directories()
    user_id = "demo_user"
    mgr = create_memory_manager(user_id, MemoryMode.NOTES)
    mgr.verbose = False
    # Start from a clean slate so repeated demo runs don't stack up old data
    if hasattr(mgr, "clear_all_memories"):
        mgr.notes = []

    print("\n" + "=" * 62)
    print("  Multi-session user memory demo (offline, no API needed)")
    print(f"  Storage directory: {Config.MEMORY_STORAGE_DIR}")
    print("=" * 62)

    # ---- Session 1 (earlier): learning the user's preferences for the first time ----
    print("\n[Session 1 · 2024-03-01] First conversation; the agent extracts these facts:")
    mgr.add_memory("User prefers a window seat", "session_2024_03", tags=["seat_preference"])
    mgr.add_memory("User lives in Chaoyang District, Beijing", "session_2024_03", tags=["home_address"])
    mgr.add_memory("User likes Sichuan food", "session_2024_03", tags=["food_preference"])
    for n in mgr.notes:
        print(f"    + {n.content}  [{n.tags[0]}]")

    # ---- Session 2 (later): the user moves (conflict) and repeats the seat preference (duplicate) ----
    print("\n[Session 2 · 2024-09-15] The user supplies updated information:")
    mgr.add_memory("User has moved to Pudong, Shanghai", "session_2024_09", tags=["home_address"])
    mgr.add_memory("User prefers a window seat", "session_2024_09", tags=["seat_preference"])  # duplicate
    print("    + User has moved to Pudong, Shanghai  [home_address]  (conflicts with the Beijing address from session 1)")
    print("    + User prefers a window seat  [seat_preference]  (duplicate of session 1)")
    print(f"\n  {len(mgr.notes)} memories before consolidation (1 duplicate, 1 conflict)")

    # ---- Consolidation: dedup + versioned conflict resolution ----
    print("\n[Background consolidation] Running consolidate_memories(): dedup + resolve conflicts by update time")
    report = mgr.consolidate_memories(resolve_conflicts=True)
    print(f"    Duplicates removed: {report['duplicates_removed']}")
    for c in report["conflicts_resolved"]:
        print(f"    Conflict resolved: attribute \"{c['attribute']}\" kept \"{c['kept']}\", superseded {c['superseded']}")
    print(f"    {report['final_count']} memories after consolidation")

    # ---- Session 3 (later still): recalling user info in a follow-up session ----
    print("\n[Session 3 · 2025-01-20] User asks: \"Book me a flight — do you still remember where I live?\"")
    hits = mgr.search_memories("home_address")
    recalled = hits[0].content if hits else "(no relevant memory)"
    print(f"    Agent searches memories(home_address) → recalls: {recalled}")
    print("    ✅ Agent replies: I've suggested flights based on your address in Pudong, Shanghai.")
    print("       (Note: what was recalled is the latest address after conflict resolution, not the old one from session 1)")

    print("\nFinal memory snapshot:")
    print(mgr.get_context_string())
    return 0


def cmd_extract(args):
    """Extract memories from a conversation automatically -- requires an LLM API (online).

    Argument parsing and validation for this subcommand can be checked offline;
    the actual extraction calls the background memory processor, which needs an
    API key configured for the chosen provider.
    """
    provider = args.provider or Config.PROVIDER
    if not Config.get_api_key(provider):
        print(f"⚠️  extract requires an LLM API: no API key found for provider '{provider}'.")
        print("    Set the matching *_API_KEY in .env and retry (argument parsing succeeded).")
        return 2

    # Read the conversation text: --conversation may be a file path or the text itself
    import os
    text = args.conversation
    if text and os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()
    if not text:
        print("❌ Please provide conversation text or a file path via --conversation")
        return 1

    _apply_store_path(args.store_path)
    mode = MODE_MAP[args.memory_mode] if args.memory_mode else Config.MEMORY_MODE

    from background_memory_processor import BackgroundMemoryProcessor
    processor = BackgroundMemoryProcessor(
        user_id=args.user, provider=provider, model=args.model, memory_mode=mode, verbose=True
    )
    # Split the plain-text conversation into user/assistant turns for the processor to analyze
    lines = [ln for ln in text.splitlines() if ln.strip()]
    conversation = [{"role": "user" if i % 2 == 0 else "assistant", "content": ln}
                    for i, ln in enumerate(lines)]
    processor.analyze_conversation(conversation)
    print("\n✅ Extraction complete, current memories:")
    print(processor.memory_manager.get_context_string())
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="memory_cli.py",
        description="Offline user-memory CLI: add/query/update/consolidate memories, demonstrating "
                    "cross-session memory storage and conflict resolution (no API needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="subcommand")

    def add_common(p, need_mode=True):
        p.add_argument("--user", default="default_user", help="User ID (default: default_user)")
        p.add_argument("--store-path", default=None,
                       help="Memory storage directory (default: data/memories; point elsewhere to leave real data untouched)")
        if need_mode:
            p.add_argument("--memory-mode", choices=list(MODE_MAP.keys()), default=None,
                           help="Memory storage format (defaults to the MEMORY_MODE env var)")

    p_add = sub.add_parser("add", help="Write one memory (simulates a fact extracted from a session)")
    add_common(p_add)
    p_add.add_argument("--session", default="cli_session", help="Source session ID (default: cli_session)")
    p_add.add_argument("--content", required=True, help="Memory content text")
    p_add.add_argument("--tags", default=None,
                       help="Tags, comma separated; the first tag is the attribute key used for conflict resolution")
    p_add.set_defaults(func=cmd_add)

    p_query = sub.add_parser("query", help="Search memories by keyword (cross-session recall)")
    add_common(p_query)
    p_query.add_argument("--query", required=True, help="Search keyword")
    p_query.set_defaults(func=cmd_query)

    p_update = sub.add_parser("update", help="Update an existing memory by ID")
    add_common(p_update)
    p_update.add_argument("--id", required=True, help="ID of the memory to update")
    p_update.add_argument("--session", default="cli_session", help="Session ID for this update")
    p_update.add_argument("--content", required=True, help="Updated memory content")
    p_update.add_argument("--tags", default=None, help="Updated tags, comma separated")
    p_update.set_defaults(func=cmd_update)

    p_cons = sub.add_parser("consolidate", help="Dedup + versioned conflict resolution (fully offline)")
    add_common(p_cons)
    p_cons.add_argument("--no-conflict", action="store_true",
                        help="Only dedup, skip conflict resolution")
    p_cons.set_defaults(func=cmd_consolidate)

    p_show = sub.add_parser("show", help="Print all memories currently stored for a user")
    add_common(p_show)
    p_show.set_defaults(func=cmd_show)

    p_demo = sub.add_parser("demo", help="Multi-session offline example: write -> conflict/duplicate -> consolidate -> later-session recall")
    p_demo.add_argument("--store-path", default=None,
                        help="Storage directory for demo data (default: a temp directory, never touches data/)")
    p_demo.set_defaults(func=cmd_demo)

    p_ext = sub.add_parser("extract", help="Extract memories from a conversation automatically (needs an LLM API)")
    add_common(p_ext)
    p_ext.add_argument("--conversation", required=True, help="Conversation text or path to a conversation file")
    p_ext.add_argument("--provider", default=None,
                       choices=["openai", "siliconflow", "doubao", "openrouter"],
                       help="LLM provider (defaults to the PROVIDER env var)")
    p_ext.add_argument("--model", default=None, help="Model name (defaults to the provider's default model)")
    p_ext.set_defaults(func=cmd_extract)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
