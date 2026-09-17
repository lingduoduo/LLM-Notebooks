"""Regression tests for defects found while auditing Experiment 10-1.

Each test pins one observable behaviour that was previously wrong:
the judge's position swap, the gate that was meant to police it, the
tool-execution evidence gate, multi-transfer turns, and the Skill
document-load accounting.
"""

import json
import random
import types

import evaluation
import judge_comparison
import package_comparison
from orchestrator import MultiRoleOrchestrator
from skill_orchestrator import SkillOrchestrator


# --------------------------------------------------------------- judge swap

def test_position_swap_flags_are_opposites_for_the_two_repeats():
    """Both repeats of a pair must show the arms in opposite positions."""
    rng = random.Random(101)
    for _ in range(50):
        flags = judge_comparison.position_swap_flags(rng)
        assert len(flags) == 2
        assert flags[0] != flags[1], flags


def test_position_swap_flags_still_randomise_which_arm_leads():
    rng = random.Random(101)
    leading = {judge_comparison.position_swap_flags(rng)[0] for _ in range(50)}
    assert leading == {False, True}


# ----------------------------------------------------- judge gate in packaging

def _judge_payload(orders_per_pair):
    return {
        "paired_n": 30,
        "judge_receipt_count": 60,
        "unique_response_ids": 60,
        "parse_complete": True,
        "pairs": [
            {"judgments": [{"shown_order": list(order)} for order in orders]}
            for orders in orders_per_pair
        ],
    }


def test_judge_gate_rejects_pairs_that_were_never_position_swapped():
    swapped = (["transfer", "skill"], ["skill", "transfer"])
    unswapped = (["transfer", "skill"], ["transfer", "skill"])
    assert package_comparison.judge_position_swapped(_judge_payload([swapped] * 30))
    assert not package_comparison.judge_position_swapped(
        _judge_payload([swapped] * 29 + [unswapped])
    )


def test_judge_gate_rejects_the_retained_campaign_that_lost_its_swap():
    archived = json.loads(
        (package_comparison.ROOT / "validation" / "comparison" / "runs"
         / "exp10-1-qwen35flash-20260809-v2" / "judge.json").read_text(encoding="utf-8")
    )
    assert not package_comparison.judge_position_swapped(archived)


# ------------------------------------------------- tool execution evidence

def test_tool_execution_evidence_fails_when_the_required_tool_itself_failed():
    """An unrelated successful tool result must not vouch for a failed tool."""
    history = [
        {"role": "user", "content": "find the figures"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {
                "name": "web_search", "arguments": json.dumps({"query": "nev sales"})}},
            {"id": "call-2", "type": "function", "function": {
                "name": "transfer_to_agent",
                "arguments": json.dumps({"target_role": "writing", "reason": "draft"})}},
        ]},
        {"role": "tool", "tool_call_id": "call-1",
         "content": "Tool web_search call failed: Tavily request failed"},
        {"role": "tool", "tool_call_id": "call-2", "content": "Handed off to writing."},
    ]
    score = evaluation.evaluate_task(
        "done", history, kind="complex", spec={"required_tools": ["web_search"]}
    )
    assert score["dimensions"]["tool_execution_evidence"] == 0
    assert not score["pass"]


def test_tool_execution_evidence_passes_on_a_real_result():
    history = [
        {"role": "user", "content": "find the figures"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {
                "name": "web_search", "arguments": json.dumps({"query": "nev sales"})}},
        ]},
        {"role": "tool", "tool_call_id": "call-1",
         "content": "2021 3.521m 2022 6.887m 2023 9.495m https://example.test/a"},
    ]
    score = evaluation.evaluate_task(
        "done", history, kind="complex", spec={"required_tools": ["web_search"]}
    )
    assert score["dimensions"]["tool_execution_evidence"] == 1


def test_tool_execution_evidence_survives_a_retry_after_one_failure():
    history = [
        {"role": "user", "content": "find the figures"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {
                "name": "web_search", "arguments": json.dumps({"query": "a"})}},
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": "Tool web_search call failed: boom"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-2", "type": "function", "function": {
                "name": "web_search", "arguments": json.dumps({"query": "b"})}},
        ]},
        {"role": "tool", "tool_call_id": "call-2", "content": "https://example.test/a 9.495m"},
    ]
    score = evaluation.evaluate_task(
        "done", history, kind="complex", spec={"required_tools": ["web_search"]}
    )
    assert score["dimensions"]["tool_execution_evidence"] == 1


# ----------------------------------------------------------- stub LLM client

class _Message:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def _tool_call(call_id, name, args):
    return types.SimpleNamespace(
        id=call_id, type="function",
        function=types.SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class _Usage:
    def model_dump(self, mode="json"):
        return {"prompt_tokens": 10, "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0}}


class _Response:
    def __init__(self, message, index):
        self.choices = [types.SimpleNamespace(message=message)]
        self.usage = _Usage()
        self.id = f"chatcmpl-{index}"
        self.model = "stub"

    def model_dump(self, mode="json"):
        return {"id": self.id}


class _StubClient:
    """Replays a scripted sequence of assistant messages."""

    def __init__(self, script):
        self._script = list(script)
        self._index = 0
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **_kwargs):
        message = (self._script[self._index] if self._index < len(self._script)
                   else _Message("final"))
        self._index += 1
        return _Response(message, self._index)


# ------------------------------------------------------- multi-transfer turn

def test_only_the_first_transfer_in_a_turn_is_applied_and_the_rest_are_refused():
    script = [
        _Message("", [
            _tool_call("t1", "transfer_to_agent", {"target_role": "research", "reason": "r"}),
            _tool_call("t2", "transfer_to_agent", {"target_role": "writing", "reason": "w"}),
        ]),
        _Message("done"),
    ]
    agent = MultiRoleOrchestrator(client=_StubClient(script), verbose=False)
    agent.run("task")

    assert [(item.from_role, item.to_role) for item in agent.handoffs] == [("triage", "research")]
    assert agent.current_role == "research"
    # The activity log must not claim a handoff the orchestrator refused.
    assert [item for item in agent.activity if item[1] == "transfer"] == [
        ("triage", "transfer", "research")
    ]
    # The refused call must be told the truth, not "Handed off".
    tool_messages = [item["content"] for item in agent.history if item["role"] == "tool"]
    assert tool_messages[0].startswith("Handed off to research")
    assert "already handing off" in tool_messages[1]
    assert "Handed off to writing" not in tool_messages[1]


def test_a_single_transfer_per_turn_still_works():
    script = [
        _Message("", [_tool_call("t1", "transfer_to_agent",
                                 {"target_role": "research", "reason": "r"})]),
        _Message("", [_tool_call("t2", "transfer_to_agent",
                                 {"target_role": "writing", "reason": "w"})]),
        _Message("done"),
    ]
    agent = MultiRoleOrchestrator(client=_StubClient(script), verbose=False)
    agent.run("task")
    assert agent.handoff_chain_str() == "triage → research → writing"


# ------------------------------------------------- skill document accounting

def test_skill_summary_reports_loads_and_refused_reloads_not_a_dead_cache():
    script = [
        _Message("", [_tool_call("s1", "load_skill", {"name": "triage"})]),
        _Message("", [_tool_call("s2", "load_skill", {"name": "data_analysis"})]),
        _Message("", [_tool_call("s3", "load_skill", {"name": "data_analysis"})]),
        _Message("", [_tool_call("s4", "load_skill", {"name": "data_analysis"})]),
        _Message("done"),
    ]
    agent = SkillOrchestrator(client=_StubClient(script), verbose=False)
    agent.run("task")
    summary = agent.summary()
    assert summary["skill_documents_loaded"] == 2
    assert summary["skill_reloads_refused"] == 2
    assert len(summary["skill_load_latency_seconds"]) == 2
    # The always-zero cache counters are gone.
    assert "skill_cache_hits" not in summary
    assert "skill_cache_misses" not in summary


# --------------------------------------------------------------- translation

def test_translation_map_has_no_chinese_left_in_its_values():
    import re
    import translate_validation as tv
    translations = tv.load_translations()
    assert translations, "translation map should not be empty"
    untranslated = {key: value for key, value in translations.items()
                    if re.search(r"[一-鿿]", value)}
    assert not untranslated, list(untranslated.values())[:3]


def test_canonical_system_prompts_resolve_to_repository_english():
    import translate_validation as tv
    from roles import ROLES
    for marker, role in tv.ROLE_PROMPT_MARKERS.items():
        assert tv.canonical_system_prompt(marker + " ...") == ROLES[role].system_prompt
    assert tv.canonical_system_prompt(tv.SKILL_PROMPT_MARKER + " ...")
    assert tv.canonical_system_prompt("You are the intake triage role") is None


def test_official_bundle_leaves_only_verbatim_source_excerpts_untranslated():
    """No model output in the official run may be left in Chinese."""
    import translate_validation as tv
    translator = tv.Translator(tv.load_translations())
    official = [
        "runs/exp10-1-kimi-k2.5-tavily-receipts-20260730-v3/evidence.json",
        "runs/exp10-1-kimi-k2.5-tavily-receipts-20260730-v3/tavily_receipts.json",
        "runs/exp10-1-kimi-k2.5-tavily-receipts-20260730-v3/moonshot_receipts.json",
    ]
    for name in official:
        translator.reset()
        translator.walk(json.loads((tv.SOURCE_ROOT / name).read_text(encoding="utf-8")))
        leftover = [item for item in translator.untranslated
                    if item.get("kind") != "source_excerpt"]
        assert not leftover, (name, leftover[:2])
        assert translator.translated > 0


# ------------------------------------------------- model-backed map filling

def test_batches_respect_the_budget_and_isolate_oversized_units():
    import fill_translation_map as fill
    units = ["中" * 100, "中" * 100, "中" * 5000, "中" * 100]
    batches = fill.build_batches(units, batch_chars=300)
    assert [len(batch) for batch in batches] == [2, 1, 1]
    assert batches[1] == ["中" * 5000], "an oversized unit gets its own batch"


def test_translate_batch_maps_results_back_to_their_source_strings():
    import fill_translation_map as fill
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        answer = {key: f"english for {key}" for key in payload}
        message = types.SimpleNamespace(content=json.dumps(answer))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    args = types.SimpleNamespace(model="stub", request_timeout=1)
    batch = ["第一段", "第二段"]
    result = fill.translate_batch(client, args, batch)
    assert result == {"第一段": "english for 0", "第二段": "english for 1"}
    assert captured["response_format"] == {"type": "json_object"}
    assert "faithfully" in captured["messages"][0]["content"]


def test_translate_batch_drops_blank_or_missing_answers():
    import fill_translation_map as fill

    def create(**kwargs):
        message = types.SimpleNamespace(content=json.dumps({"0": "  ", "1": "ok"}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    args = types.SimpleNamespace(model="stub", request_timeout=1)
    result = fill.translate_batch(client, args, ["甲", "乙", "丙"])
    assert result == {"乙": "ok"}, "blank and absent answers must not enter the map"


def test_residual_queue_shrinks_as_the_map_grows():
    """Dropping an entry puts exactly its source back on the queue."""
    import translate_validation as tv
    translations = tv.load_translations()
    sources = tv.load_sources()
    baseline = tv.residual_units(translations)
    # A source that really occurs in the bundles: removing its id from the map
    # must make that exact string reappear as outstanding work.
    victim = next(
        (key for key, text in sources.items()
         if text not in baseline and tv.residual_units(
             {k: v for k, v in translations.items() if k != key}) != baseline),
        None,
    )
    assert victim is not None, "expected at least one translated string in use"
    without = tv.residual_units(
        {k: v for k, v in translations.items() if k != victim})
    assert sources[victim] in without
    assert len(without) == len(baseline) + 1
    assert tv.residual_units(translations) == baseline


def test_historical_summaries_are_english_in_place():
    """The two unreferenced summaries carry no Chinese and are not truncated."""
    import re
    import translate_validation as tv
    for stem in ("exp10-1-kimi-k2.5-tavily-20260730-v1",
                 "exp10-1-kimi-k2.5-tavily-20260730-v2"):
        english = tv.SOURCE_ROOT / f"{stem}.json"
        text = english.read_text(encoding="utf-8")
        assert not re.search(r"[一-鿿]", text)
        # Shape intact, so this is a translation and not a truncation.  The
        # status itself is part of the record (v1 really is "incomplete") and is
        # deliberately not asserted to a value.
        document = json.loads(text)
        assert document["status"]
        assert document["handoff_chain"]
        assert document["final_answer"]
        assert document["history"]
        assert document["acceptance_gates"]
