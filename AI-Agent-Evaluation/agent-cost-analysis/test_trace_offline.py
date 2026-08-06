"""
Regression tests for offline trace parsing (Experiment 6-7 cost analysis).

Covers two crash classes found in --offline mode:
  - Tracer.from_records: trace JSON with explicit null token fields -> int(None) TypeError
  - demo.collect_offline: scenario dict missing the optional "spans" key -> KeyError
"""
import json
from pathlib import Path

import pytest

import config
import demo
from config import Pricing
from demo import DEFAULT_TRACE, dump_output
from tracer import Tracer


def _span(**overrides):
    span = {
        "step": "turn-1",
        "tool": "query_order",
        "kind": "llm",
        "prompt_tokens": 100,
        "cached_tokens": 10,
        "completion_tokens": 12,
        "tool_ctx_tokens": 50,
        "latency_s": 1.2,
    }
    span.update(overrides)
    return span


def test_from_records_tolerates_null_fields():
    """Explicit JSON nulls in numeric fields are coerced, not int(None) TypeError."""
    records = [_span(prompt_tokens=None, cached_tokens=None,
                     completion_tokens=None, tool_ctx_tokens=None, latency_s=None)]
    tr = Tracer.from_records(records, pricing=config.default_pricing())
    s = tr.spans[0]
    assert s.prompt_tokens == 0
    assert s.cached_tokens == 0
    assert s.completion_tokens == 0
    assert s.tool_ctx_tokens == -1   # A null tool context is unknown.
    assert s.latency_s == 0.0


def test_from_records_tolerates_missing_fields():
    """Minimal span dicts (only step/tool) still parse."""
    tr = Tracer.from_records([{"step": "turn-1", "tool": "query_order"}],
                             pricing=config.default_pricing())
    assert tr.spans[0].prompt_tokens == 0
    assert tr.spans[0].tool_ctx_tokens == -1


def test_from_records_keeps_real_values():
    """Normal values pass through unchanged (no coercion side effects)."""
    tr = Tracer.from_records([_span()], pricing=config.default_pricing())
    s = tr.spans[0]
    assert (s.prompt_tokens, s.cached_tokens, s.completion_tokens) == (100, 10, 12)
    assert s.tool_ctx_tokens == 50
    assert s.latency_s == 1.2


def test_from_records_keeps_zero_tool_ctx_tokens():
    """Explicit 0 means known-zero tool context, not unknown (-1)."""
    tr = Tracer.from_records([_span(tool_ctx_tokens=0)],
                             pricing=config.default_pricing())
    assert tr.spans[0].tool_ctx_tokens == 0
    assert tr.total_tool_ctx_tokens() == 0


def _write_trace(tmp_path, scenarios):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"model": "gpt-5.6-luna", "scenarios": scenarios}),
                    encoding="utf-8")
    return str(path)


def test_collect_offline_skips_scenario_without_spans(tmp_path, capsys):
    """A scenario missing 'spans' is skipped with a warning, not a KeyError crash."""
    trace = _write_trace(tmp_path, [
        {"key": "naive", "name": "A naive"},                      # no spans -> skip
        {"key": "both", "name": "B both", "spans": [_span()]},    # valid
    ])
    tracers = demo.collect_offline(["naive", "both"], config.default_pricing(), trace)
    assert [k for k, _ in tracers] == ["both"]
    assert "has no span data" in capsys.readouterr().err


def test_collect_offline_exits_when_no_usable_scenario(tmp_path):
    """When every selected scenario lacks spans, exit cleanly like the empty-trace path."""
    trace = _write_trace(tmp_path, [{"key": "naive", "name": "A naive"}])
    with pytest.raises(SystemExit):
        demo.collect_offline(["naive"], config.default_pricing(), trace)


def test_bundled_trace_is_labeled_synthetic_english():
    """The bundled trace is generated data and must never read as live usage."""
    data = json.loads(Path(DEFAULT_TRACE).read_text(encoding="utf-8"))
    assert data["provenance"] == "synthetic"
    assert data["language"] == "en"


def test_saved_live_trace_is_labeled_observed(tmp_path):
    """A trace captured from a live run is labeled as observed measurement."""
    path = tmp_path / "trace.json"
    dump_output(path, [], Pricing(0.15, 0.075, 0.6), "test-model")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provenance"] == "observed"
    assert data["language"] == "en"


def test_bundled_trace_cached_counts_are_within_prompt_counts():
    """Cached tokens are a subset of prompt tokens, so 0 <= cached <= prompt."""
    data = json.loads(Path(DEFAULT_TRACE).read_text(encoding="utf-8"))
    for sc in data["scenarios"]:
        for span in sc["spans"]:
            assert 0 <= span["cached_tokens"] <= span["prompt_tokens"], (
                f"{sc['key']}/{span['step']}"
            )


def test_bundled_trace_uncached_scenarios_have_no_cache_hits():
    """Scenarios built on a volatile prefix cannot report cached tokens."""
    data = json.loads(Path(DEFAULT_TRACE).read_text(encoding="utf-8"))
    by_key = {sc["key"]: sc for sc in data["scenarios"]}
    for key in ("naive", "compress"):
        assert all(s["cached_tokens"] == 0 for s in by_key[key]["spans"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
