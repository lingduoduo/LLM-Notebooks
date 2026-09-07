"""web_search must default invalid num_results values (null or nonnumeric strings)
instead of interrupting the agent loop with TypeError or ValueError."""
import json

import base_tools


def _search_without_network(monkeypatch, num_results):
    """Disable networking and backoff sleeps to check argument handling without crashes."""
    def _fake_post(*a, **k):
        raise RuntimeError("network disabled in test")

    monkeypatch.setattr(base_tools.requests, "post", _fake_post)
    monkeypatch.setattr(base_tools.time, "sleep", lambda *_a, **_k: None)
    return base_tools.web_search("python stock library", num_results)


def test_num_results_null_falls_back(monkeypatch):
    # Explicit JSON null bypasses .get(..., 6); fall back to 6
    args = json.loads('{"query": "python stock library", "num_results": null}')
    result = _search_without_network(monkeypatch, args.get("num_results", 6))
    assert result["success"] is False  # Networking is disabled: return failure instead of raising an exception
    assert "search failed" in result["error"]


def test_num_results_garbage_string_falls_back(monkeypatch):
    args = json.loads('{"query": "python stock library", "num_results": "five"}')
    result = _search_without_network(monkeypatch, args.get("num_results", 6))
    assert result["success"] is False
    assert "search failed" in result["error"]


def test_num_results_normal_still_clamped(monkeypatch):
    result = _search_without_network(monkeypatch, 3)
    assert result["success"] is False
    assert "search failed" in result["error"]
