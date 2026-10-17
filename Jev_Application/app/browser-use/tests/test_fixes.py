"""Offline regression tests. Browser Harness and h2 are stubbed; no Chrome, no model calls."""

import re
import sys
import types
from pathlib import Path

import pytest

harness = types.ModuleType("browser_harness")
harness.admin = types.ModuleType("browser_harness.admin")
harness.admin.ensure_daemon = lambda: None
harness.helpers = types.ModuleType("browser_harness.helpers")
harness.helpers.cdp = lambda *args, **kwargs: pytest.fail("unexpected CDP call")
sys.modules.setdefault("browser_harness", harness)
sys.modules.setdefault("browser_harness.admin", harness.admin)
sys.modules.setdefault("browser_harness.helpers", harness.helpers)
sys.modules.setdefault("h2", types.ModuleType("h2"))

from jev_ultrafast import agent as agent_module  # noqa: E402
from jev_ultrafast import browser as browser_module  # noqa: E402
from jev_ultrafast.browser import Browser, StalePage  # noqa: E402
from jev_ultrafast.model import choose, validate_choice  # noqa: E402

INVALID = "Invalid TypeSafe response"


# --- model validation -------------------------------------------------------------------------

def test_validate_choice_accepts_valid_answer():
    answer = {"choice": "a", "probabilities": {"a": 0.9, "b": 0.1}, "confidence": 0.9}
    assert validate_choice(answer, {"a": 1, "b": 2}) is answer


@pytest.mark.parametrize(
    "answer",
    [
        {"choice": "a", "probabilities": [0.9, 0.1], "confidence": 0.9},
        {"choice": "a", "probabilities": "a", "confidence": 0.9},
        {"choice": "a", "confidence": 0.9},
        {"choice": "a", "probabilities": {"a": 0.9, "b": 0.1}},
        {"probabilities": {"a": 0.9, "b": 0.1}, "confidence": 0.9},
        ["a"],
        None,
    ],
)
def test_validate_choice_rejects_malformed(answer):
    with pytest.raises(ValueError, match=INVALID):
        validate_choice(answer, {"a": 1, "b": 2})


PAGE = {
    "url": "http://x/",
    "title": "t",
    "text": "",
    "actions": [{"id": "e1", "kind": "click", "node": 1, "role": "button", "label": "Go", "value": ""}],
}
OPERATION = {"choice": "DONE", "probabilities": {"CLICK": 0.0, "DONE": 1.0, "BLOCKED": 0.0}, "confidence": 1.0}


@pytest.mark.parametrize(
    "result",
    [
        {"model": "m"},
        {"answers": [], "model": "m"},
        {"answers": {"operation": OPERATION}},
        ["not", "a", "dict"],
    ],
)
def test_choose_rejects_malformed_response(monkeypatch, result):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr("jev_ultrafast.model.post_json", lambda *a: result)
    with pytest.raises(ValueError, match=INVALID):
        choose(PAGE, "goal", [])


def test_choose_accepts_valid_response(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr("jev_ultrafast.model.post_json", lambda *a: {"answers": {"operation": OPERATION}, "model": "m"})
    assert choose(PAGE, "goal", [])["choice"] == "DONE"


# --- agent: covered/stale targets must not loop ------------------------------------------------

class FakeBrowser:
    def __init__(self, url):
        self.closed = False

    def observe(self, screenshot=True):
        return {**PAGE, "fingerprint": "f", "screenshot": ""}

    def fresh(self, page, action=None):
        return True

    def act(self, action, page, text=None):
        raise StalePage("Target changed or is covered. Observe again.")

    def close(self):
        self.closed = True


def decision():
    return {
        "choice": "e1", "operation": "CLICK", "target": "1", "confidence": 0.9,
        "probabilities": {"e1": 1.0}, "latency_ms": 1, "usage": {}, "request": {"big": "x" * 100},
    }


def test_consecutive_stale_ticks_block(monkeypatch):
    monkeypatch.setattr(agent_module, "Browser", FakeBrowser)
    calls = []
    monkeypatch.setattr(agent_module, "choose", lambda *a: calls.append(1) or decision())
    agent = agent_module.Agent("http://x/", "goal")
    states = list(agent.run())
    assert states[-1]["status"] == "blocked"
    assert "consecutive" in states[-1]["reason"]
    assert len(calls) == agent_module.MAX_STALE_TICKS
    assert states[-1]["history"] == []


def test_stale_counter_resets_after_success(monkeypatch):
    monkeypatch.setattr(agent_module, "Browser", FakeBrowser)
    monkeypatch.setattr(agent_module, "choose", lambda *a: decision())
    agent = agent_module.Agent("http://x/", "goal")
    outcomes = iter([StalePage("x"), StalePage("x"), None, StalePage("x"), StalePage("x")])

    def act(action, page, text=None):
        outcome = next(outcomes)
        if outcome:
            raise outcome

    agent.browser.act = act
    for _ in range(5):
        state = agent.command("tick")
    assert state["status"] == "ready"
    assert len(state["history"]) == 1


def test_snapshot_keeps_request_only_on_latest_decision(monkeypatch):
    monkeypatch.setattr(agent_module, "Browser", FakeBrowser)
    monkeypatch.setattr(agent_module, "choose", lambda *a: decision())
    agent = agent_module.Agent("http://x/", "goal")
    for _ in range(2):
        agent.command("tick")
    decisions = agent.snapshot()["decisions"]
    assert len(decisions) == 2
    assert "request" not in decisions[0] and "request" in decisions[1]
    assert "request" in agent.state["decisions"][0]  # internal state untouched


# --- browser: setup cleanup and evaluation errors --------------------------------------------

class Recorder:
    def __init__(self, fail_on=None, ready="complete", evaluation=None):
        self.calls, self.fail_on, self.ready, self.evaluation = [], fail_on, ready, evaluation

    def __call__(self, method, **params):
        self.calls.append(method)
        if method == self.fail_on:
            raise RuntimeError("boom")
        if method == "Target.createTarget":
            return {"targetId": "T"}
        if method == "Target.attachToTarget":
            return {"sessionId": "S"}
        if method == "Runtime.evaluate":
            if params["expression"] == "document.readyState":
                return {"result": {"value": self.ready}}
            return self.evaluation
        return {}


@pytest.mark.parametrize("method", ["Target.attachToTarget", "Emulation.setDeviceMetricsOverride", "Page.navigate"])
def test_browser_setup_failure_closes_tab(monkeypatch, method):
    cdp = Recorder(fail_on=method)
    monkeypatch.setattr(browser_module, "cdp", cdp)
    with pytest.raises(RuntimeError, match="boom"):
        Browser("http://x/")
    assert cdp.calls[-1] == "Target.closeTarget"


def test_browser_load_deadline_raises_and_closes(monkeypatch):
    cdp = Recorder(ready="loading")
    monkeypatch.setattr(browser_module, "cdp", cdp)
    clock = iter(range(0, 1000, 5))
    monkeypatch.setattr(browser_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(browser_module.time, "sleep", lambda s: None)
    with pytest.raises(TimeoutError):
        Browser("http://x/")
    assert cdp.calls[-1] == "Target.closeTarget"


def details(description):
    return {"exceptionDetails": {"text": "Uncaught", "exception": {"description": description}}}


def make_browser(monkeypatch, evaluation):
    monkeypatch.setattr(browser_module, "cdp", Recorder(evaluation=evaluation))
    return Browser("http://x/")


def test_script_error_raises_runtime_error_with_description(monkeypatch):
    browser = make_browser(monkeypatch, details("TypeError: x is not a function"))
    with pytest.raises(RuntimeError, match="TypeError: x is not a function") as error:
        browser.evaluate("x()")
    assert not isinstance(error.value, StalePage)
    with pytest.raises(RuntimeError, match="TypeError"):
        browser_module.browser_operation({"operation": "observe", "session": "S"})


@pytest.mark.parametrize(
    "description", ["Error: Execution context was destroyed.", "Error: Inspected target navigated or closed"]
)
def test_navigation_error_raises_stale_page(monkeypatch, description):
    browser = make_browser(monkeypatch, details(description))
    with pytest.raises(StalePage):
        browser.evaluate("1")
    with pytest.raises(StalePage):
        browser_module.browser_operation({"operation": "observe", "session": "S"})


def test_select_interruption_still_raises_runtime_error(monkeypatch):
    make_browser(monkeypatch, details("Error: Execution context was destroyed."))
    action = {"id": "e1", "kind": "select", "node": 1, "value": "x"}
    with pytest.raises(RuntimeError, match="Dropdown execution was interrupted"):
        browser_module.browser_operation({"operation": "act", "session": "S", "action": action})


# --- fixture XSS ------------------------------------------------------------------------------

FIXTURE = (Path(__file__).resolve().parents[1] / "jev_ultrafast" / "static" / "fixture.html").read_text()


def test_fixture_does_not_interpolate_raw_query_or_category():
    assert not re.search(r"\$\{\s*(query|category)\b", FIXTURE)
    assert "${escapeHtml(query" in FIXTURE and "${escapeHtml(category)}" in FIXTURE


def test_fixture_escape_function_covers_html_metacharacters():
    match = re.search(r"const escapeHtml = .*?;\n", FIXTURE, re.S)
    assert match
    for char, entity in {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}.items():
        assert f'"{char}": "{entity}"' in match.group(0) or f"'{char}': \"{entity}\"" in match.group(0)
