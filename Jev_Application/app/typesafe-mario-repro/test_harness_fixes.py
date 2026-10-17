"""Offline regression tests for the reproduction-layer fixes.

The emulator and the upstream ``typesafe_mario`` / ``typesafe_sdk`` / ``httpx2`` packages
are not needed: minimal stand-ins are installed into ``sys.modules`` below, before any
module under test is imported.

Run:  python -m pytest test_harness_fixes.py -q
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
import types
from enum import Enum
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# --------------------------------------------------------------------------- stubs
def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


class _Action(str, Enum):
    NOOP = "NOOP"
    RIGHT = "right"
    RIGHT_JUMP = "right_jump"
    RIGHT_RUN = "right_run"
    RIGHT_RUN_JUMP = "right_run_jump"
    JUMP = "jump"
    LEFT = "left"


class _Decision:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _TypeSafePolicy:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _BaseTransport:
    pass


class _Request:
    def __init__(self, content: bytes = b"") -> None:
        self.content = content


class _Response:
    def __init__(self, status_code: int, json=None) -> None:  # noqa: A002 - httpx name
        self.status_code = status_code
        self.payload = json


class _TypeSafeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs


_module("httpx2", BaseTransport=_BaseTransport, Request=_Request, Response=_Response)
_module("typesafe_sdk", TypeSafeClient=_TypeSafeClient, RetryPolicy=lambda **k: k)
_module("typesafe_mario")
_module(
    "typesafe_mario.actions",
    Action=_Action,
    ACTION_DESCRIPTIONS={a.value: a.value for a in _Action},
    ACTION_TO_INDEX={},
    JUMP_ACTIONS=set(),
    JUMP_RELEASE_ACTION={},
)
_module(
    "typesafe_mario.policy",
    Decision=_Decision,
    TypeSafePolicy=_TypeSafePolicy,
    HeuristicPolicy=_TypeSafePolicy,
)
_module("typesafe_mario.state", MarioStateParser=type("MarioStateParser", (), {}))
_module("typesafe_mario.dashboard", DashboardCommand=type("DashboardCommand", (), {}))
_module("typesafe_mario.runner", create_mario_env=None, run_episode=None)
_module("gymnasium", make=lambda *a, **k: None)
_module("gym_super_mario_bros")
_module("gym_super_mario_bros.actions", SIMPLE_MOVEMENT=[["NOOP"]])
_module("nes_py")
_module("nes_py.wrappers", JoypadSpace=lambda env, actions: env)

import harness_ab  # noqa: E402
import repro_run  # noqa: E402
import viz_server  # noqa: E402
from attempt_memory import AttemptMemory  # noqa: E402
from mario_harness_v2 import feasibility_features, pilot_choose_v2  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _state(speed: float, *, grounded: bool = True, **terrain) -> dict:
    return {
        "player": {"horizontal_speed_px_per_frame": speed, "grounded": grounded, "x": 500},
        "terrain": terrain,
        "reaction_timing": {"action_horizon_frames": 8, "total_reaction_horizon_frames": 8},
        "episode": {"stalled_frames": 0},
        "hazard": {},
    }


def _pick(state: dict) -> str:
    state = {**state, "takeoff_window": feasibility_features(state)}
    return pilot_choose_v2(state)[0]


def _record(*, x: int, dead: bool, terminated: bool, reward: float = 0.0,
            stage_clear: bool = False) -> dict:
    return {
        "action": "right_run",
        "latency_ms": 1.0,
        "terminated": terminated,
        "reward": reward,
        "state": {
            "player": {"x": x, "y": 80, "grounded": False, "jump_phase": "falling"},
            "episode": {"dead": dead, "stage_clear": stage_clear, "best_progress": x},
            "hazard": {"jump_must_start_this_decision": False, "nearest_enemy_kind": None},
            "trajectory": {"crossing_known_gap": False},
            "terrain": {"gap_distance_tiles": 1, "obstacle_distance_tiles": None},
        },
    }


# --------------------------------------------------------------------------- 1: speed 0
@pytest.mark.parametrize("height", [1, 2])
def test_standing_still_far_from_low_obstacle_does_not_divide_by_zero(height):
    verdict = feasibility_features(
        _state(0.0, obstacle_distance_tiles=8, obstacle_height_tiles=height)
    )
    assert verdict["verdict"] == "wait"
    assert "not moving" in verdict["advice"]
    assert _pick(_state(0.0, obstacle_distance_tiles=8, obstacle_height_tiles=height)) == "right_run"


def test_moving_still_reports_frame_count():
    verdict = feasibility_features(_state(2.6, obstacle_distance_tiles=12, obstacle_height_tiles=1))
    assert verdict["verdict"] == "wait"
    assert "frames too far" in verdict["advice"]


# --------------------------------------------------------------------------- 4: gaps
def test_unclearable_wide_gap_at_lip_says_back_off():
    state = _state(2.6, gap_distance_tiles=0, gap_width_tiles_visible=6)
    verdict = feasibility_features(state)
    assert verdict["verdict"] == "unclearable"
    assert "get closer" not in verdict["advice"]
    assert not verdict["clearable"]
    assert _pick(state) not in ("right_run", "right", "right_run_jump", "right_jump")


def test_clearable_gap_at_lip_jumps_instead_of_get_closer():
    # 5 tiles = 80px fits the 82px arc from the lip, though the far edge projects past it.
    state = _state(0.5, gap_distance_tiles=0, gap_width_tiles_visible=5)
    verdict = feasibility_features(state)
    assert verdict["verdict"] == "too_late"
    assert verdict["clearable"]
    assert _pick(state) == "right_run_jump"


def test_far_clearable_gap_still_says_get_closer():
    verdict = feasibility_features(_state(2.6, gap_distance_tiles=6, gap_width_tiles_visible=3))
    assert verdict["verdict"] == "too_wide_from_here"


# --------------------------------------------------------------------------- 5: too tall
@pytest.mark.parametrize("speed", [2.6, 0.5])
@pytest.mark.parametrize("tiles", [1, 3, 10])
def test_obstacle_taller_than_arc_is_unclearable(speed, tiles):
    state = _state(speed, obstacle_distance_tiles=tiles, obstacle_height_tiles=5)
    verdict = feasibility_features(state)
    assert verdict["verdict"] == "unclearable"
    assert verdict["hazard_kind"] == "obstacle"
    assert _pick(state) not in ("right_run", "right", "right_run_jump", "right_jump")


# --------------------------------------------------------------------------- 3: outcome
def _write_log(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_summarise_marks_death_in_final_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(repro_run, "HERE", tmp_path)
    log = _write_log(
        tmp_path,
        [_record(x=100, dead=False, terminated=False), _record(x=300, dead=False, terminated=True)],
    )
    summary = repro_run.summarise(log)
    assert summary["died"] is True
    assert summary["cleared"] is False
    assert summary["ended_reason"] == "death"


def test_summarise_budget_and_clear_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(repro_run, "HERE", tmp_path)
    log = _write_log(tmp_path, [_record(x=300, dead=False, terminated=False)])
    assert repro_run.summarise(log)["ended_reason"] == "budget_exhausted"
    log = _write_log(tmp_path, [_record(x=3100, dead=False, terminated=True, reward=150.0)])
    summary = repro_run.summarise(log)
    assert (summary["cleared"], summary["died"]) == (True, False)


def test_harness_ab_counts_episode_before_recording_death(tmp_path, monkeypatch):
    log = _write_log(
        tmp_path,
        [_record(x=100, dead=False, terminated=False), _record(x=420, dead=False, terminated=True)],
    )
    monkeypatch.setattr(harness_ab, "HERE", tmp_path)
    monkeypatch.setattr(harness_ab, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(harness_ab.runner_module, "run_episode", lambda **kwargs: log)
    memory = AttemptMemory()
    out = harness_ab.run_episode(
        variant="v2", episode=0, seed=1, max_decisions=10, frames_per_decision=8, memory=memory
    )
    assert out["died"] is True and out["at_x"] == 420
    assert memory.episodes == 1
    assert [d.episode for d in memory.deaths] == [1]


# --------------------------------------------------------------------------- 2: history
def test_viz_transport_history_is_bounded():
    hub = viz_server.VizHub(fps=30.0, latency_ms=0.0, frames_per_decision=8)
    transport = viz_server.V2HubTransport(hub)
    questions = {"next_action": {"criteria": {a.value: "" for a in _Action}}}
    for i in range(50):
        state = _state(2.6, obstacle_distance_tiles=10, obstacle_height_tiles=1)
        state["player"]["x"] = i
        body = json.dumps({"state": state, "questions": questions}).encode()
        transport.handle_request(_Request(body))
    assert len(transport.requests) == 1 and len(transport.responses) == 1
    assert transport.requests[-1]["state"]["player"]["x"] == 49
    assert hub.model_input["player"]["x"] == 49


def test_short_run_transport_keeps_full_history():
    transport = repro_run.ScriptedPilotTransport()
    questions = {"next_action": {"criteria": {a.value: "" for a in _Action}}}
    body = json.dumps({"state": _state(2.6), "questions": questions}).encode()
    for _ in range(5):
        transport.handle_request(_Request(body))
    assert len(transport.requests) == 5


# --------------------------------------------------------------------------- 6: death x
def test_viz_death_records_actual_position_not_best_progress():
    hub = viz_server.VizHub(fps=30.0, latency_ms=0.0, frames_per_decision=8)
    hub.note_episode_end({"state": {"dead": True, "best_progress": 900, "x": 640}})
    assert hub.death_marks[-1]["x"] == 900  # the tally still uses best progress
    assert hub.memory.deaths[-1].at_x == 640


# --------------------------------------------------------------------------- 7: /control
class _Body:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.asked: list[int] = []

    def read(self, n: int) -> bytes:
        self.asked.append(n)
        return self.data[:n]


@pytest.mark.parametrize(
    "headers,data",
    [
        ({"Content-Length": "2"}, b"[]"),
        ({"Content-Length": "4"}, b"null"),
        ({"Content-Length": "3"}, b"\xff\xfe\xfd"),
        ({"Content-Length": "5"}, b"{nope"),
        ({"Content-Length": "-1"}, b"{}"),
        ({"Content-Length": str(10**9)}, b"{}"),
        ({"Content-Length": "abc"}, b"{}"),
        ({"Content-Length": "2", "Host": "127.0.0.1:8770", "Origin": "https://evil.example"}, b"{}"),
        ({"Content-Length": "2", "Host": "127.0.0.1:8770", "Origin": "null"}, b"{}"),
    ],
)
def test_parse_control_body_rejects(headers, data):
    body = _Body(data)
    assert viz_server.parse_control_body(headers, body.read) is None
    assert all(0 <= n <= viz_server.MAX_CONTROL_BODY for n in body.asked)


def test_parse_control_body_accepts_same_origin_and_no_origin():
    data = b'{"command": "pause"}'
    same = {"Content-Length": str(len(data)), "Host": "127.0.0.1:8770",
            "Origin": "http://127.0.0.1:8770"}
    assert viz_server.parse_control_body(same, _Body(data).read) == {"command": "pause"}
    bare = {"Content-Length": str(len(data))}
    assert viz_server.parse_control_body(bare, _Body(data).read) == {"command": "pause"}


@pytest.fixture()
def server():
    hub = viz_server.VizHub(fps=30.0, latency_ms=0.0, frames_per_decision=8)
    handler = type("TestHandler", (viz_server.Handler,), {"hub": hub})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield hub, httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(port: int, data: bytes, headers: dict | None = None) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", "/control", body=data, headers=headers or {})
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()


def test_control_endpoint_rejects_bad_requests_without_raising(server):
    hub, port = server
    assert _post(port, b"[1, 2]") == 400
    assert _post(port, b"\xff\xfe") == 400
    assert _post(port, b'{"command": "speed", "fps": "fast"}') == 400
    assert _post(port, b'{"command": "speed", "fps": [1]}') == 400
    assert _post(port, b'{"command": "quit"}', {"Origin": "https://evil.example"}) == 400
    assert hub.requested is None and hub.fps == 30.0

    assert _post(port, b'{"command": "speed", "fps": 60}', {"Origin": f"http://127.0.0.1:{port}"}) == 200
    assert hub.fps == 60.0
    assert _post(port, b'{"command": "pause"}') == 200
    assert hub.paused.is_set()
