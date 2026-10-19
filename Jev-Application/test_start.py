"""start.py: menu file server, CSRF guard, .env parsing, spawning and stop ports.

Run from Jev_Application/:  python -m pytest -q test_start.py
"""
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import start  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def menu(tmp_path, monkeypatch):
    """Run the real menu server on a spare port over a temp chapter tree."""
    (tmp_path / "app/demo").mkdir(parents=True)
    (tmp_path / "app/demo/index.html").write_text("<h1>ok</h1>")
    (tmp_path / "app/demo/.env").write_text("TYPESAFE_API_KEY=secret-app")
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=secret-root")
    (tmp_path / "start.py").write_text("print('menu')")
    port = free_port()
    monkeypatch.setattr(start, "HERE", tmp_path)
    monkeypatch.setattr(start, "APP_ROOT", (tmp_path / "app").resolve())
    monkeypatch.setattr(start, "MENU_PORT", port)
    monkeypatch.setattr(start, "RUNNABLES", {"echo": ("Echo", "echo '<b>hi</b>'; date +%s%N", "tip", "app/demo/index.html")})
    monkeypatch.setattr(start, "RUN_LOCKS", {"echo": threading.Lock()})
    monkeypatch.setattr(start, "RESULTS", {})
    monkeypatch.setattr(start, "alive", lambda port: False)
    threading.Thread(target=start.serve_menu, daemon=True).start()
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    return port


def get(port, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, ""


@pytest.mark.parametrize("path", [
    "/app/../.env",          # root .env via traversal
    "/app/../start.py",      # anything outside app/
    "/app/demo/.env",        # per-project .env inside app/
    "/app/%2e%2e/.env",      # encoded traversal stays a literal name -> missing
    "/app/demo/missing.html",
])
def test_file_server_refuses_secrets_and_escapes(menu, path):
    status, body = get(menu, path)
    assert status == 404
    assert "secret" not in body


def test_file_server_still_serves_app_files(menu):
    assert get(menu, "/app/demo/index.html") == (200, "<h1>ok</h1>")


def test_run_rejects_cross_site_requests(menu):
    assert get(menu, "/run/echo", {"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert get(menu, "/run/echo", {"Host": "evil.example:80"})[0] == 403
    assert start.RESULTS == {}


def test_run_escapes_output_and_rerun_reruns(menu):
    status, first = get(menu, "/run/echo")
    assert status == 200 and "&lt;b&gt;hi&lt;/b&gt;" in first and "<b>hi</b>" not in first
    assert get(menu, "/run/echo")[1] == first                  # cached
    time.sleep(0.01)
    assert get(menu, "/run/echo?rerun=1")[1] != first           # actually re-ran
    assert "/run/echo?rerun=1" in first


def test_load_env_strips_export_and_quotes(tmp_path, monkeypatch):
    for key in ("A_KEY", "B_KEY", "C_KEY", "D_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("D_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text('A_KEY="quoted"\nexport B_KEY=\'single\'\n# C_KEY=comment\nC_KEY = plain \nD_KEY=file\n')
    start.load_env(env)
    assert os.environ["A_KEY"] == "quoted"
    assert os.environ["B_KEY"] == "single"
    assert os.environ["C_KEY"] == "plain"
    assert os.environ["D_KEY"] == "from-shell"   # existing values win


def test_spawn_skips_missing_directory_and_command(tmp_path, capsys):
    assert start.spawn("x", ["node"], tmp_path / "missing") is False
    assert start.spawn("y", ["no-such-binary-xyz"], tmp_path) is False
    assert "跳过" in capsys.readouterr().out


def test_spawned_services_survive_terminal_ctrl_c(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "LOGS", tmp_path / "logs")
    assert start.spawn("sleeper", [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path)
    proc = start.PROCS.pop("sleeper")
    try:
        assert os.getpgid(proc.pid) != os.getpgid(0)   # own session -> no shared SIGINT
    finally:
        proc.kill()
        proc.wait()


def test_stop_ports_cover_every_service_menu_and_mario():
    ports = start.chapter_ports()
    assert {p for _, _, p, *_ in start.SERVICES} <= set(ports)
    assert start.MENU_PORT in ports and start.MARIO_PORT in ports


def test_background_server_pattern_does_not_hang_capture(tmp_path):
    # Same shape as the browser-smoke command: the backgrounded server must be the
    # killed PID and must not hold the captured stdout pipe open.
    cmd = f'(cd {tmp_path} && exec sleep 30) >/dev/null 2>&1 & DPID=$!; sleep 0.2; kill $DPID; true'
    t0 = time.time()
    subprocess.run(["bash", "-c", cmd], capture_output=True, timeout=10)
    assert time.time() - t0 < 5
    assert "(cd app/browser-use && exec" in start.RUNNABLES["browser-smoke"][1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_menu_checks_services_concurrently_and_preserves_order(monkeypatch):
    # A barrier makes serial probing fail deterministically, without timing assertions.
    barrier = threading.Barrier(len(start.SERVICES))

    def probe(port):
        barrier.wait(timeout=2)
        return port == start.SERVICES[0][2]

    monkeypatch.setattr(start, "alive", probe)
    monkeypatch.setattr(start, "mario_python", lambda: None)
    page = start.menu_page()
    positions = [page.index(title) for _, title, *_ in start.SERVICES]
    assert positions == sorted(positions)
    assert page.count('class="dot on"') == 4  # first service + three replays
    assert page.count('class="dot off"') == len(start.SERVICES) - 1


def test_runnable_supports_python_path_with_spaces(menu, tmp_path, monkeypatch):
    executable = tmp_path / "python executable"
    executable.symlink_to(sys.executable)
    monkeypatch.setattr(start.sys, "executable", str(executable))
    monkeypatch.setitem(start.RUNNABLES, "echo", (
        "Python", '\"$PY\" -c \'print("python-path-ok")\'', "tip", "app/demo/index.html"))
    status, body = get(menu, "/run/echo")
    assert status == 200
    assert "[exit 0" in body
    assert "python-path-ok" in body
