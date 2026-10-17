"""Offline tests for the smart-home proxy: hostile requests are rejected before any
upstream call (upstream URLs point at an unroutable stub and must never be reached)."""
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import serve_smart_home as srv


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setattr(srv, "jev_key", "dummy-jev")
    monkeypatch.setattr(srv, "stepfun_key", "dummy-step")
    monkeypatch.setattr(srv, "jev_base", "http://127.0.0.1:9")  # discard port, never used
    monkeypatch.setattr(srv, "STEP_HOST", "upstream.invalid")
    calls = []
    for name in ("_proxy_jev", "_proxy_asr", "_proxy_chat"):
        monkeypatch.setattr(srv.Handler, name,
                            lambda self, n=name: (calls.append(n), self._send_json({"stub": n}))[1])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1], calls
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


BODY = json.dumps({"messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize("path", ["/api/jev", "/api/asr", "/api/chat"])
def test_cross_origin_post_rejected(server, path):
    port, calls = server
    status, headers, _ = request(port, "POST", path, BODY, {
        "Content-Type": "application/json", "Origin": "https://evil.example"})
    assert status == 403
    assert "Access-Control-Allow-Origin" not in headers
    assert calls == []


@pytest.mark.parametrize("ctype", ["text/plain", "application/x-www-form-urlencoded", None])
def test_non_json_post_rejected(server, ctype):
    port, calls = server
    headers = {"Content-Type": ctype} if ctype else {}
    status, resp_headers, _ = request(port, "POST", "/api/chat", BODY, headers)
    assert status == 415
    assert "Access-Control-Allow-Origin" not in resp_headers
    assert calls == []


def test_same_origin_json_reaches_handler(server):
    port, calls = server
    status, headers, _ = request(port, "POST", "/api/chat", BODY, {
        "Content-Type": "application/json; charset=utf-8",
        "Origin": f"http://127.0.0.1:{port}"})
    assert status == 200 and calls == ["_proxy_chat"]
    assert "Access-Control-Allow-Origin" not in headers


def test_no_cors_on_get_and_no_preflight(server):
    port, _ = server
    for path in ("/", "/api/health"):
        status, headers, _ = request(port, "GET", path)
        assert status == 200
        assert "Access-Control-Allow-Origin" not in headers
    status, headers, _ = request(port, "OPTIONS", "/api/chat", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert status != 204
    assert "Access-Control-Allow-Origin" not in headers


def test_max_tokens_clamped():
    assert srv._clamp_max_tokens(10**9) == srv.MAX_CHAT_TOKENS == 2048
    assert srv._clamp_max_tokens(400) == 400
    assert srv._clamp_max_tokens(0) == 1
    assert srv._clamp_max_tokens("999999") == 500
    assert srv._clamp_max_tokens(True) == 500


def test_html_uses_relative_api_urls():
    html = (Path(__file__).resolve().parent / "smart_home_playground.html").read_text("utf-8")
    assert "fetch('/api/jev'" in html and "fetch('/api/chat'" in html and "fetch('/api/asr'" in html
    assert "fetch('http" not in html and 'fetch("http' not in html
