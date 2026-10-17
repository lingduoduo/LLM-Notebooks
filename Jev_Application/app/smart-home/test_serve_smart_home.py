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
    monkeypatch.setattr(srv, "openai_key", "dummy-openai")
    monkeypatch.setattr(srv, "jev_base", "http://127.0.0.1:9")  # discard port, never used
    monkeypatch.setattr(srv, "OPENAI_HOST", "upstream.invalid")
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


# ---------- provider: OpenAI only (upstream stubbed, no network) ----------

import base64
import io
import re


def _serve(monkeypatch):
    monkeypatch.setattr(srv, "openai_key", "sk-test-openai")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._buf = io.BytesIO(body)

    def read(self, n=-1):
        return self._buf.read(n)


class _FakeConn:
    def __init__(self, record, status, body):
        self.record, self.status, self.body = record, status, body

    def request(self, method, path, body=None, headers=None):
        self.record.update(method=method, path=path, body=body, headers=headers or {})

    def getresponse(self):
        return _FakeResp(self.status, self.body)

    def close(self):
        self.record["closed"] = True


def test_no_stepfun_references_and_openai_endpoints():
    root = Path(__file__).resolve().parent
    for name in ("serve_smart_home.py", "smart_home_playground.html", "README.md"):
        text = (root / name).read_text("utf-8")
        if name == "README.md":
            # The labelled historical section keeps the original StepFun measurements as a record;
            # only the live instructions must be StepFun-free.
            live, sep, history = text.partition("## 历史实测：阶跃 StepFun 版")
            assert sep and "该后端已移除" in history
            text = live + history.split("\n## ", 1)[-1] if "\n## " in history else live
        assert not re.search(r"stepfun|阶跃|stepaudio|step-\d", text, re.I), name
    assert srv.OPENAI_HOST == "api.openai.com"
    assert srv.ASR_PATH == "/v1/audio/transcriptions"
    assert srv.CHAT_PATH == "/v1/chat/completions"


def test_chat_proxies_to_openai(monkeypatch):
    seen = {}

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}],
                               "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()

    def fake_urlopen(req, timeout=None):
        seen.update(url=req.full_url, auth=req.get_header("Authorization"), body=json.loads(req.data))
        return Resp()

    monkeypatch.setattr(srv.urllib.request, "urlopen", fake_urlopen)
    httpd = _serve(monkeypatch)
    try:
        port = httpd.server_address[1]
        status, _, raw = request(port, "POST", "/api/chat", json.dumps({
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10**9, "purpose": "compare"}),
            {"Content-Type": "application/json"})
    finally:
        httpd.shutdown(); httpd.server_close()
    out = json.loads(raw)
    assert status == 200 and out["ok"] and out["content"] == "ok"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test-openai"
    assert seen["body"]["model"] == srv.compare_model
    assert seen["body"]["max_tokens"] == srv.MAX_CHAT_TOKENS


def test_asr_proxies_pcm_as_wav_to_openai_and_streams_sse(monkeypatch):
    record = {}
    sse = (b'data: {"type":"transcript.text.delta","delta":"\xe6\x89\x93\xe5\xbc\x80"}\n\n'
           b'data: {"type":"transcript.text.done","text":"\xe6\x89\x93\xe5\xbc\x80\xe7\x81\xaf"}\n\n')
    monkeypatch.setattr(srv, "_openai_connection", lambda timeout: _FakeConn(record, 200, sse))
    monkeypatch.setattr(srv, "asr_model", "gpt-4o-mini-transcribe")
    httpd = _serve(monkeypatch)
    pcm = b"\x01\x00" * 1600
    try:
        port = httpd.server_address[1]
        status, headers, raw = request(port, "POST", "/api/asr", json.dumps({
            "audio_b64": base64.b64encode(pcm).decode(), "hotwords": ["客厅", "灯光"],
            "format": {"type": "pcm", "codec": "pcm_s16le", "rate": 16000, "bits": 16, "channel": 1}}),
            {"Content-Type": "application/json"})
    finally:
        httpd.shutdown(); httpd.server_close()
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    assert raw == sse
    assert record["path"] == "/v1/audio/transcriptions"
    assert record["headers"]["Authorization"] == "Bearer sk-test-openai"
    assert record["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    body = record["body"]
    assert b'name="model"\r\n\r\ngpt-4o-mini-transcribe' in body
    assert b'name="stream"\r\n\r\ntrue' in body
    assert "客厅".encode() in body
    assert b"RIFF" in body and b"WAVE" in body and pcm in body
    assert record.get("closed")


def test_asr_non_streaming_model_emits_done_event(monkeypatch):
    record = {}
    monkeypatch.setattr(srv, "_openai_connection",
                        lambda timeout: _FakeConn(record, 200, json.dumps({"text": "关灯"}).encode()))
    monkeypatch.setattr(srv, "asr_model", "whisper-1")
    httpd = _serve(monkeypatch)
    try:
        port = httpd.server_address[1]
        status, _, raw = request(port, "POST", "/api/asr", json.dumps({
            "audio_b64": base64.b64encode(b"\x00\x00" * 800).decode()}), {"Content-Type": "application/json"})
    finally:
        httpd.shutdown(); httpd.server_close()
    assert status == 200
    assert b'name="stream"' not in record["body"]
    line = raw.decode().strip()
    assert line.startswith("data: ")
    assert json.loads(line[6:]) == {"type": "transcript.text.done", "text": "关灯"}


def test_openai_routes_require_key(monkeypatch):
    httpd = _serve(monkeypatch)
    monkeypatch.setattr(srv, "openai_key", "")
    monkeypatch.setattr(srv, "_openai_connection", lambda timeout: pytest.fail("network used"))
    monkeypatch.setattr(srv.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    try:
        port = httpd.server_address[1]
        hdr = {"Content-Type": "application/json"}
        s1, _, _ = request(port, "POST", "/api/asr", json.dumps({"audio_b64": "AAAA"}), hdr)
        s2, _, _ = request(port, "POST", "/api/chat", BODY, hdr)
        s3, _, health = request(port, "GET", "/api/health")
    finally:
        httpd.shutdown(); httpd.server_close()
    assert s1 == 503 and s2 == 503 and s3 == 200
    h = json.loads(health)
    assert h["asr"] is False and h["chat"] is False


def test_pcm_to_wav_header():
    wav = srv._pcm_to_wav(b"\x00\x01" * 10, 16000, 1, 16)
    assert wav[:4] == b"RIFF" and wav[8:16] == b"WAVEfmt " and len(wav) == 44 + 20
