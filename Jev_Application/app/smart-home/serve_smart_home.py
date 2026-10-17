"""智能家居 3D 演练场本地服务：静态页面 + 三个上游 API 代理。

浏览器页面为什么要经过这层本地代理，而不是直连：
1. api.typesafe.ai 与 api.openai.com 都不保证给浏览器 CORS 头，直连会被拦；
2. API key 不能暴露在页面里——放在这台本地服务的环境变量中。

代理端点：
- POST /api/jev   → https://api.typesafe.ai/v1/systemone   （单次投机提示调用）
- POST /api/asr   → https://api.openai.com/v1/audio/transcriptions（OpenAI 语音转写；
                     浏览器上传的 16k PCM 在服务端封成 WAV，流式模型的 SSE 原样透传，
                     不支持流式的模型（whisper-1）由服务端转成同格式的单条 done 事件）
- POST /api/chat  → https://api.openai.com/v1/chat/completions（复合指令拆分 / 常识问答）
- GET  /api/health→ 报告哪些 key 已配置，页面据此决定「真实 Jev / 本地模拟」模式

运行：  python3 serve_smart_home.py --port 8810
然后打开 http://127.0.0.1:8810

环境变量：
- TYPESAFE_API_KEY   console.typesafe.ai 获取。不设则页面自动降级为本地模拟引擎。
- OPENAI_API_KEY        platform.openai.com 获取。不设则语音输入、复合指令拆分、常识问答与传统 LLM 对比不可用。
- OPENAI_ASR_MODEL      默认 gpt-4o-mini-transcribe（支持 stream=true 流式转写）；也可设 whisper-1。
- OPENAI_CHAT_MODEL     默认 gpt-4o-mini（拆指令：求快）。
- OPENAI_ANSWER_MODEL   默认 gpt-4o-mini（知识问答）。
- OPENAI_COMPARE_MODEL  默认 gpt-4o（传统 LLM 对比场景）。
"""
from __future__ import annotations

import argparse
import base64
import binascii
import http.client
import json
import os
import ssl
import struct
import time
import uuid
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML_PATH = Path(__file__).resolve().parent / "smart_home_playground.html"

JEV_HOST = "api.typesafe.ai"
JEV_PATH = "/v1/systemone"
OPENAI_HOST = "api.openai.com"  # 官方 OpenAI API
ASR_PATH = "/v1/audio/transcriptions"
CHAT_PATH = "/v1/chat/completions"

openai_key = os.environ.get("OPENAI_API_KEY", "")  # platform.openai.com 获取，语音/拆分/问答需要
jev_key = os.environ.get("TYPESAFE_API_KEY", "")
# 与官方 SDK 同名约定：指向自建/代理上游（默认官方端点），用于联调与离线测试
jev_base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
asr_model = os.environ.get("OPENAI_ASR_MODEL", "gpt-4o-mini-transcribe")
chat_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")        # 拆指令：求快
answer_model = os.environ.get("OPENAI_ANSWER_MODEL", "gpt-4o-mini")    # 知识问答
compare_model = os.environ.get("OPENAI_COMPARE_MODEL", "gpt-4o")       # 传统 LLM 对比场景
# whisper-1 不支持 stream=true；其余转写模型（gpt-4o-*-transcribe）流式返回 transcript.text.delta/done
NON_STREAMING_ASR_MODELS = {"whisper-1"}

_SSL = ssl.create_default_context()
MAX_CHAT_TOKENS = 2048  # /api/chat 的 max_tokens 上限，防止客户端任意放大付费调用


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 简短日志：方法 路径 状态 耗时
        print(f"[{time.strftime('%H:%M:%S')}] {self.command} {self.path}")

    # ---------- 基础设施 ----------

    # 页面与 API 同源（相对 URL），故不发任何 CORS 头：其他网站无法借本地代理消耗 API key

    def _cross_origin(self) -> bool:
        """带 Origin 且与 Host 不一致 → 跨站请求。"""
        origin = self.headers.get("Origin")
        if not origin:
            return False
        host = self.headers.get("Host", "")
        netloc = origin.split("://", 1)[-1].rstrip("/")
        return netloc.lower() != host.lower()

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            data = json.loads(self.rfile.read(length))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    # ---------- 路由 ----------

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/api/health":
            self._send_json({
                "jev": bool(jev_key),
                "asr": bool(openai_key),
                "chat": bool(openai_key),
                "asr_model": asr_model,
                "chat_model": chat_model,
                "answer_model": answer_model,
                "compare_model": compare_model,
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self._cross_origin():
            self.close_connection = True  # 未读请求体，关闭连接以免污染 keep-alive
            self._send_json({"error": "cross-origin request rejected"}, 403)
            return
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            # 只收 JSON：text/plain 等「简单请求」可绕过浏览器预检，必须拒绝
            self.close_connection = True
            self._send_json({"error": "Content-Type must be application/json"}, 415)
            return
        if self.path == "/api/jev":
            self._proxy_jev()
        elif self.path == "/api/asr":
            self._proxy_asr()
        elif self.path == "/api/chat":
            self._proxy_chat()
        else:
            self._send_json({"error": "not found"}, 404)

    # ---------- /api/jev：Jev systemone 单次调用 ----------

    def _proxy_jev(self) -> None:
        payload = self._read_json_body()
        if not jev_key:
            self._send_json({"ok": False, "status": 0,
                             "error": "未设置 TYPESAFE_API_KEY，服务端以模拟模式运行"}, 503)
            return
        body = json.dumps(payload, ensure_ascii=False).encode()
        req = urllib.request.Request(
            f"{jev_base}{JEV_PATH}", data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {jev_key}",
                "User-Agent": "jev-cookbook-smart-home/1.0",
            })
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read()
            status = e.code
        except Exception as e:  # 连接类错误
            self._send_json({"ok": False, "status": 0, "error": f"{type(e).__name__}: {e}"}, 502)
            return
        upstream_ms = round((time.monotonic() - t0) * 1000)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw.decode(errors="replace")[:2000]}
        self._send_json({"ok": 200 <= status < 300, "status": status,
                         "upstream_ms": upstream_ms, "body": parsed})

    # ---------- /api/asr：OpenAI 语音转写（SSE 透传给前端） ----------

    def _proxy_asr(self) -> None:
        payload = self._read_json_body()
        audio_b64 = payload.get("audio_b64") or ""
        if not audio_b64:
            self._send_json({"error": "audio_b64 缺失"}, 400)
            return
        if not openai_key:
            self._send_json({"error": "未设置 OPENAI_API_KEY，语音识别不可用"}, 503)
            return
        try:
            audio = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError):
            self._send_json({"error": "audio_b64 不是合法 base64"}, 400)
            return
        fmt = payload.get("format") or {}
        if not isinstance(fmt, dict):
            fmt = {}
        filename, mime = "audio.wav", "audio/wav"
        if (fmt.get("type") or "pcm") == "pcm":
            # OpenAI 只收文件格式：把裸 PCM s16le 封成 WAV
            audio = _pcm_to_wav(audio, int(fmt.get("rate") or 16000),
                                int(fmt.get("channel") or 1), int(fmt.get("bits") or 16))
        streaming = asr_model not in NON_STREAMING_ASR_MODELS
        fields = {"model": asr_model, "response_format": "json"}
        if streaming:
            fields["stream"] = "true"
        if isinstance(payload.get("language"), str) and payload["language"]:
            fields["language"] = payload["language"]
        hotwords = payload.get("hotwords")
        if isinstance(hotwords, list) and hotwords:
            # OpenAI 无热词参数，用 prompt 提示领域词汇
            fields["prompt"] = "、".join(str(w) for w in hotwords[:50])
        body, content_type = _multipart(fields, ("file", filename, mime, audio))

        try:
            conn = _openai_connection(timeout=60)
            conn.request("POST", ASR_PATH, body=body, headers={
                "Content-Type": content_type,
                "Accept": "text/event-stream" if streaming else "application/json",
                "Authorization": f"Bearer {openai_key}",
            })
            resp = conn.getresponse()
        except Exception as e:
            self._send_json({"error": f"ASR 上游连接失败: {type(e).__name__}: {e}"}, 502)
            return

        try:
            if resp.status != 200:
                err = resp.read().decode(errors="replace")[:500]
                self._send_json({"error": f"ASR 上游 HTTP {resp.status}", "detail": err}, 502)
                return

            # 前端按 SSE 解析 transcript.text.delta / transcript.text.done
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            if streaming:
                while True:
                    chunk = resp.read(512)
                    if not chunk:
                        break
                    self._write_chunk(chunk)
            else:
                try:
                    text = json.loads(resp.read()).get("text") or ""
                except (json.JSONDecodeError, AttributeError):
                    text = ""
                event = {"type": "transcript.text.done", "text": text}
                self._write_chunk(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # 浏览器提前断开（如用户取消）
        finally:
            conn.close()

    def _write_chunk(self, chunk: bytes) -> None:
        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
        self.wfile.flush()

    # ---------- /api/chat：复合指令拆分 / 常识问答 ----------

    def _proxy_chat(self) -> None:
        payload = self._read_json_body()
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_json({"error": "messages 缺失"}, 400)
            return
        if not openai_key:
            self._send_json({"ok": False, "status": 0, "error": "未设置 OPENAI_API_KEY"}, 503)
            return
        purpose_model = {"answer": answer_model, "compare": compare_model}
        body = json.dumps({
            "model": purpose_model.get(payload.get("purpose"), chat_model),
            "messages": messages,
            "temperature": payload.get("temperature", 0),
            "max_tokens": _clamp_max_tokens(payload.get("max_tokens", 500)),
        }).encode()
        req = urllib.request.Request(
            f"https://{OPENAI_HOST}{CHAT_PATH}", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {openai_key}"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                parsed = json.loads(resp.read())
                status = resp.status
        except urllib.error.HTTPError as e:
            self._send_json({"ok": False, "status": e.code,
                             "error": e.read().decode(errors="replace")[:500]}, 200)
            return
        except Exception as e:
            self._send_json({"ok": False, "status": 0, "error": f"{type(e).__name__}: {e}"}, 502)
            return
        content = ""
        try:
            content = parsed["choices"][0]["message"].get("content") or ""
            if not content:
                # 推理型模型可能把正文放在 reasoning_content
                content = parsed["choices"][0]["message"].get("reasoning_content") or ""
        except (KeyError, IndexError, TypeError):
            pass
        self._send_json({"ok": 200 <= status < 300, "status": status, "content": content,
                         "upstream_ms": round((time.monotonic() - t0) * 1000),
                         "model": purpose_model.get(payload.get("purpose"), chat_model),
                         "usage": parsed.get("usage")})


def _openai_connection(timeout: float) -> http.client.HTTPConnection:
    """到官方 OpenAI API 的 HTTPS 连接（测试里替换成本地桩，不走网络）。"""
    return http.client.HTTPSConnection(OPENAI_HOST, timeout=timeout, context=_SSL)


def _pcm_to_wav(pcm: bytes, rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """裸 PCM（little-endian）加 44 字节 RIFF/WAVE 头。"""
    block_align = channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * block_align, block_align, bits)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def _multipart(fields: dict, file: tuple[str, str, str, bytes]) -> tuple[bytes, str]:
    """构造 multipart/form-data 请求体：普通字段 + 一个文件。"""
    boundary = f"----smarthome{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    field, filename, mime, data = file
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                 f"Content-Type: {mime}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _clamp_max_tokens(value) -> int:
    """客户端传入的 max_tokens 限制在 [1, MAX_CHAT_TOKENS]；非法值回落到默认 500。"""
    if type(value) is not int:
        return 500
    return max(1, min(value, MAX_CHAT_TOKENS))


def make_selfsigned_cert(cert_dir: Path) -> tuple[Path, Path]:
    """openssl 生成一年期自签证书，供局域网 https（手机测麦克风）使用。"""
    cert_dir.mkdir(exist_ok=True)
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if not (cert.exists() and key.exists()):
        import subprocess
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key), "-out", str(cert), "-days", "365", "-nodes",
            "-subj", "/CN=smart-home-demo",
        ], check=True, capture_output=True)
    return cert, key


def main() -> None:
    parser = argparse.ArgumentParser(description="智能家居 3D 演练场本地服务")
    parser.add_argument("--port", type=int, default=8810)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--tls", action="store_true",
                        help="启用 HTTPS（openssl 自签证书）。局域网设备测语音输入时必须用，"
                             "浏览器里对证书警告点「高级 → 继续前往」即可")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    scheme = "http"
    if args.tls:
        cert, key = make_selfsigned_cert(Path(__file__).resolve().parent / "certs")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    host_disp = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    url = f"{scheme}://{host_disp}:{args.port}"
    print(f"智能家居演练场: {url}")
    if args.host == "0.0.0.0":
        import subprocess as _sp
        try:
            lan = _sp.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True).stdout.strip() \
                or _sp.run(["ipconfig", "getifaddr", "en1"], capture_output=True, text=True).stdout.strip()
        except Exception:
            lan = ""
        if lan:
            print(f"  局域网: {scheme}://{lan}:{args.port}")
    print(f"  Jev   真实调用: {'✅ 已配置 TYPESAFE_API_KEY' if jev_key else '⚠️ 未配置 → 页面自动用本地模拟引擎'}")
    print(f"  ASR   语音识别: {'✅ ' + asr_model if openai_key else '❌ 未配置 OPENAI_API_KEY'}")
    print(f"  Chat  拆分/问答: {'✅ ' + chat_model if openai_key else '❌ 未配置 OPENAI_API_KEY'}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
