"""Serve one preloaded Laya checkpoint on a Jev-compatible local HTTP route."""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlsplit

from .client import LayaClient


def make_handler(client: LayaClient) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LayaLocal/0.1"

        def _json(self, status: int, body: dict[str, Any]) -> None:
            self._respond(status, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          "application/json; charset=utf-8")

        def _respond(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if status == 405:
                self.send_header("Allow", "POST")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                example = json.dumps({
                    "model": client.model_name,
                    "state": "I was charged twice.",
                    "questions": {"department": {
                        "type": "choice", "instructions": "Which department should handle this?",
                        "criteria": ["billing", "technical", "other"],
                    }},
                }, ensure_ascii=False, indent=2)
                # Shell quoting keeps custom model names safe in the copyable command.
                command = (f"curl http://127.0.0.1:{self.server.server_port}/v1/systemone "
                           "-H 'Content-Type: application/json' -d " + shlex.quote(example))
                page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laya local API</title></head>
<body><main><h1>Laya local API</h1>
<p>Loaded model: <strong>{html.escape(client.model_name)}</strong></p>
<p><a href="/healthz">Check server health</a> · <a href="/v1/models">List models</a></p>
<h2>Run inference</h2><p>Send JSON to <code>POST /v1/systemone</code>.
Run this example in a terminal:</p>
<pre style="white-space:pre-wrap;overflow-wrap:anywhere">{html.escape(command)}</pre>
</main></body></html>"""
                self._respond(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/v1/systemone":
                self._json(405, {"error": {
                    "message": "Use POST /v1/systemone with a JSON body containing model, state, and questions. See / for an example.",
                    "type": "method_not_allowed",
                }})
                return
            if path == "/healthz":
                self._json(200, {"status": "ok", "model": client.model_name, "device": client.device})
                return
            if path == "/v1/models":
                self._json(
                    200,
                    {
                        "models": [
                            {
                                "name": client.model_name,
                                "description": "Local Laya typed-decision checkpoint",
                                "release_date": "2026-09-20",
                            }
                        ]
                    },
                )
                return
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self._json(404, {"error": {"message": "not found", "type": "not_found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4 * 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 4 MiB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                if "state" not in body or "questions" not in body or "model" not in body:
                    raise ValueError("required fields: model, state, questions")
                if not isinstance(body["model"], str) or not body["model"].strip():
                    raise ValueError("model must be a non-empty string")
                result = client.system_one(body["state"], body["questions"], model=body["model"])
                self._json(200, result)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                self._json(400, {"error": {"message": str(exc), "type": "invalid_request"}})
            except Exception:
                self.log_error("Laya inference failed")
                self._json(500, {"error": {"message": "inference failed", "type": "server_error"}})

        def log_message(self, fmt: str, *args: Any) -> None:
            print("laya-api: " + fmt % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("LAYA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", "8811")))
    parser.add_argument("--model-dir", default=os.environ.get("LAYA_MODEL_DIR"))
    parser.add_argument("--model-name", default=os.environ.get("LAYA_MODEL_NAME"))
    parser.add_argument("--device", default=os.environ.get("LAYA_DEVICE"))
    args = parser.parse_args()

    # Load once at startup so request latency excludes checkpoint loading.
    client = LayaClient(args.model_dir, model_name=args.model_name, device=args.device)
    server = HTTPServer((args.host, args.port), make_handler(client))
    print(f"Laya ready: model={client.model_name} device={client.device} http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
