import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Jev_LocalModel.serve import make_handler


class FakeClient:
    """Stands in for LayaClient so routing is tested without weights or torch."""

    model_name = "laya-multilingual"
    device = "cpu"

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions, *, model=None):
        self.calls.append((state, questions, model))
        if model is not None and model != self.model_name:
            raise ValueError("model mismatch")
        if state == "boom":
            raise RuntimeError("inference crashed")
        return {"model": self.model_name, "answers": {"q": {"type": "noul", "noul": 0.7}},
                "usage": {"input_tokens": 3, "output_tokens": 0}}


class LocalServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        handler = make_handler(self.client)
        handler.log_message = lambda *args: None
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, path, body=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_homepage_explains_api_and_links_to_health(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "text/html")
            page = response.read().decode("utf-8")
        self.assertIn('href="/healthz"', page)
        self.assertIn('href="/v1/models"', page)
        self.assertIn("laya-multilingual", page)
        self.assertIn("POST /v1/systemone", page)
        self.assertIn("curl", page)

    def test_get_inference_route_explains_required_method(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(self.base + "/v1/systemone", timeout=5)
        with raised.exception as response:
            self.assertEqual(response.code, 405)
            self.assertEqual(response.headers["Allow"], "POST")
            body = json.loads(response.read())
        self.assertEqual(body["error"]["type"], "method_not_allowed")
        self.assertIn("POST", body["error"]["message"])

    def test_health_and_model_listing(self):
        self.assertEqual(self.request("/healthz"), (200, {"status": "ok", "model": "laya-multilingual", "device": "cpu"}))
        status, body = self.request("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([m["name"] for m in body["models"]], ["laya-multilingual"])

    def test_systemone_forwards_jev_shaped_request(self):
        questions = {"q": {"type": "noul", "instructions": "紧急？", "criteria": {"true": "是", "false": "否"}}}
        status, body = self.request("/v1/systemone", {"model": "laya-multilingual", "state": {"t": "x"}, "questions": questions})
        self.assertEqual(status, 200)
        self.assertEqual(body["answers"]["q"]["noul"], 0.7)
        self.assertEqual(self.client.calls, [({"t": "x"}, questions, "laya-multilingual")])

    def test_invalid_requests_return_400(self):
        status, body = self.request("/v1/systemone", {"state": "x", "questions": {}})
        self.assertEqual((status, body["error"]["type"]), (400, "invalid_request"))
        self.assertEqual(self.request("/v1/systemone", raw=b"not json")[0], 400)
        self.assertEqual(self.request("/v1/systemone", raw=b"[]")[0], 400)
        status, body = self.request("/v1/systemone", {"model": "other", "state": "x", "questions": {"q": {}}})
        self.assertEqual((status, body["error"]["message"]), (400, "model mismatch"))

    def test_invalid_encoding_and_model_fields_return_400(self):
        self.assertEqual(self.request("/v1/systemone", raw=b'\xff')[0], 400)
        for model in (None, "", [], {}):
            with self.subTest(model=model):
                status, body = self.request("/v1/systemone", {
                    "model": model, "state": "x", "questions": {"q": {}}})
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["type"], "invalid_request")

    def test_inference_errors_do_not_leak_details(self):
        status, body = self.request("/v1/systemone", {"model": "laya-multilingual", "state": "boom", "questions": {"q": {}}})
        self.assertEqual(status, 500)
        self.assertEqual(body["error"], {"message": "inference failed", "type": "server_error"})

    def test_unknown_routes_return_404(self):
        self.assertEqual(self.request("/nope")[0], 404)
        self.assertEqual(self.request("/healthz", {"x": 1})[0], 404)


if __name__ == "__main__":
    unittest.main()
