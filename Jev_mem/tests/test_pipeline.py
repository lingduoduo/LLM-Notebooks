"""Standalone pipeline regressions; no network or model downloads."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from Jev_mem.pipeline.data import load_dataset, synthetic_dataset
from Jev_mem.pipeline.providers import (
    LexicalEncoder,
    OfflineCompletion,
    parse_judge_score,
)
from Jev_mem.pipeline.memory import Memory, RetrievalConfig
from Jev_mem.pipeline.experiment import run_experiment


class ProviderTests(unittest.TestCase):
    def test_judge_rejects_invalid_scores(self):
        for raw in (
            "bad",
            '{"score":7}',
            '{"score":"NaN"}',
            '{"score":true}',
            '{"score":null}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_judge_score(raw)
        self.assertEqual(parse_judge_score('```json\n{"score":0.8}\n```'), 0.8)

    def test_encoder_is_stable_multilingual_and_empty_safe(self):
        a, b = LexicalEncoder(), LexicalEncoder()
        vectors = a.encode(["花生过敏", "花生", ""])
        self.assertEqual(vectors, b.encode(["花生过敏", "花生", ""]))
        self.assertEqual(len(vectors[0]), a.dimension)
        self.assertGreater(sum(x * y for x, y in zip(vectors[0], vectors[1])), 0)
        self.assertTrue(all(math.isfinite(x) for v in vectors for x in v))


class DataTests(unittest.TestCase):
    def test_all_distractors_are_used_at_large_scale(self):
        data = synthetic_dataset(384)
        contents = " ".join(t["content"] for t in data["observations"])
        for text in ("小李", "表哥", "小张", "西湖", "美式", "信用卡"):
            self.assertIn(text, contents)
        stamps = [o["timestamp"] for o in data["observations"]]
        self.assertEqual(stamps, sorted(stamps))

    def test_locomo_preserves_speaker_timestamp_and_evidence(self):
        sample = [
            {
                "sample_id": "sample-1",
                "conversation": {
                    "session_1_date_time": "1:30 pm on 8 May, 2023",
                    "session_1": [
                        {"speaker": "Alice", "text": "I like tea", "dia_id": "D1:1"}
                    ],
                },
                "qa": [
                    {
                        "question": "Who likes tea?",
                        "answer": "Alice",
                        "category": 4,
                        "evidence": ["D1:1"],
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "locomo.json"
            p.write_text(json.dumps(sample))
            data = load_dataset(p)
        self.assertEqual(data["observations"][0]["metadata"]["speaker"], "Alice")
        self.assertEqual(data["observations"][0]["metadata"]["dia_id"], "D1:1")
        self.assertEqual(data["observations"][0]["timestamp"], "2023-05-08T13:30:00")
        self.assertEqual(data["questions"][0]["evidence"], ["D1:1"])


class MemoryTests(unittest.TestCase):
    def test_build_retrieve_persist_reload_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = Memory(audit_path=Path(tmp) / "audit.jsonl")
            memory.build(
                [
                    {
                        "content": "Alice is allergic to peanuts",
                        "timestamp": "2026-01-01",
                        "metadata": {"entities": ["Alice"], "dia_id": "D1:1"},
                    },
                    {
                        "content": "Alice changed her trip to Wednesday",
                        "timestamp": "2026-01-02",
                        "metadata": {"entities": ["Alice"]},
                    },
                ]
            )
            result = memory.retrieve("What is Alice allergic to?")
            self.assertIn("peanuts", result["context"])
            self.assertEqual(len(memory.nodes), 2)
            self.assertTrue(any(e["kind"] == "temporal" for e in memory.edges))
            memory.save(Path(tmp) / "store")
            loaded = Memory.load(Path(tmp) / "store")
            self.assertEqual(
                memory.retrieve("Alice")["context"], loaded.retrieve("Alice")["context"]
            )
            self.assertTrue((Path(tmp) / "audit.jsonl").read_text().strip())

    def test_bad_input_does_not_partially_build(self):
        memory = Memory()
        with self.assertRaises(ValueError):
            memory.build([{"content": "valid"}, {"content": ""}])
        self.assertEqual(memory.nodes, [])

    def test_bounded_retrieval_and_empty_graph(self):
        memory = Memory(
            config=RetrievalConfig(
                max_nodes=3, max_depth=1, graph_budget=2, anchor_count=2
            )
        )
        self.assertEqual(memory.retrieve("anything")["evidence"], [])
        memory.build([{"content": f"Alice meeting {i}"} for i in range(20)])
        result = memory.retrieve("Alice meetings")
        self.assertLessEqual(len(result["evidence"]), 3)
        self.assertLessEqual(result["diagnostics"]["expanded_edges"], 2)

    def test_reload_rejects_incompatible_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = Memory()
            memory.build([{"content": "hello"}])
            memory.save(tmp)
            with self.assertRaisesRegex(ValueError, "encoder"):
                Memory.load(tmp, encoder=LexicalEncoder(64))


class ExperimentTests(unittest.TestCase):
    def test_four_arms_write_separate_usage_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = synthetic_dataset(48)
            run = run_experiment(data, Path(tmp), mode="offline")
            rows = [
                json.loads(s) for s in (run / "answers.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 4 * len(data["questions"]))
            self.assertEqual({r["arm"] for r in rows}, {"A", "B", "C", "D"})
            self.assertTrue(all(r["usage"]["prompt_tokens"] is None for r in rows))
            self.assertTrue(all(r["status"] == "ok" for r in rows))
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["mode"], "offline")
            self.assertEqual(summary["score_metric"], "lexical_f1")
            self.assertTrue((run / "report.svg").is_file())
            self.assertTrue((run / "memory" / "memory.json").is_file())
            other = run_experiment(data, Path(tmp), mode="offline")
            self.assertNotEqual(other, run)

    def test_api_failure_is_recorded_as_error_not_zero_score(self):
        class Broken(OfflineCompletion):
            def complete(self, *args, **kwargs):
                raise RuntimeError("provider unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            run = run_experiment(
                synthetic_dataset(48), Path(tmp), mode="offline", answerer=Broken()
            )
            rows = [
                json.loads(s) for s in (run / "answers.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(r["status"] == "answer_error" for r in rows))
            self.assertTrue(all(r["score"] is None for r in rows))


class LiveAdapterTests(unittest.TestCase):
    def test_chat_and_embedding_protocol_and_usage(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        from Jev_mem.pipeline.providers import APIEncoder, ChatCompletion

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                if self.path.endswith("/embeddings"):
                    response = {
                        "data": [
                            {"index": i, "embedding": [float(i + 1), 1.0]}
                            for i in reversed(range(len(body["input"])))
                        ],
                        "usage": {"prompt_tokens": 5, "total_tokens": 5},
                    }
                else:
                    response = {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"score":0.8}'
                                    if body["response_format"]["type"] == "json_object"
                                    else "tea"
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 17,
                            "completion_tokens": 3,
                            "total_tokens": 20,
                        },
                    }
                raw = json.dumps(response).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}/v1"
            chat = ChatCompletion("test", base, "secret")
            completion = chat.complete("question")
            self.assertEqual(completion.prompt_tokens, 17)
            self.assertEqual(completion.total_tokens, 20)
            self.assertEqual(
                parse_judge_score(chat.complete("judge", json_mode=True).text), 0.8
            )
            embedding = APIEncoder("embed", base, "secret")
            self.assertEqual(embedding.encode(["a", "b"]), [[1.0, 1.0], [2.0, 1.0]])
            self.assertEqual(embedding.dimension, 2)
            self.assertEqual(requests[0][1]["model"], "test")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_live_accounting_is_attributed_to_each_call(self):
        from Jev_mem.pipeline.providers import Completion, LocalDecisions

        class Recorded:
            source = "test-live"

            def __init__(self):
                self.count = 0

            def complete(self, prompt, *, json_mode=False):
                self.count += 1
                return Completion(
                    '{"score":0.75}' if json_mode else "answer",
                    self.count,
                    2,
                    self.count + 2,
                    source=self.source,
                )

        answerer, judge = Recorded(), Recorded()
        data = synthetic_dataset(48)
        data["questions"] = data["questions"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            run = run_experiment(
                data,
                Path(tmp),
                mode="live",
                answerer=answerer,
                judge=judge,
                decisions=LocalDecisions(),
                manifest={"api_key": "never-save-this"},
            )
            rows = [
                json.loads(s)
                for s in (run / "evaluations.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["usage"]["prompt_tokens"] for r in rows], [1, 2, 3, 4])
            self.assertEqual(
                [r["judge_usage"]["prompt_tokens"] for r in rows], [1, 2, 3, 4]
            )
            self.assertTrue(all(r["score"] == 0.75 for r in rows))
            self.assertNotIn("never-save-this", (run / "manifest.json").read_text())

    def test_judge_failure_retains_answer_and_null_score(self):
        from Jev_mem.pipeline.providers import Completion, LocalDecisions

        class InvalidJudge:
            def complete(self, *args, **kwargs):
                return Completion("not JSON", 10, 2, 12)

        data = synthetic_dataset(48)
        data["questions"] = data["questions"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            run = run_experiment(
                data,
                Path(tmp),
                mode="live",
                answerer=OfflineCompletion(),
                judge=InvalidJudge(),
                decisions=LocalDecisions(),
            )
            rows = [
                json.loads(s)
                for s in (run / "evaluations.jsonl").read_text().splitlines()
            ]
            self.assertTrue(
                all(
                    r["answer"] is not None
                    and r["score"] is None
                    and r["status"] == "judge_error"
                    for r in rows
                )
            )
            summary = json.loads((run / "summary.json").read_text())
            self.assertTrue(
                all(
                    v["scored"] == 0 and v["mean_score"] is None
                    for v in summary["arms"].values()
                )
            )


class RunIsolationTests(unittest.TestCase):
    def test_provider_usage_contains_only_current_run(self):
        from Jev_mem.pipeline.providers import LocalDecisions

        provider = LocalDecisions()
        data = synthetic_dataset(48)
        data["questions"] = data["questions"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            first = run_experiment(data, tmp, decisions=provider)
            second = run_experiment(data, tmp, decisions=provider)
            a = json.loads((first / "provider-usage.json").read_text())
            b = json.loads((second / "provider-usage.json").read_text())
            self.assertEqual(len(a["decision_calls"]), len(b["decision_calls"]))

    def test_memory_owns_embedding_vectors(self):
        class ReusedEncoder(LexicalEncoder):
            def encode(self, texts):
                self.last = super().encode(texts)
                return self.last

        encoder = ReusedEncoder()
        memory = Memory(encoder=encoder)
        memory.build([{"content": "Alice likes tea"}])
        expected = memory.nodes[0]["vector"][:]
        encoder.last[0][0] += 100
        self.assertEqual(memory.nodes[0]["vector"], expected)


class SecurityAndReproducibilityTests(unittest.TestCase):
    def test_api_redirect_does_not_forward_bearer(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        from Jev_mem.pipeline.providers import ChatCompletion

        seen = []

        class Receiver(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                seen.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        receiver = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)

        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(302)
                self.send_header("Content-Length", "0")
                self.send_header(
                    "Location", f"http://127.0.0.1:{receiver.server_port}/capture"
                )
                self.end_headers()

        redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        threads = [
            threading.Thread(target=s.serve_forever, daemon=True)
            for s in (receiver, redirect)
        ]
        for thread in threads:
            thread.start()
        try:
            client = ChatCompletion(
                "test", f"http://127.0.0.1:{redirect.server_port}", "secret"
            )
            with self.assertRaises((ValueError, RuntimeError)):
                client.complete("question")
            self.assertEqual(seen, [])
        finally:
            for server in (receiver, redirect):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_manifest_records_public_provider_endpoints(self):
        from Jev_mem.pipeline.providers import LocalDecisions

        class NamedOffline(OfflineCompletion):
            model = "same-model"
            base_url = "https://answer.example/v1"

        data = synthetic_dataset(48)
        data["questions"] = data["questions"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            run = run_experiment(
                data, tmp, answerer=NamedOffline(), decisions=LocalDecisions()
            )
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(
                manifest["providers"]["answer"]["base_url"], "https://answer.example/v1"
            )
            self.assertEqual(manifest["providers"]["answer"]["model"], "same-model")

    def test_locomo_image_caption_and_adversarial_reference(self):
        sample = {
            "conversation": {
                "session_1_date_time": "1:30 pm on 8 May, 2023",
                "session_1": [
                    {
                        "speaker": "Alice",
                        "text": "",
                        "img_url": "https://example/image",
                        "blip_caption": "Alice at a cafe",
                        "dia_id": "D1:1",
                    }
                ],
            },
            "qa": [
                {
                    "question": "What secret drink?",
                    "answer": "bait",
                    "adversarial_answer": "Not mentioned",
                    "category": 5,
                    "evidence": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "locomo.json"
            path.write_text(json.dumps([sample]))
            data = load_dataset(path)
            self.assertIn("Alice at a cafe", data["observations"][0]["content"])
            self.assertEqual(data["questions"][0]["gold"], "Not mentioned")


class JevAdapterTests(unittest.TestCase):
    def test_sdk_noul_decisions_and_usage_contract(self):
        try:
            from typesafe_sdk import SystemOneResponse, Noul
        except ImportError:
            self.skipTest("Optional TypeSafe SDK is not installed")
        from unittest.mock import patch
        from Jev_mem.pipeline.providers import JevDecisions

        decisions = JevDecisions(api_key="test-key")
        response = SystemOneResponse.model_validate(
            {
                "model": "jev-latest",
                "usage": {"input_tokens": 13, "output_tokens": 0},
                "answers": {"semantic": {"type": "noul", "noul": 0.8}},
            }
        )
        try:
            with patch.object(
                decisions.client, "system_one", return_value=response
            ) as call:
                result = decisions.decide(
                    "routing",
                    {"question": "Alice?", "heuristics": {"semantic": 0.2}},
                    {"semantic": "Is semantic traversal useful?"},
                )
                self.assertEqual(result, {"semantic": 0.8})
                self.assertIsInstance(
                    call.call_args.kwargs["questions"]["semantic"], Noul
                )
                self.assertNotIn("heuristics", call.call_args.kwargs["state"])
                self.assertEqual(decisions.calls[-1]["usage"]["input_tokens"], 13)
            with patch.object(
                decisions.client, "system_one", side_effect=RuntimeError("private-key")
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "no silent fallback"
                ) as error:
                    decisions.decide("routing", {}, {"semantic": "Useful?"})
                self.assertNotIn("private-key", str(error.exception))
        finally:
            decisions.close()
