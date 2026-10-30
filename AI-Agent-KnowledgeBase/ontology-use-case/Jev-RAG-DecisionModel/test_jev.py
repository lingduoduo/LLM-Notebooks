import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import jev
from tree import Node


def nodes(n, prefix="n"):
    return [Node(f"{prefix}{i}", f"desc {i}") for i in range(n)]


class FakeTransport:
    def __init__(self, cost=0.0):
        self.bodies = []
        self.cost = cost

    def __call__(self, body):
        self.bodies.append(body)
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.9}
            else:
                opts = list(q["criteria"])
                answers[name] = {"type": "choice", "choice": opts[0], "confidence": 1.0,
                                 "probabilities": {o: 1 / len(opts) for o in opts}}
        return {"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0, "cost": self.cost}}


class FakeResp:
    def __init__(self, payload, status=200):
        self.status = status
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class RequestTests(unittest.TestCase):
    def test_binary_request_shares_state(self):
        cands = [Node("a", "alpha"), Node("b", "beta"), Node("c", "gamma")]
        body = jev.build_binary_request("q", cands, "Is candidate {n} relevant?")
        self.assertEqual(set(body["questions"]), {"c1", "c2", "c3"})
        for d in cands:
            self.assertIn(d.description, body["state"]["candidates"])
        self.assertIn("1. alpha", body["state"]["candidates"])
        self.assertIn("3. gamma", body["state"]["candidates"])
        for q in body["questions"].values():
            self.assertEqual(q["type"], "noul")
            self.assertIn("true", q["criteria"])
            self.assertIn("false", q["criteria"])
        self.assertIn("candidate 2 ", body["questions"]["c2"]["instructions"])

    def test_choice_request_keys_are_ids(self):
        cands = [Node("a", "alpha"), Node("b", "beta")]
        body = jev.build_choice_request("q", cands, "Which?")
        q = body["questions"]["most_relevant"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(set(q["criteria"]), {"a", "b"})

    def test_split_at_max_per_request(self):
        t = FakeTransport()
        j = jev.JevJudge(t, max_per_request=50)
        out = j.binary("q", nodes(120), "Is candidate {n} relevant?")
        self.assertEqual(sorted(len(b["questions"]) for b in t.bodies), [20, 50, 50])
        self.assertEqual(len(out), 120)
        self.assertTrue(all(b["model"] == jev.MODEL for b in t.bodies))
        for b in t.bodies:
            self.assertIn("c1", b["questions"])

    def test_choice_never_splits(self):
        t = FakeTransport()
        j = jev.JevJudge(t, max_per_request=5)
        out = j.choice("q", nodes(7), "Which?")
        self.assertEqual(len(out), 7)
        self.assertEqual(len(t.bodies), 1)
        self.assertEqual(len(t.bodies[0]["questions"]["most_relevant"]["criteria"]), 7)
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_flat_request_fits_one_context(self):
        cands = [Node(f"id{i}", ("Candidate concept — Category > Sub-category (e.g. " + "x" * 120)[:120])
                 for i in range(73)]
        body = {"model": jev.MODEL,
                **jev.build_binary_request("Subject: a room\nTags: t", cands,
                                           "Does this room involve Candidate {n}?")}
        size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        self.assertLess(size, 20_000)
        crit = body["questions"]["c1"]["criteria"]
        self.assertIn("negated", crit["false"])
        self.assertIn("unrelated", body["state"]["rule"])


class ParseTests(unittest.TestCase):
    def test_parse_binary(self):
        resp = {"answers": {"c1": {"type": "noul", "noul": 0.96}}}
        self.assertEqual(jev.parse_binary(resp, [Node("a", "x")]), {"a": 0.96})

    def test_parse_choice_missing_option_is_zero(self):
        resp = {"answers": {"most_relevant": {"type": "choice", "choice": "a",
                                              "probabilities": {"a": 1.0}}}}
        out = jev.parse_choice(resp, [Node("a", "x"), Node("b", "y")])
        self.assertEqual(out, {"a": 1.0, "b": 0.0})

    def test_parse_missing_key_raises_jev_error(self):
        with self.assertRaises(jev.JevError) as cm:
            jev.parse_binary({"answers": {}}, [Node("a", "x")])
        self.assertIn("c1", str(cm.exception))
        with self.assertRaises(jev.JevError):
            jev.parse_choice({"answers": {}}, [Node("a", "x")])

    def test_null_missing_or_out_of_range_noul_raises(self):
        for ans in ({"type": "noul", "noul": None}, {"type": "noul"},
                    {"type": "noul", "noul": 1.5}, {"type": "noul", "noul": -0.1},
                    {"type": "noul", "noul": "0.5"}, {"type": "choice", "noul": 0.5}, 0.5):
            with self.subTest(ans=ans), self.assertRaises(jev.JevError) as cm:
                jev.parse_binary({"answers": {"c1": ans}}, [Node("a", "x")])
            self.assertIn("c1", str(cm.exception))

    def test_missing_or_bad_probabilities_raises(self):
        for ans in ({"type": "choice", "choice": "a"},
                    {"type": "choice", "probabilities": None},
                    {"type": "choice", "probabilities": [1.0]},
                    {"type": "noul", "probabilities": {"a": 1.0}}):
            with self.subTest(ans=ans), self.assertRaises(jev.JevError):
                jev.parse_choice({"answers": {"most_relevant": ans}}, [Node("a", "x")])

    def test_error_body_raises_jev_error(self):
        resp = {"error": {"code": 400, "message": "bad"}}
        with self.assertRaises(jev.JevError) as cm:
            jev.parse_binary(resp, [Node("a", "x")])
        self.assertIn("bad", str(cm.exception))


class HttpTests(unittest.TestCase):
    def test_http_non_200_raises_jev_error(self):
        err = urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(
            json.dumps({"error": {"code": 401, "message": "bad key"}}).encode()))
        with mock.patch("urllib.request.urlopen", side_effect=err), \
             mock.patch("time.sleep") as sl:
            with self.assertRaises(jev.JevError) as cm:
                jev.HttpTransport("k")({"a": 1})
        self.assertIn("401: bad key", str(cm.exception))
        sl.assert_not_called()

    def test_http_retries_429_then_succeeds(self):
        err = urllib.error.HTTPError("u", 429, "slow", {}, io.BytesIO(b"{}"))
        ok = FakeResp({"answers": {}})
        with mock.patch("urllib.request.urlopen", side_effect=[err, err, ok]) as uo, \
             mock.patch("time.sleep") as sl:
            out = jev.HttpTransport("k")({"q": "Café"})
        self.assertEqual(out, {"answers": {}})
        self.assertEqual([c.args[0] for c in sl.call_args_list], [1, 2])
        req = uo.call_args.args[0]
        self.assertEqual(req.get_header("Authorization"), "Bearer k")
        self.assertIn("Café".encode(), req.data)

    def test_http_retries_5xx_then_succeeds(self):
        err = urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"oops"))
        ok = FakeResp({"answers": {}})
        with mock.patch("urllib.request.urlopen", side_effect=[err, ok]), \
             mock.patch("time.sleep") as sl:
            out = jev.HttpTransport("k")({})
        self.assertEqual(out, {"answers": {}})
        self.assertEqual([c.args[0] for c in sl.call_args_list], [1])

    def test_http_retries_timeout_and_connection_errors(self):
        class TimeoutResp(FakeResp):
            def read(self):
                raise TimeoutError("read timed out")
        ok = FakeResp({"answers": {}})
        side = [TimeoutResp({}), ConnectionResetError("reset"),
                http.client.IncompleteRead(b""), ok]
        with mock.patch("urllib.request.urlopen", side_effect=side) as uo, \
             mock.patch("time.sleep") as sl:
            out = jev.HttpTransport("k")({})
        self.assertEqual(out, {"answers": {}})
        self.assertEqual(uo.call_count, 4)
        self.assertEqual([c.args[0] for c in sl.call_args_list], [1, 2, 4])

    def test_http_invalid_json_raises_jev_error(self):
        class BadResp(FakeResp):
            def read(self):
                return b"<html>not json</html>"
        with mock.patch("urllib.request.urlopen", return_value=BadResp({})), \
             mock.patch("time.sleep"):
            with self.assertRaises(jev.JevError) as cm:
                jev.HttpTransport("k")({})
        self.assertIn("invalid JSON", str(cm.exception))

    def test_http_gives_up_after_three_retries(self):
        err = urllib.error.URLError("down")
        with mock.patch("urllib.request.urlopen", side_effect=err) as uo, \
             mock.patch("time.sleep") as sl:
            with self.assertRaises(jev.JevError):
                jev.HttpTransport("k")({})
        self.assertEqual(uo.call_count, 4)
        self.assertEqual([c.args[0] for c in sl.call_args_list], [1, 2, 4])


class CacheTests(unittest.TestCase):
    def test_cache_hit_skips_inner(self):
        with tempfile.TemporaryDirectory() as d:
            inner = FakeTransport()
            c = jev.CachedTransport(inner, Path(d))
            body = {"model": "m", "state": "s", "questions": {}}
            r1 = c(body)
            r2 = c(dict(body))
            self.assertEqual(r1, r2)
            self.assertEqual(len(inner.bodies), 1)
            self.assertEqual((c.hits, c.misses), (1, 1))

    def test_cache_key_ignores_dict_order_and_keeps_unicode(self):
        with tempfile.TemporaryDirectory() as d:
            inner = FakeTransport()
            c = jev.CachedTransport(inner, Path(d))
            c({"state": {"query": "Café — 夜"}, "model": "m", "questions": {}})
            c({"questions": {}, "model": "m", "state": {"query": "Café — 夜"}})
            self.assertEqual((c.hits, c.misses), (1, 1))
            req = jev.build_binary_request("Café — 夜", [Node("a", "x")], "{n}")
            raw = json.dumps(req, ensure_ascii=False).encode()
            self.assertIn("Café".encode(), raw)

    def test_error_responses_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def inner(b):
                calls.append(1)
                return {"error": {"code": 500, "message": "x"}}
            c = jev.CachedTransport(inner, Path(d))
            c({"a": 1}); c({"a": 1})
            self.assertEqual(len(calls), 2)

    def test_unparseable_200_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def inner(b):
                calls.append(1)
                return {"answers": {"c1": {"type": "noul", "noul": None}}}
            c = jev.CachedTransport(inner, Path(d))
            j = jev.JevJudge(c)
            for _ in range(2):
                with self.assertRaises(jev.JevError):
                    j.binary("q", [Node("a", "x")], "{n}")
            self.assertEqual(len(calls), 2)
            self.assertEqual(list(Path(d).glob("*")), [])
            self.assertEqual((c.hits, c.misses), (0, 2))

    def test_valid_response_is_cached(self):
        with tempfile.TemporaryDirectory() as d:
            c = jev.CachedTransport(FakeTransport(), Path(d))
            jev.JevJudge(c).binary("q", [Node("a", "x")], "{n}")
            self.assertEqual(len(list(Path(d).glob("*.json"))), 1)


class CostTests(unittest.TestCase):
    def test_cost_accumulates(self):
        t = FakeTransport(cost=0.001)
        j = jev.JevJudge(t)
        j.binary("q", nodes(2), "{n}")
        t.cost = 0.002
        j.choice("q", nodes(2), "Which?")
        self.assertAlmostEqual(j.cost_usd, 0.003)
        self.assertEqual(j.requests, 2)


if __name__ == "__main__":
    unittest.main()
