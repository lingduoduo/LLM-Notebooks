import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jev
import main as cli
from judge import FakeJudge


class FailingJudge(FakeJudge):
    def binary(self, query, candidates, question):
        raise jev.JevError("503: upstream down")


class StatsTransport:
    hits, misses = 3, 4


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def test_missing_key_exits_2(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            code, _, err = run(["tag", "x"])
        self.assertEqual(code, 2)
        self.assertIn("OPENROUTER_API_KEY is not set", err)

    def test_tag_fake_prints_tags(self):
        code, out, _ = run(["tag", "bdsm dungeon", "--tags", "bdsm", "--fake"])
        self.assertEqual(code, 0)
        self.assertIn("p_bin", out)

    def test_eval_fake_writes_report(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.json"
            code, _, _ = run(["eval", "--mode", "all", "--fake", "--output", str(path)])
            self.assertEqual(code, 0)
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".md").exists())

    def test_demo_fake_runs(self):
        self.assertEqual(cli.main(["demo", "--fake"]), 0)


    def test_eval_jev_error_mid_run_writes_partial_report(self):
        built = []
        def factory(**kw):
            built.append(kw)
            return FakeJudge() if len(built) == 1 else FailingJudge()
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(cli, "make_judge_factory", return_value=(factory, StatsTransport())):
            path = Path(d) / "report.json"
            code, out, err = run(["eval", "--mode", "all", "--output", str(path)])
            self.assertEqual(code, 1)
            self.assertIn("error: 503: upstream down", err)
            names = [r["name"] for r in json.loads(path.read_text())]
            self.assertEqual(names, ["decisionrag-navigate"])
            self.assertTrue(path.with_suffix(".md").exists())
        self.assertIn("cache: 3 hits, 4 misses", out)

    def test_tag_and_demo_jev_error_return_1(self):
        factory = lambda **kw: FailingJudge()
        with mock.patch.object(cli, "make_judge_factory", return_value=(factory, None)):
            for argv in (["tag", "x"], ["demo"]):
                code, _, err = run(argv)
                self.assertEqual(code, 1)
                self.assertIn("error: 503: upstream down", err)

    def test_flat_uses_one_request_for_all_units(self):
        built = []
        def factory(**kw):
            built.append(kw)
            return FakeJudge()
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(cli, "make_judge_factory", return_value=(factory, None)):
            run(["eval", "--mode", "flat", "--output", str(Path(d) / "r.json")])
            run(["tag", "x", "--mode", "flat"])
        self.assertEqual(built, [{"max_per_request": cli.FLAT_MAX_PER_REQUEST}] * 2)
        self.assertGreaterEqual(cli.FLAT_MAX_PER_REQUEST, 10_000)

    def test_real_judge_factory_flat_judge_is_unsplit(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            factory, transport = cli.make_judge_factory(False)
        self.assertEqual(factory().max_per_request, 50)
        self.assertEqual(factory(max_per_request=10_000).max_per_request, 10_000)

    def test_tag_and_demo_print_usage_for_real_judge(self):
        judge = jev.JevJudge(FakeTransportOK(), max_per_request=10_000)
        with mock.patch.object(cli, "make_judge_factory",
                               return_value=(lambda **kw: judge, StatsTransport())):
            for argv in (["tag", "bdsm dungeon", "--tags", "bdsm"], ["demo"]):
                code, out, _ = run(argv)
                self.assertEqual(code, 0)
                self.assertIn("nominal cost: $0.0", out)
                self.assertRegex(out, r"requests: \d+")
                self.assertIn("cache: 3 hits, 4 misses", out)


class FakeTransportOK:
    def __call__(self, body):
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.6}
            else:
                opts = list(q["criteria"])
                answers[name] = {"type": "choice", "probabilities": {o: 1 / len(opts) for o in opts}}
        return {"answers": answers, "usage": {"cost": 0.0001}}


if __name__ == "__main__":
    unittest.main()
