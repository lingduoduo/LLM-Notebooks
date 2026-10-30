import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main as cli


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


if __name__ == "__main__":
    unittest.main()
