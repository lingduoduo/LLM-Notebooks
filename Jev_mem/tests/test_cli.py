"""Exercise documented commands using only the standard library."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


class CLITests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run(
            [sys.executable, "-S", "-m", "Jev_mem", *map(str, args)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_build_query_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data = directory / "input.json"
            store = directory / "store"
            data.write_text(
                json.dumps(
                    [{"content": "Alice likes tea", "metadata": {"speaker": "Alice"}}]
                )
            )
            built = self.command("build", "--data", data, "--store", store)
            self.assertEqual(built.returncode, 0, built.stderr)
            query = self.command("query", "--store", store, "What does Alice like?")
            self.assertEqual(query.returncode, 0, query.stderr)
            result = json.loads(query.stdout)
            self.assertIn("tea", result["answer"])
            self.assertIsNone(result["usage"]["prompt_tokens"])
            self.assertNotEqual(
                self.command("build", "--data", data, "--store", store).returncode, 0
            )

    def test_run_and_chart_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.command("run", "--max-questions", "1", "--output", tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = Path(json.loads(result.stdout)["run"])
            ET.parse(run / "report.svg")
            script = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "Jev_mem/scripts/render_jev_mem_charts.py"),
                    "--run",
                    str(run),
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            self.assertEqual(script.returncode, 0, script.stderr)

    def test_invalid_dataset_creates_no_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data = directory / "invalid.json"
            output = directory / "output"
            data.write_text('{"observations":[{"content":""}],"questions":[]}')
            result = self.command("run", "--data", data, "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_failed_experiment_returns_nonzero_with_artifacts(self):
        import contextlib, io
        from unittest.mock import patch
        from Jev_mem.pipeline.cli import main
        from Jev_mem.pipeline.providers import (
            LexicalEncoder,
            LocalDecisions,
            OfflineCompletion,
        )

        class Broken(OfflineCompletion):
            def complete(self, *args, **kwargs):
                raise RuntimeError("failure")

        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with patch(
                "Jev_mem.pipeline.cli._providers",
                return_value=(LexicalEncoder(), LocalDecisions(), Broken(), None, {}),
            ):
                status = main(["run", "--max-questions", "1", "--output", tmp])
            self.assertNotEqual(status, 0)
            self.assertEqual(len(list(Path(tmp).glob("*/summary.json"))), 1)

    def test_build_live_does_not_require_answer_credentials(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from Jev_mem.pipeline.cli import _providers

        args = SimpleNamespace(
            mode="live",
            command="build",
            embedding="lexical",
            decisions="heuristic",
            answer_base_url=None,
            answer_model=None,
            judge_model=None,
            judge_base_url=None,
        )
        with patch.dict(os.environ, {}, clear=True):
            encoder, decisions, answer, judge, manifest = _providers(args)
        self.assertIsNone(answer)
        self.assertIsNone(judge)
