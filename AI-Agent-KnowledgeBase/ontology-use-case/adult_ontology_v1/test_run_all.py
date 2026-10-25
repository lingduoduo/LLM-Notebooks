"""The one-command pipeline: stage reporting, skipping and failure handling."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_all

# Stages that rewrite repository files or are slow stay out of the unit test.
FAST = ["--skip", "build", "--skip", "semicircle"]


class RunAllTests(unittest.TestCase):
    def run_main(self, *args):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(
            io.StringIO()
        ) as output:
            code = run_all.main(["--out", directory, *FAST, *args])
            files = sorted(p.name for p in Path(directory).iterdir())
        return code, output.getvalue(), files

    def test_pipeline_writes_reports_and_visuals(self):
        code, output, files = self.run_main()
        self.assertEqual(code, 0, output)
        for name in (
            "keyword_evaluation.json",
            "minilm_recorded_evaluation.json",
            "sample_output.json",
            "sample_room_output.json",
            "taxonomy_explorer.html",
        ):
            self.assertIn(name, files)
        self.assertIn("OK       fixtures", output)
        self.assertIn("SKIPPED  tests        not requested", output)

    def test_missing_optional_package_skips_the_stage(self):
        with self.assertRaisesRegex(run_all.Skip, "pip install no_such_pkg"):
            run_all.needs("json", "no_such_pkg")

    def test_a_failed_stage_is_reported_and_later_stages_still_run(self):
        def broken(out):
            raise RuntimeError("benchmark missing")

        with patch.dict(run_all.RUNNERS, {"evaluate": broken}):
            code, output, files = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("FAILED   evaluate     benchmark missing", output)
        self.assertIn("taxonomy_explorer.html", files)


if __name__ == "__main__":
    unittest.main()
