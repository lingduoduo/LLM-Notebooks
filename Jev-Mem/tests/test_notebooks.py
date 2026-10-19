"""Execute both notebooks in clean subprocesses without upstream or API access."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_both_notebooks_execute_offline_from_supported_directories(self):
        for name in ("jev_mem_walkthrough.ipynb", "jev_mem_scaling.ipynb"):
            notebook = json.loads((ROOT / name).read_text())
            sources = [
                "".join(c["source"])
                for c in notebook["cells"]
                if c["cell_type"] == "code"
            ]
            script = "\n".join(sources)
            for cwd in (ROOT.parent, ROOT):
                with (
                    self.subTest(notebook=name, cwd=cwd),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    env = {
                        **os.environ,
                        "JEV_MEM_MODE": "offline",
                        "JEV_MEM_OUTPUT": tmp,
                    }
                    env.pop("JEV_MEM_REPO", None)
                    run = subprocess.run(
                        [sys.executable, "-S", "-c", script],
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=90,
                    )
                    self.assertEqual(run.returncode, 0, run.stderr[-3000:])
                    summaries = list(Path(tmp).glob("*/summary.json"))
                    self.assertEqual(len(summaries), 4 if "scaling" in name else 1)
                    self.assertTrue(
                        all(
                            json.loads(p.read_text())["mode"] == "offline"
                            for p in summaries
                        )
                    )
