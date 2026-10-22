import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "finetune_reviewed_jsonl.py",
    "finetune_variant_jsonl.py",
    "finetune_rlcd_official_jsonl.py",
    "evaluate_candidate_holdouts.py",
)


def candidate(index, split):
    noul = {"t": "noul", "ins": "需要澄清？", "crit": {}, "y": 0, "soft": [1.0, 0.0]}
    return {
        "id": "case-%04d" % index,
        "split": "train_candidate",
        "source_group_id": "group-%04d" % index,
        "state": {"text": "样本 %d" % index},
        "qs": [dict(noul, id="q%d" % n) for n in range(5)],
        "metadata": {"planned_split": split, "review_status": "needs_human_review",
                     "label_source": "deepseek_flash_independent_vote_proxy"},
    }


class ProjectLayoutTests(unittest.TestCase):
    def test_training_scripts_resolve_siblings_from_any_folder_name(self):
        # AutoDL uses laya/, this repository Jev_LocalModel/; any folder name must work.
        with tempfile.TemporaryDirectory() as temp_dir:
            copy = Path(temp_dir) / "renamed_project"
            shutil.copytree(PROJECT_DIR, copy, ignore=shutil.ignore_patterns("experiments", "notebooks", "tests", "*.safetensors"))
            code = (
                "import runpy, sys; ns = runpy.run_path(sys.argv[1]); ns['_add_project_paths']();"
                "import finetune_reviewed_jsonl, rl_common; print(rl_common.__file__)"
            )
            for script in SCRIPTS:
                result = subprocess.run([sys.executable, "-c", code, str(copy / script)], cwd=temp_dir,
                                        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=""))
                self.assertEqual(result.returncode, 0, script + ": " + result.stderr)
                self.assertEqual(Path(result.stdout.strip()).resolve(), (copy / "models" / "rl_common.py").resolve())

    def test_cli_entry_points_start_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for script in (*SCRIPTS, "generate_synthetic_data.py", "visualize_training.py"):
                result = subprocess.run([sys.executable, str(PROJECT_DIR / script), "--help"],
                                        cwd=temp_dir, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, script + ": " + result.stderr)

    def test_prepare_candidate_splits_exports_planned_splits(self):
        counts = {"train": 1200, "dev": 200, "calibration": 100, "test": 400}
        rows, index = [], 0
        for split, n in counts.items():
            for _ in range(n):
                rows.append(candidate(index, split))
                index += 1
        script = PROJECT_DIR / "data_generation" / "prepare_candidate_splits.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "candidates.jsonl"
            source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            out_dir = Path(temp_dir) / "splits"
            result = subprocess.run([sys.executable, str(script), "--input", str(source), "--output-dir", str(out_dir)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["planned_split_counts"], counts)
            self.assertEqual(manifest["files"]["train-dev.jsonl"]["records"], 1400)
            test_rows = [json.loads(line) for line in (out_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(r["split"] == "test" and r["metadata"]["review_status"] == "needs_human_review" for r in test_rows))

            rows[-1]["source_group_id"] = rows[0]["source_group_id"]
            source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), "--input", str(source), "--output-dir", str(Path(temp_dir) / "leak")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source_group_id", result.stderr)


if __name__ == "__main__":
    unittest.main()
