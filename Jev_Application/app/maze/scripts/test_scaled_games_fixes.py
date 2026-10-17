"""Regression tests for scaled-game dataset splits and evaluator CLI guards."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SPLITS = ("train", "dev", "calibration", "test", "ood")


def training_stub():
    # The shared trainer is not shipped with this chapter; the builder only
    # needs its split names and a row validator here.
    stub = types.ModuleType("train_pipeline_decisions")
    stub.SPLITS = SPLITS
    stub.validate_training_row = lambda row: None
    return stub


class ScaledGameEpisodes(unittest.TestCase):
    def test_default_build_has_no_initial_state_shared_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(sys.modules, {"train_pipeline_decisions": training_stub()}):
            import build_scaled_games
            import scaled_maze
            import snake_game
            out = Path(tmp) / "data"
            with patch.object(sys, "argv", ["build_scaled_games.py", "--output-dir", str(out)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                build_scaled_games.main()
            episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
            policy_split = {}
            for split in SPLITS:
                for line in (out / "policy" / f"{split}.jsonl").read_text().splitlines():
                    policy_split[json.dumps(json.loads(line)["state"], sort_keys=True)] = split
            visible_splits = {}
            for episode in episodes:
                record = (snake_game if episode["game"] == "snake" else scaled_maze).make_record
                key = json.dumps(record(episode["initial_state"], episode["split"])["state"], sort_keys=True)
                with self.subTest(episode=episode["id"]):
                    # Every held-out start was accepted into its own split's rows.
                    self.assertEqual(policy_split.get(key), episode["split"])
                visible_splits.setdefault(key, set()).add(episode["split"])
            self.assertEqual([k for k, v in visible_splits.items() if len(v) > 1], [])
            self.assertTrue(any(e["game"] == "snake" and e["split"] != "train" for e in episodes))


class EvaluatorGuards(unittest.TestCase):
    def test_scaled_game_splits_are_stripped(self):
        import evaluate_scaled_games
        episodes = [{"split": split, "game": "snake", "size": 8, "initial_state": {}} for split in SPLITS]
        selected = evaluate_scaled_games.select_episodes(episodes, "test, ood ,", 2)
        self.assertEqual([e["split"] for e in selected], ["test", "ood"])

    def test_navigation_v3_refuses_existing_output(self):
        import evaluate_navigation_v3
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            output.write_text("keep\n")
            with patch.object(sys, "argv", ["evaluate_navigation_v3.py", "--output", str(output),
                                            "--data-dir", str(Path(tmp) / "missing")]), \
                    patch.object(evaluate_navigation_v3, "select_episodes",
                                 side_effect=AssertionError("must refuse before loading data")):
                with self.assertRaisesRegex(ValueError, "new output"):
                    evaluate_navigation_v3.main()
            self.assertEqual(output.read_text(), "keep\n")


if __name__ == "__main__":
    unittest.main()
