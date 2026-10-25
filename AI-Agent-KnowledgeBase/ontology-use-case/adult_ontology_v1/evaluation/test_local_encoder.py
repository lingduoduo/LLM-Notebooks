"""Offline resolution tests without optional ML dependencies."""

from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from evaluation.local_encoder import MODEL_ID, MODEL_REVISION, resolve_cached_model


class EncoderTests(unittest.TestCase):
    def test_model_resolution_uses_pinned_cache_only_and_no_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.safetensors", "config.json", "tokenizer.json"):
                (root / name).write_text("{}")
            calls = []

            def snapshot(**kwargs):
                calls.append(kwargs)
                return directory

            fake = types.SimpleNamespace(snapshot_download=snapshot)
            with patch.dict("sys.modules", {"huggingface_hub": fake}):
                self.assertEqual(resolve_cached_model(), root.resolve())
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "local_files_only": True,
                        "token": False,
                    }
                ],
            )

    def test_explicit_missing_or_incomplete_local_path_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (Path(directory) / "missing", Path(directory)):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    resolve_cached_model(path)

    def test_explicit_path_never_calls_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.safetensors", "config.json", "tokenizer.json"):
                (root / name).write_text("{}")
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                self.assertEqual(resolve_cached_model(root), root.resolve())


if __name__ == "__main__":
    unittest.main()
