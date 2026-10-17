"""Default receipt path agreement and crash-safe ordering of the win filter."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_predict_position_demo as builder
import filter_predict_position_wins as filt
from build_predict_position_demo import OUTPUT_NAME


class _Defaults(Exception):
    pass


def parser_defaults(main):
    captured = {}

    def capture(self, argv=None, namespace=None):
        captured.update(vars(self.parse_known_args([])[0]))
        raise _Defaults

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        with contextlib.suppress(_Defaults):
            main([])
    return captured


class DefaultPaths(unittest.TestCase):
    def test_filter_reads_the_receipt_the_builder_writes_by_default(self):
        build = parser_defaults(builder.main)
        filtered = parser_defaults(filt.main)
        self.assertEqual(filtered["source_receipt"], build["receipt"])
        self.assertEqual(build["receipt"], Path("results/predict_position_demo_v1/build_manifest.json"))
        self.assertEqual(filtered["selection"], build["selection"])


def entry(root, rel):
    data = (root / rel).read_bytes()
    return {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


class FilterOrdering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.site = root / "site"
        (self.site / "media").mkdir(parents=True)
        self.keep, self.drop = "media/predict_position_a.webp", "media/predict_position_b.webp"
        (self.site / self.keep).write_bytes(b"keep")
        (self.site / self.drop).write_bytes(b"drop")
        case = lambda cid, src: {"id": cid, "systems": [{"id": "nanojev", "frames": [{"sprite": {"src": src}}]}]}
        self.data = {"cases": [case("a", self.keep), case("b", self.drop)], "summary": {"nanojev": 1}}
        (self.site / OUTPUT_NAME).write_text(json.dumps(self.data))
        self.index_bytes = (self.site / OUTPUT_NAME).read_bytes()
        self.source = root / "build_manifest.json"
        self.source.write_text(json.dumps({"schema": "nanojev-predict-position-media-receipt-v1", "passed": True,
            "generated_files": [entry(self.site, p) for p in (OUTPUT_NAME, self.keep, self.drop)]}))
        self.selection = root / "selection.json"
        self.selection.write_text("{}")
        self.backup = root / "backup"
        self.receipt = root / "out" / "filter_manifest.json"
        self.argv = ["--site", str(self.site), "--source-receipt", str(self.source), "--selection", str(self.selection),
                     "--backup", str(self.backup), "--receipt", str(self.receipt)]

    def tearDown(self):
        self.tmp.cleanup()

    def fake_select(self, data, config):
        return {"cases": [data["cases"][0]], "default_case_id": "a", "summary": data["summary"],
                "illustration_summary": {}}

    def run_main(self):
        with patch.object(filt, "select_wins", self.fake_select), contextlib.redirect_stdout(io.StringIO()):
            return filt.main(self.argv)

    def test_receipt_write_failure_leaves_site_untouched(self):
        with patch.object(filt.json, "dump", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_main()
        self.assertEqual((self.site / OUTPUT_NAME).read_bytes(), self.index_bytes)
        self.assertTrue((self.site / self.keep).is_file())
        self.assertTrue((self.site / self.drop).is_file())
        self.assertFalse(self.receipt.exists())
        self.assertFalse((self.site / (OUTPUT_NAME + ".filtered")).exists())
        self.assertFalse(self.receipt.with_name(self.receipt.name + ".pending").exists())

    def test_success_swaps_index_removes_media_and_writes_receipt(self):
        self.assertEqual(self.run_main(), 0)
        index = json.loads((self.site / OUTPUT_NAME).read_text())
        self.assertEqual([c["id"] for c in index["cases"]], ["a"])
        self.assertTrue((self.site / self.keep).is_file())
        self.assertFalse((self.site / self.drop).exists())
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt["new_index_sha256"], hashlib.sha256((self.site / OUTPUT_NAME).read_bytes()).hexdigest())
        self.assertFalse(self.receipt.with_name(self.receipt.name + ".pending").exists())
        self.assertEqual((self.backup / self.drop).read_bytes(), b"drop")


if __name__ == "__main__":
    unittest.main()
