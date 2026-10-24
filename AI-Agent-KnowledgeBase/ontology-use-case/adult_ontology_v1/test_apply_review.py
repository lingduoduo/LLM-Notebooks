"""Writing reviewer decisions from the review workbook back to the vocabulary."""

import csv
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import build

try:
    from openpyxl import Workbook
except ImportError:  # The review tooling is optional; the package is stdlib-only.
    Workbook = None

HEADER = [
    "Keyword",
    "Mapped concept_id",
    "Decision",
    "Corrected concept_id",
    "Policy route",
    "Set context_required",
    "Reviewer note",
]
# Small fixture vocabulary, independent of the live (already reviewed) files.
VOCABULARY = (
    "keyword,language,concept_id,source,context_required,review_status\r\n"
    "latex,en,latex,curated_seed,true,seed_review_required\r\n"
    "dildo,en,adult_toy,ldnoobw_en,false,mapped_unreviewed\r\n"
    "fingering,en,partnered_activity,ldnoobw_en,false,mapped_unreviewed\r\n"
    "snowballing,en,partnered_activity,ldnoobw_en,true,mapped_unreviewed\r\n"
    "voyeur,en,fetish_kink,ldnoobw_en,true,mapped_unreviewed\r\n"
    "twink,en,crude_slang,ldnoobw_en,true,mapped_unreviewed\r\n"
)
POLICY = (
    "keyword,language,policy_route,source,review_status\n"
    "rape,en,non_consent,ldnoobw_en,policy_review_required\n"
)
FILES = ("taxonomy_v1.json", "keywords_v1.csv", "policy_candidates_ldnoobw_en.csv")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@unittest.skipIf(Workbook is None, "openpyxl not installed")
class ApplyReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        shutil.copy(build.ROOT / "taxonomy_v1.json", self.root / "taxonomy_v1.json")
        (self.root / "keywords_v1.csv").write_bytes(VOCABULARY.encode())
        (self.root / "policy_candidates_ldnoobw_en.csv").write_bytes(POLICY.encode())
        self.patch = patch.object(build, "ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.directory.cleanup()

    def workbook(self, rows):
        wb = Workbook()
        ws = wb.active
        ws.title = "Review"
        ws.append(HEADER)
        for row in rows:
            ws.append(row)
        path = self.root / "review.xlsx"
        wb.save(path)
        return path

    def apply(self, rows, **kwargs):
        import apply_review

        with redirect_stdout(io.StringIO()):
            return apply_review.apply_review(self.workbook(rows), **kwargs)

    def test_each_decision_updates_vocabulary_and_policy_file(self):
        summary = self.apply(
            [
                ["dildo", "adult_toy", "accept", None, None, None, "ok"],
                ["fingering", "partnered_activity", "accept", None, None, "true", ""],
                [
                    "snowballing",
                    "partnered_activity",
                    "change",
                    "other_reference",
                    None,
                    None,
                    "",
                ],
                [
                    "voyeur",
                    "fetish_kink",
                    "move to policy",
                    None,
                    "non_consent",
                    None,
                    "",
                ],
                ["twink", "crude_slang", "drop", None, None, None, ""],
                [
                    "latex",
                    "latex",
                    None,
                    None,
                    None,
                    None,
                    "undecided rows are left alone",
                ],
            ],
            status="model_reviewed",
        )
        self.assertEqual(
            summary, {"accept": 2, "change": 1, "move to policy": 1, "drop": 1}
        )
        rows = {
            (r["keyword"], r["source"]): r
            for r in read_csv(self.root / "keywords_v1.csv")
        }
        self.assertEqual(
            rows[("dildo", "ldnoobw_en")]["review_status"], "model_reviewed"
        )
        self.assertEqual(rows[("fingering", "ldnoobw_en")]["context_required"], "true")
        changed = rows[("snowballing", "ldnoobw_en")]
        self.assertEqual(
            (changed["concept_id"], changed["review_status"]),
            ("other_reference", "model_reviewed"),
        )
        self.assertNotIn(("voyeur", "ldnoobw_en"), rows)
        self.assertNotIn(("twink", "ldnoobw_en"), rows)
        self.assertEqual(
            rows[("latex", "curated_seed")]["review_status"], "seed_review_required"
        )
        policy = {
            r["keyword"]: r
            for r in read_csv(self.root / "policy_candidates_ldnoobw_en.csv")
        }
        self.assertEqual(policy["voyeur"]["policy_route"], "non_consent")
        self.assertEqual(policy["voyeur"]["review_status"], "policy_review_required")
        build.Matcher(*build.load())

    def test_seed_rows_and_policy_file_dialect_are_preserved(self):
        before = (self.root / "keywords_v1.csv").read_bytes()
        self.apply([["dildo", "adult_toy", "accept", None, None, None, ""]], status="x")
        after = (self.root / "keywords_v1.csv").read_bytes()
        self.assertEqual(before.split(b"\r\n")[:2], after.split(b"\r\n")[:2])
        self.assertNotIn(
            b"\r\n", (self.root / "policy_candidates_ldnoobw_en.csv").read_bytes()
        )

    def test_invalid_decisions_change_nothing(self):
        originals = {name: (self.root / name).read_bytes() for name in FILES}
        for row in (
            ["dildo", "adult_toy", "maybe", None, None, None, ""],
            ["dildo", "adult_toy", "change", None, None, None, ""],
            ["dildo", "adult_toy", "change", "orientation", None, None, ""],
            ["dildo", "adult_toy", "change", "missing", None, None, ""],
            ["dildo", "adult_toy", "move to policy", None, "unknown_route", None, ""],
            ["dildo", "adult_toy", "accept", None, None, "yes", ""],
            ["dildo", "missing_concept", "accept", None, None, None, ""],
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.apply([row], status="model_reviewed")
            for name, data in originals.items():
                self.assertEqual((self.root / name).read_bytes(), data, name)

    def test_reapplying_the_same_workbook_changes_nothing(self):
        rows = [
            ["dildo", "adult_toy", "accept", None, None, "true", ""],
            [
                "snowballing",
                "partnered_activity",
                "change",
                "other_reference",
                None,
                None,
                "",
            ],
            ["voyeur", "fetish_kink", "move to policy", None, "non_consent", None, ""],
            ["twink", "crude_slang", "drop", None, None, None, ""],
        ]
        self.apply(rows, status="model_reviewed")
        first = {name: (self.root / name).read_bytes() for name in FILES}
        summary = self.apply(rows, status="model_reviewed")
        self.assertEqual(
            summary, {"accept": 1, "change": 1, "move to policy": 1, "drop": 1}
        )
        for name, data in first.items():
            self.assertEqual((self.root / name).read_bytes(), data, name)

    def test_overriding_an_applied_drop_or_policy_move_restores_the_row(self):
        self.apply(
            [
                ["twink", "crude_slang", "drop", None, None, None, ""],
                [
                    "voyeur",
                    "fetish_kink",
                    "move to policy",
                    None,
                    "non_consent",
                    None,
                    "",
                ],
            ],
            status="model_reviewed",
        )
        self.apply(
            [
                ["twink", "crude_slang", "accept", None, None, "false", ""],
                ["voyeur", "fetish_kink", "change", "other_reference", None, None, ""],
            ],
            status="human_reviewed",
        )
        rows = {r["keyword"]: r for r in read_csv(self.root / "keywords_v1.csv")}
        self.assertEqual(
            (rows["twink"]["concept_id"], rows["twink"]["context_required"]),
            ("crude_slang", "false"),
        )
        self.assertEqual(rows["voyeur"]["concept_id"], "other_reference")
        self.assertEqual(rows["voyeur"]["review_status"], "human_reviewed")
        # Re-created without an original flag: conservatively needs context review.
        self.assertEqual(rows["voyeur"]["context_required"], "true")
        policy = [
            r["keyword"]
            for r in read_csv(self.root / "policy_candidates_ldnoobw_en.csv")
        ]
        self.assertEqual(policy, ["rape"])
        build.Matcher(*build.load())

    def test_duplicate_keywords_in_workbook_rejected(self):
        with self.assertRaisesRegex(ValueError, "more than once"):
            self.apply(
                [
                    ["dildo", "adult_toy", "accept", None, None, None, ""],
                    ["dildo", "adult_toy", "drop", None, None, None, ""],
                ],
                status="model_reviewed",
            )

    def test_dry_run_reports_without_writing(self):
        before = (self.root / "keywords_v1.csv").read_bytes()
        summary = self.apply(
            [["twink", "crude_slang", "drop", None, None, None, ""]],
            status="model_reviewed",
            dry_run=True,
        )
        self.assertEqual(summary["drop"], 1)
        self.assertEqual((self.root / "keywords_v1.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
