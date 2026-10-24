import csv
import io
import json
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch
import unittest
from pathlib import Path

import build


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.parents, self.rows = build.load()

    def test_sample_output_unchanged(self):
        expected = json.loads(
            (Path(__file__).parent / "sample_output.json").read_text()
        )
        self.assertEqual(
            build.match(expected["subject"], expected["tags"], self.parents, self.rows),
            expected,
        )

    def test_reusable_matcher_preserves_boundaries_unicode_and_tag_limit(self):
        matcher = build.Matcher(self.parents, self.rows)
        result = matcher.match(
            "ＮＵＤＥ nudely private\tshow", ["LATEX", "", "", "", "", "bondage"]
        )
        self.assertEqual(
            result["direct_concepts"], ["full_nudity", "latex", "private_show"]
        )
        self.assertEqual(len(result["tags"]), 5)
        self.assertEqual(
            [h["term"] for h in result["matches"]], ["nude", "private show", "latex"]
        )
        self.assertEqual(matcher.match("nudely", [])["matches"], [])
        self.assertEqual(
            matcher.match("nude", [])["inferred_ancestors"],
            ["adult_content", "body_appearance", "nudity"],
        )

    def test_invalid_hierarchies_fail_instead_of_hanging_or_silently_truncating(self):
        for parents in ({"a": "b", "b": "a"}, {"a": "missing"}):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                build.Matcher(parents, [])

    def test_invalid_vocabulary_rejected(self):
        for change in (
            {"keyword": "  "},
            {"concept_id": "missing"},
            {"context_required": "maybe"},
        ):
            row = dict(self.rows[0], **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build.Matcher(self.parents, [row])


class FileTests(unittest.TestCase):
    def test_generated_files_match_supplied_artifacts(self):
        source = build.ROOT
        with tempfile.TemporaryDirectory() as directory, patch.object(
            build, "ROOT", Path(directory)
        ):
            vocabulary = build.ROOT / "keywords_v1.csv"
            build.generate()
            for name in ("taxonomy_v1.json", "taxonomy_schema.json"):
                self.assertEqual(
                    json.loads((build.ROOT / name).read_text()),
                    json.loads((source / name).read_text()),
                )
            with vocabulary.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(build.SEEDS))
            self.assertTrue(all(r["source"] == "curated_seed" for r in rows))
            # Regenerating restores edited seed rows and keeps imported rows.
            original = (source / "keywords_v1.csv").read_bytes()
            vocabulary.write_bytes(
                original.replace(b"nude,en,full_nudity", b"nude,en,lingerie", 1)
            )
            build.generate()
            self.assertEqual(vocabulary.read_bytes(), original)
            parents, rows = build.load()
            build.Matcher(parents, rows)

    def test_duplicate_concept_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            build, "ROOT", Path(directory)
        ):
            build.generate()
            path = build.ROOT / "taxonomy_v1.json"
            taxonomy = json.loads(path.read_text())
            taxonomy["nodes"].append(taxonomy["nodes"][0])
            path.write_text(json.dumps(taxonomy))
            with self.assertRaisesRegex(ValueError, "Duplicate concept ID"):
                build.load()

    def test_candidate_import_preserves_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            build, "ROOT", Path(directory)
        ):
            words = build.ROOT / "words.txt"
            words.write_text("alpha\u2028beta\x85gamma\vdelta\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                build.import_words(words)
            with (build.ROOT / "candidate_terms.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["keyword"] for row in rows], ["alpha", "beta", "gamma", "delta"]
            )

    def test_candidate_import_rejects_its_own_output_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            build, "ROOT", Path(directory)
        ):
            destination = build.ROOT / "candidate_terms.csv"
            original = "keyword,language,source,review_status\nhello,en,seed,unmapped_unreviewed\n"
            destination.write_text(original, encoding="utf-8")
            for source in (destination, build.ROOT / "alias.csv"):
                if source != destination:
                    source.symlink_to(destination)
                with self.subTest(source=source), self.assertRaisesRegex(
                    ValueError, "same file"
                ):
                    build.import_words(source)
                self.assertEqual(destination.read_text(encoding="utf-8"), original)

    def test_candidate_import_normalizes_and_deduplicates_per_language(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            build, "ROOT", Path(directory)
        ):
            words = build.ROOT / "words.txt"
            words.write_text("ＣＡＦÉ\n café \n\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                build.import_words(words)
                build.import_words(words)
                build.import_words(words, "fr")
            with (build.ROOT / "candidate_terms.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [(r["keyword"], r["language"]) for r in rows],
                [("café", "en"), ("café", "fr")],
            )
            self.assertTrue(
                all(r["review_status"] == "unmapped_unreviewed" for r in rows)
            )


if __name__ == "__main__":
    unittest.main()
