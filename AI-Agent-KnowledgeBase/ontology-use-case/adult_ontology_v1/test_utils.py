"""Taxonomy, vocabulary, matching, room classification, review and visuals in utils.py."""

import copy
import csv
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import utils


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.parents, self.rows = utils.load()

    def test_sample_output_unchanged(self):
        expected = json.loads(
            (Path(__file__).parent / "sample_output.json").read_text()
        )
        self.assertEqual(
            utils.match(expected["subject"], expected["tags"], self.parents, self.rows),
            expected,
        )

    def test_reusable_matcher_preserves_boundaries_unicode_and_tag_limit(self):
        matcher = utils.Matcher(self.parents, self.rows)
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
                utils.Matcher(parents, [])

    def test_invalid_vocabulary_rejected(self):
        for change in (
            {"keyword": "  "},
            {"concept_id": "missing"},
            {"context_required": "maybe"},
        ):
            row = dict(self.rows[0], **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                utils.Matcher(self.parents, [row])


class FileTests(unittest.TestCase):
    def test_generated_files_match_supplied_artifacts(self):
        source = utils.ROOT
        with tempfile.TemporaryDirectory() as directory, patch.object(
            utils, "ROOT", Path(directory)
        ):
            vocabulary = utils.ROOT / "keywords.csv"
            utils.generate()
            for name in ("taxonomy_v1.json", "taxonomy_schema.json"):
                self.assertEqual(
                    json.loads((utils.ROOT / name).read_text()),
                    json.loads((source / name).read_text()),
                )
            with vocabulary.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(utils.SEEDS))
            self.assertTrue(all(r["source"] == "curated_seed" for r in rows))
            # Regenerating restores edited seed rows and keeps imported rows.
            original = (source / "keywords.csv").read_bytes()
            vocabulary.write_bytes(
                original.replace(b"nude,en,full_nudity", b"nude,en,lingerie", 1)
            )
            utils.generate()
            self.assertEqual(vocabulary.read_bytes(), original)
            parents, rows = utils.load()
            utils.Matcher(parents, rows)

    def test_duplicate_concept_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            utils, "ROOT", Path(directory)
        ):
            utils.generate()
            path = utils.ROOT / "taxonomy_v1.json"
            taxonomy = json.loads(path.read_text())
            taxonomy["nodes"].append(taxonomy["nodes"][0])
            path.write_text(json.dumps(taxonomy))
            with self.assertRaisesRegex(ValueError, "Duplicate concept ID"):
                utils.load()

    def test_candidate_import_preserves_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            utils, "ROOT", Path(directory)
        ):
            words = utils.ROOT / "words.txt"
            words.write_text("alpha\u2028beta\x85gamma\vdelta\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                utils.import_words(words)
            with (utils.ROOT / "candidate_terms.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["keyword"] for row in rows], ["alpha", "beta", "gamma", "delta"]
            )

    def test_candidate_import_rejects_its_own_output_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            utils, "ROOT", Path(directory)
        ):
            destination = utils.ROOT / "candidate_terms.csv"
            original = "keyword,language,source,review_status\nhello,en,seed,unmapped_unreviewed\n"
            destination.write_text(original, encoding="utf-8")
            for source in (destination, utils.ROOT / "alias.csv"):
                if source != destination:
                    source.symlink_to(destination)
                with self.subTest(source=source), self.assertRaisesRegex(
                    ValueError, "same file"
                ):
                    utils.import_words(source)
                self.assertEqual(destination.read_text(encoding="utf-8"), original)

    def test_candidate_import_skips_terms_already_in_keywords_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            utils, "ROOT", Path(directory)
        ):
            (utils.ROOT / "keywords.csv").write_text(
                ",".join(utils.FIELDS)
                + "\r\nlatex,en,latex,curated_seed,true,seed_review_required,"
                + "\r\nrape,en,,ldnoobw_en,,policy_review_required,non_consent"
                + "\r\ntwink,en,,ldnoobw_en,,excluded,\r\n",
                encoding="utf-8",
            )
            words = utils.ROOT / "words.txt"
            words.write_text("LATEX\nrape\ntwink\nnewterm\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()) as out:
                utils.import_words(words)
                utils.import_words(words, "fr")
            with (utils.ROOT / "candidate_terms.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = [(r["keyword"], r["language"]) for r in csv.DictReader(stream)]
        # Known English terms (keyword or policy row) are skipped; other
        # languages are independent.
        self.assertEqual(
            rows,
            [
                ("newterm", "en"),
                ("latex", "fr"),
                ("rape", "fr"),
                ("twink", "fr"),
                ("newterm", "fr"),
            ],
        )
        self.assertIn("skipped 3 already in keywords.csv", out.getvalue())

    def test_candidate_import_normalizes_and_deduplicates_per_language(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            utils, "ROOT", Path(directory)
        ):
            words = utils.ROOT / "words.txt"
            words.write_text("ＣＡＦÉ\n café \n\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                utils.import_words(words)
                utils.import_words(words)
                utils.import_words(words, "fr")
            with (utils.ROOT / "candidate_terms.csv").open(
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


class ExplorerTests(unittest.TestCase):
    def test_explorer_embeds_validated_taxonomy_without_script_breakout(self):

        taxonomy = utils.load_taxonomy()
        taxonomy["nodes"][1]["label"] = "</script><b>x</b>"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "taxonomy.json"
            output = Path(directory) / "explorer.html"
            source.write_text(json.dumps(taxonomy), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                utils.build_explorer(output, source)
            html = output.read_text(encoding="utf-8")
        self.assertNotIn("__TAXONOMY_JSON__", html)
        self.assertNotIn("</script><b>", html)
        self.assertIn("\\u003c/script>", html)

    def test_explorer_rejects_invalid_taxonomy(self):

        taxonomy = utils.load_taxonomy()
        taxonomy["edges"] = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "taxonomy.json"
            output = Path(directory) / "explorer.html"
            source.write_text(json.dumps(taxonomy), encoding="utf-8")
            with self.assertRaises(ValueError):
                utils.build_explorer(output, source)
            self.assertFalse(output.exists())


CATEGORIES = {
    "BODY",
    "ACT",
    "KINK",
    "FORMAT",
    "PRODUCT",
    "SOCIAL",
    "IDENTITY",
    "COMMERCE",
    "GENERIC",
}


class TaxonomyTests(unittest.TestCase):
    def test_category_codes_and_three_levels(self):
        taxonomy = utils.load_taxonomy()
        nodes = {node["id"]: node for node in taxonomy["nodes"]}
        self.assertEqual(
            {node["category_id"] for node in nodes.values() if node["level"] == 1},
            CATEGORIES,
        )
        self.assertEqual(nodes["adult_content"]["level"], 0)
        self.assertIsNone(nodes["adult_content"]["category_id"])
        self.assertEqual(nodes["lingerie"]["parent_id"], "clothing")
        self.assertEqual(nodes["lingerie"]["category_id"], "BODY")
        self.assertEqual(nodes["partnered_intimacy"]["parent_id"], "intimacy")
        self.assertEqual(nodes["chat"]["level"], 3)
        for node in nodes.values():
            self.assertIn(node["level"], (0, 1, 2, 3))
            if node["parent_id"] is not None:
                parent = nodes[node["parent_id"]]
                self.assertEqual(node["level"], parent["level"] + 1)
                if parent["level"]:
                    self.assertEqual(node["category_id"], parent["category_id"])

    def test_only_content_categories_sit_under_the_adult_content_root(self):
        nodes = {n["id"]: n for n in utils.load_taxonomy()["nodes"]}
        self.assertEqual(nodes["adult_content"]["axis"], "content")
        for node in nodes.values():
            if node["level"] == 1:
                expected = "adult_content" if node["axis"] == "content" else None
                self.assertEqual(node["parent_id"], expected, node["id"])
        self.assertEqual(
            {n["category_id"] for n in nodes.values() if n["axis"] == "content"}
            - {None},
            {"BODY", "ACT", "KINK", "PRODUCT"},
        )
        self.assertEqual(nodes["private_show"]["axis"], "format")
        self.assertEqual(nodes["orientation"]["axis"], "identity")
        self.assertEqual(nodes["tip_goal"]["parent_id"], "monetization")

    def test_axes_are_validated_so_policy_labels_cannot_be_mixed_in(self):
        original = utils.load_taxonomy()
        changes = (
            ("lingerie", {"axis": "policy"}),
            ("private_show", {"axis": "content"}),
            ("performance_format", {"axis": "content"}),
        )
        for node_id, change in changes:
            taxonomy = copy.deepcopy(original)
            next(n for n in taxonomy["nodes"] if n["id"] == node_id).update(change)
            with self.subTest(node=node_id, change=change), self.assertRaises(
                ValueError
            ):
                utils.validate_taxonomy(taxonomy)
        # A non-content category may not be attached under the content root.
        taxonomy = copy.deepcopy(original)
        next(n for n in taxonomy["nodes"] if n["id"] == "performance_format")[
            "parent_id"
        ] = "adult_content"
        taxonomy["edges"].append(
            {
                "source": "performance_format",
                "predicate": "IS_A",
                "target": "adult_content",
            }
        )
        with self.assertRaises(ValueError):
            utils.validate_taxonomy(taxonomy)

    def test_relabeling_and_reparenting_do_not_require_new_ids(self):
        taxonomy = copy.deepcopy(utils.load_taxonomy())
        node = next(n for n in taxonomy["nodes"] if n["id"] == "lingerie")
        node["label"] = "Intimate clothing"
        utils.validate_taxonomy(taxonomy)
        classifier = utils.RoomClassifier(taxonomy, utils.load()[1])
        result = classifier.classify_room("one", "lingerie", [])
        self.assertEqual(result["assignments"][0]["concept_id"], "lingerie")

    def test_invalid_taxonomy_levels_categories_and_edges_rejected(self):
        original = utils.load_taxonomy()
        for change in ({"level": 4}, {"category_id": "KINK"}, {"parent_id": "missing"}):
            taxonomy = copy.deepcopy(original)
            node = next(n for n in taxonomy["nodes"] if n["id"] == "lingerie")
            node.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                utils.validate_taxonomy(taxonomy)
        taxonomy = copy.deepcopy(original)
        taxonomy["edges"] = [
            edge for edge in taxonomy["edges"] if edge["source"] != "lingerie"
        ]
        with self.assertRaises(ValueError):
            utils.validate_taxonomy(taxonomy)

    def test_l0_root_is_optional(self):
        taxonomy = copy.deepcopy(utils.load_taxonomy())
        taxonomy["nodes"] = [n for n in taxonomy["nodes"] if n["level"] != 0]
        for node in taxonomy["nodes"]:
            if node["level"] == 1:
                node["parent_id"] = None
        taxonomy["edges"] = [
            e for e in taxonomy["edges"] if e["target"] != "adult_content"
        ]
        utils.validate_taxonomy(taxonomy)
        result = utils.RoomClassifier(taxonomy, utils.load()[1]).classify_room(
            "one", "latex", []
        )
        self.assertEqual(result["category_ids"], ["KINK"])
        self.assertNotIn("adult_content", result["inferred_ancestors"])


class RoomTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = utils.load_taxonomy()
        self.parents, self.rows = utils.load()
        self.classifier = utils.RoomClassifier(self.taxonomy, self.rows)

    def test_multiple_branches_and_grouped_raw_evidence(self):
        result = self.classifier.classify_room(
            "one", "Private show with cosplay and LATEX", ["latex"]
        )
        assignments = {a["concept_id"]: a for a in result["assignments"]}
        self.assertEqual(
            set(assignments), {"private_show", "costume_roleplay", "latex"}
        )
        self.assertEqual(result["category_ids"], ["FORMAT", "KINK"])
        self.assertEqual(result["taxonomy_version"], self.taxonomy["version"])
        self.assertEqual(len(assignments["latex"]["evidence"]), 2)
        self.assertEqual(
            assignments["latex"]["evidence"][0]["text"],
            "Private show with cosplay and LATEX",
        )
        self.assertEqual(assignments["latex"]["evidence"][1]["field"], "tag_1")
        self.assertTrue(assignments["latex"]["requires_context_review"])
        self.assertIn("material_interest", result["inferred_ancestors"])
        self.assertNotIn("full_nudity", assignments)

    def test_independent_dimensions_do_not_imply_adult_content(self):
        for subject, category, axis in (
            ("Private show", "FORMAT", "format"),
            ("tip goal tonight", "COMMERCE", "commerce"),
            ("chat", "SOCIAL", "interaction"),
        ):
            with self.subTest(subject=subject):
                result = self.classifier.classify_room("one", subject, [])
                self.assertEqual(result["category_ids"], [category])
                self.assertEqual(result["axes"], [axis])
                self.assertEqual(result["assignments"][0]["axis"], axis)
                self.assertNotIn("adult_content", result["inferred_ancestors"])
        result = self.classifier.classify_room("one", "latex", [])
        self.assertEqual(result["axes"], ["content"])
        self.assertIn("adult_content", result["inferred_ancestors"])

    def test_tip_goal_is_monetization_not_a_format(self):
        result = self.classifier.classify_room("one", "tip goal", [])
        self.assertEqual([a["concept_id"] for a in result["assignments"]], ["tip_goal"])

    def test_self_declared_identity_does_not_imply_adult_content(self):
        result = self.classifier.classify_room(
            "one", "", [], self_declared_concepts=["orientation"]
        )
        self.assertEqual(result["axes"], ["identity"])
        self.assertNotIn("adult_content", result["inferred_ancestors"])

    def test_broad_assignments_do_not_invent_specific_descendants(self):
        result = self.classifier.classify_room("one", "bdsm", [])
        self.assertEqual([a["concept_id"] for a in result["assignments"]], ["bdsm"])
        self.assertEqual(result["assignments"][0]["level"], 2)
        self.assertEqual(result["category_ids"], ["KINK"])

    def test_identity_requires_explicit_self_declaration(self):
        rows = self.rows + [
            dict(self.rows[0], keyword="identity phrase", concept_id="orientation")
        ]
        classifier = utils.RoomClassifier(self.taxonomy, rows)
        self.assertEqual(
            classifier.classify_room("one", "identity phrase", [])["assignments"], []
        )
        result = classifier.classify_room(
            "one", "", [], self_declared_concepts=["orientation"]
        )
        self.assertEqual(result["category_ids"], ["IDENTITY"])
        self.assertEqual(
            result["assignments"][0]["evidence"][0]["source"], "self_declared"
        )
        with self.assertRaises(ValueError):
            classifier.classify_room("one", "", [], self_declared_concepts=["latex"])

    def test_category_counts_deduplicate_rooms_and_multiple_concepts(self):
        rooms = [
            self.classifier.classify_room("one", "bondage latex", []),
            self.classifier.classify_room("one", "lingerie", []),
            self.classifier.classify_room("two", "latex", []),
        ]
        counts = utils.count_rooms_by_category(rooms)
        self.assertEqual(counts["KINK"], 2)
        self.assertEqual(counts["BODY"], 1)
        self.assertEqual(counts["FORMAT"], 0)
        self.assertEqual(set(counts), CATEGORIES)

    def test_root_is_not_a_room_classification_concept(self):
        rows = [dict(self.rows[0], concept_id="adult_content")]
        with self.assertRaises(ValueError):
            utils.RoomClassifier(self.taxonomy, rows)

    def test_mixed_taxonomy_versions_are_not_silently_aggregated(self):
        room = self.classifier.classify_room("one", "latex", [])
        old_room = dict(room, taxonomy_version="1.0.0")
        with self.assertRaises(ValueError):
            utils.count_rooms_by_category([room, old_room])

    def test_aggregation_requires_a_version_on_every_record(self):
        room = self.classifier.classify_room("one", "latex", [])
        for version in (None, "", 4):
            with self.subTest(version=version), self.assertRaises(ValueError):
                utils.count_rooms_by_category(
                    [dict(room, taxonomy_version=version), room]
                )

    def test_sample_room_output(self):
        expected = json.loads(
            (utils.ROOT / "sample_room_output.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            self.classifier.classify_room(
                expected["room_id"], expected["subject"], expected["tags"]
            ),
            expected,
        )

    def test_room_ids_are_required_and_tags_still_limited_to_five(self):
        with self.assertRaises(ValueError):
            self.classifier.classify_room("", "latex", [])
        result = self.classifier.classify_room("one", "", ["", "", "", "", "", "latex"])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(len(result["tags"]), 5)


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = utils.load_taxonomy()
        self.nodes = {node["id"]: node for node in self.taxonomy["nodes"]}
        self.parents, self.rows = utils.load()
        self.classifier = utils.RoomClassifier(self.taxonomy, self.rows)

    def test_generic_category_is_standalone_three_level_branch(self):
        root = self.nodes["generic_reference"]
        self.assertEqual(
            (root["category_id"], root["axis"], root["parent_id"]),
            ("GENERIC", "generic", None),
        )
        for leaf in ("explicit_general", "profanity", "brand_reference"):
            self.assertEqual(self.nodes[leaf]["level"], 3)
            self.assertEqual(self.nodes[leaf]["category_id"], "GENERIC")

    def test_ldnoobw_terms_map_to_related_categories_or_generic(self):
        for subject, concept, review in (
            ("dildo", "adult_toy", False),
            ("shibari", "bondage", False),
            ("ass", "profanity", True),
            ("porn", "explicit_general", False),
            ("bullshit", "profanity", False),
        ):
            with self.subTest(subject=subject):
                result = self.classifier.classify_room("one", subject, [])
                (assignment,) = result["assignments"]
                self.assertEqual(assignment["concept_id"], concept)
                self.assertEqual(assignment["requires_context_review"], review)
                self.assertEqual(assignment["evidence"][0]["source"], "ldnoobw_en")
        result = self.classifier.classify_room("one", "porn", [])
        self.assertNotIn("adult_content", result["inferred_ancestors"])

    def test_policy_terms_are_kept_out_of_the_matching_vocabulary(self):
        policy = utils.load_policy_candidates()
        self.assertTrue(policy)
        keywords = {utils.norm(row["keyword"]) for row in self.rows}
        for row in policy:
            self.assertEqual(row["review_status"], "policy_review_required")
            self.assertIn(row["policy_route"], utils.POLICY_ROUTES)
            self.assertEqual(row["concept_id"], "")
            self.assertNotIn(utils.norm(row["keyword"]), keywords)
        self.assertTrue(all(row["policy_route"] == "" for row in self.rows))
        self.assertEqual(
            self.classifier.classify_room("one", "jailbait", [])["assignments"], []
        )

    def test_excluded_rows_are_neither_matched_nor_policy(self):
        vocab = dict(self.rows[0])
        excluded = dict(
            vocab,
            keyword="dropped term",
            concept_id="",
            policy_route="",
            context_required="",
            review_status="excluded",
        )
        keywords, policy = utils.split_keyword_rows([vocab, excluded])
        self.assertEqual((keywords, policy), ([vocab], []))
        for bad in (
            dict(excluded, review_status="mapped_unreviewed"),  # neither kind
            dict(excluded, concept_id="latex"),  # excluded but mapped
            dict(excluded, policy_route="hate_slur"),  # excluded but routed
        ):
            with self.subTest(row=bad), self.assertRaises(ValueError):
                utils.split_keyword_rows([bad])
        clash = dict(excluded, keyword=vocab["keyword"])
        with self.assertRaisesRegex(ValueError, "excluded"):
            utils.split_keyword_rows([vocab, clash])

    def test_keywords_file_rows_must_be_exactly_one_kind(self):
        vocab = dict(self.rows[0])
        policy = dict(utils.load_policy_candidates()[0])
        for row in (
            dict(vocab, policy_route="hate_slur"),  # both a concept and a route
            dict(vocab, concept_id=""),  # neither
            dict(policy, policy_route="unknown_route"),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                utils.split_keyword_rows([row])
        clash = dict(policy, keyword=vocab["keyword"], language=vocab["language"])
        with self.assertRaisesRegex(ValueError, "both a keyword and a policy"):
            utils.split_keyword_rows([vocab, clash])

    def test_vocabulary_has_no_duplicates_or_identity_mappings(self):
        keys = [
            (utils.norm(r["keyword"]), r["language"], r["concept_id"])
            for r in self.rows
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(
            [
                r
                for r in self.rows
                if self.nodes[r["concept_id"]]["category_id"] == "IDENTITY"
            ]
        )

    def test_semicircle_rows_count_every_keyword_once(self):

        rows = utils.taxonomy_rows(self.taxonomy, self.rows)
        self.assertEqual(sum(r[3] for r in rows), len(self.rows))
        self.assertNotIn("IDENTITY", {r[0] for r in rows})
        self.assertIn(("BODY", "Nudity", "Direct to Nudity"), {r[:3] for r in rows})
        self.assertIn(
            ("KINK", "Category-level", "Direct to KINK"), {r[:3] for r in rows}
        )

    def test_duplicate_vocabulary_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate vocabulary row"):
            utils.check_vocabulary(self.rows + self.rows[:1])


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


# Small fixture keywords file, independent of the live (already reviewed) one.
HEADER_ROW = (
    "keyword,language,concept_id,source,context_required,review_status,policy_route\r\n"
)


KEYWORDS = HEADER_ROW + (
    "latex,en,latex,curated_seed,true,seed_review_required,\r\n"
    "dildo,en,adult_toy,ldnoobw_en,false,mapped_unreviewed,\r\n"
    "fingering,en,partnered_activity,ldnoobw_en,false,mapped_unreviewed,\r\n"
    "snowballing,en,partnered_activity,ldnoobw_en,true,mapped_unreviewed,\r\n"
    "voyeur,en,fetish_kink,ldnoobw_en,true,mapped_unreviewed,\r\n"
    "twink,en,crude_slang,ldnoobw_en,true,mapped_unreviewed,\r\n"
    "rape,en,,ldnoobw_en,,policy_review_required,non_consent\r\n"
)


FILES = ("taxonomy_v1.json", "keywords.csv")


def policy_rows(root):
    return [r for r in read_csv(root / "keywords.csv") if r["policy_route"]]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@unittest.skipIf(Workbook is None, "openpyxl not installed")
class ApplyReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        shutil.copy(utils.ROOT / "taxonomy_v1.json", self.root / "taxonomy_v1.json")
        (self.root / "keywords.csv").write_bytes(KEYWORDS.encode())
        self.patch = patch.object(utils, "ROOT", self.root)
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

        with redirect_stdout(io.StringIO()):
            return utils.apply_review(self.workbook(rows), **kwargs)

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
        rows = {  # keyword rows only: policy and excluded rows have no concept
            (r["keyword"], r["source"]): r
            for r in read_csv(self.root / "keywords.csv")
            if r["concept_id"]
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
        excluded = [
            r for r in read_csv(self.root / "keywords.csv") if r["keyword"] == "twink"
        ]
        self.assertEqual(
            [
                (r["concept_id"], r["policy_route"], r["review_status"])
                for r in excluded
            ],
            [("", "", "excluded")],
        )
        self.assertEqual(
            rows[("latex", "curated_seed")]["review_status"], "seed_review_required"
        )
        policy = {r["keyword"]: r for r in policy_rows(self.root)}
        self.assertEqual(policy["voyeur"]["policy_route"], "non_consent")
        self.assertEqual(policy["voyeur"]["review_status"], "policy_review_required")
        utils.Matcher(*utils.load())

    def test_seed_rows_policy_rows_and_line_endings_are_preserved(self):
        before = (self.root / "keywords.csv").read_bytes()
        self.apply([["dildo", "adult_toy", "accept", None, None, None, ""]], status="x")
        after = (self.root / "keywords.csv").read_bytes()
        self.assertEqual(before.split(b"\r\n")[:2], after.split(b"\r\n")[:2])
        self.assertTrue(
            after.endswith(
                b"rape,en,,ldnoobw_en,,policy_review_required,non_consent\r\n"
            )
        )
        self.assertEqual(after.count(b"\n"), after.count(b"\r\n"))

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
        rows = {r["keyword"]: r for r in read_csv(self.root / "keywords.csv")}
        self.assertEqual(
            (rows["twink"]["concept_id"], rows["twink"]["context_required"]),
            ("crude_slang", "false"),
        )
        self.assertEqual(rows["voyeur"]["concept_id"], "other_reference")
        self.assertEqual(rows["voyeur"]["review_status"], "human_reviewed")
        # Re-created without an original flag: conservatively needs context review.
        self.assertEqual(rows["voyeur"]["context_required"], "true")
        policy = [r["keyword"] for r in policy_rows(self.root)]
        self.assertEqual(policy, ["rape"])
        utils.Matcher(*utils.load())

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
        before = (self.root / "keywords.csv").read_bytes()
        summary = self.apply(
            [["twink", "crude_slang", "drop", None, None, None, ""]],
            status="model_reviewed",
            dry_run=True,
        )
        self.assertEqual(summary["drop"], 1)
        self.assertEqual((self.root / "keywords.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
