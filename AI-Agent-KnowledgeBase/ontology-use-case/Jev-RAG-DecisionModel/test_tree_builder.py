import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from tree_builder import (DESCRIBE_PROMPT, DescriberUnavailable, E5Embedder,
                          OpenAIDescriber, build_keyword_tree, fallback_description,
                          load_keyword_rows, read_tree, write_tree)


def fake_embed(kws):
    return [[len(k), ord(k[0])] for k in kws]


def fake_kmeans(vectors, k, seed):
    return [i % k for i in range(len(vectors))]


def fake_describe(kws):
    return "about " + kws[0]


def rows(n, concept="c"):
    return [{"keyword": f"kw{i:03d}", "concept_id": concept} for i in range(n)]


def build(r, describe=fake_describe, **kw):
    return build_keyword_tree(r, fake_embed, fake_kmeans, describe, **kw)


def leaves(node):
    if "keyword" in node:
        return [node]
    return [x for c in node["children"] for x in leaves(c)]


class TreeBuilderTest(unittest.TestCase):
    def test_excluded_rows_dropped(self):
        raw = [
            {"keyword": "a", "concept_id": "1", "review_status": ""},
            {"keyword": "b", "concept_id": "2", "review_status": "excluded"},
            {"keyword": "c", "concept_id": "3", "review_status": "approved"},
        ]
        out = load_keyword_rows(raw)
        self.assertEqual([r["keyword"] for r in out], ["a", "c"])
        self.assertEqual(out[0], {"keyword": "a", "concept_id": "1"})

    def test_shape_and_leaf_rule(self):
        tree = build(rows(40), clusters=4)["tree"]
        self.assertEqual(tree["id"], "__root__")
        self.assertEqual(len(tree["children"]), 4)
        for i, top in enumerate(tree["children"]):
            self.assertEqual(top["id"], f"c{i}")
            self.assertEqual(len(leaves(top)), 10)
            self.assertEqual(len(top["children"]), 2)
            self.assertEqual([s["id"] for s in top["children"]],
                             [f"c{i}.0", f"c{i}.1"])
        small = build(rows(12), clusters=4)["tree"]
        for top in small["children"]:
            self.assertEqual(len(top["children"]), 3)
            self.assertTrue(all("keyword" in c for c in top["children"]))

    def test_every_keyword_appears_once(self):
        tree = build(rows(40), clusters=4)["tree"]
        ids = sorted(x["id"] for x in leaves(tree))
        self.assertEqual(ids, sorted(f"kw{i:03d}" for i in range(40)))

    def test_leaf_ids_are_file_order_indexes(self):
        r = [{"keyword": k, "concept_id": c}
             for k, c in [("s&m", "bdsm"), ("\U0001f525 hot", ""), ("a b", "x")]]
        tree = build(r, clusters=1)["tree"]
        got = {x["id"]: (x["keyword"], x["concept_id"]) for x in leaves(tree)}
        self.assertEqual(got, {"kw000": ("s&m", "bdsm"), "kw001": ("\U0001f525 hot", ""),
                               "kw002": ("a b", "x")})

    def test_k_capped_at_points(self):
        tree = build(rows(2), clusters=20)["tree"]
        self.assertEqual(len(tree["children"]), 2)

    def test_no_concept_text_in_embed_or_describe_inputs(self):
        seen = []

        def embed(kws):
            seen.extend(kws)
            return fake_embed(kws)

        def describe(kws):
            seen.extend(kws)
            return "d"

        build_keyword_tree(rows(40, "SECRET_CONCEPT"), embed, fake_kmeans,
                           describe, clusters=4)
        self.assertTrue(seen)
        self.assertFalse(any("SECRET_CONCEPT" in s for s in seen))

    def test_describer_error_uses_fallback(self):
        def boom(kws):
            raise RuntimeError("no key")

        out = build(rows(40), describe=boom, clusters=4)
        top = out["tree"]["children"][0]
        self.assertTrue(top["description"].startswith("Terms such as: "))
        self.assertEqual(top["description_source"], "fallback")
        self.assertEqual(out["meta"]["counts"]["fallback_descriptions"], 4 + 8)
        self.assertEqual(out["meta"]["counts"]["clusters"], 4)
        self.assertEqual(out["meta"]["counts"]["subclusters"], 8)
        self.assertEqual(out["meta"]["counts"]["keywords"], 40)

    def test_llm_description_source(self):
        top = build(rows(12), clusters=4)["tree"]["children"][0]
        self.assertEqual(top["description_source"], "llm")
        self.assertTrue(top["description"].startswith("about "))

    def test_fallback_description_first_eight(self):
        kws = [str(i) for i in range(10)]
        self.assertEqual(fallback_description(kws),
                         "Terms such as: 0, 1, 2, 3, 4, 5, 6, 7")

    def test_describer_gets_at_most_60(self):
        sizes = []

        def describe(kws):
            sizes.append(len(kws))
            return "d"

        build_keyword_tree(rows(100), fake_embed, fake_kmeans, describe,
                           clusters=1)
        self.assertEqual(sizes[0], 60)

    def test_write_read_roundtrip(self):
        out = build(rows(12), clusters=4)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tree.json"
            write_tree(out, p)
            self.assertEqual(read_tree(p), out)
            self.assertEqual(list(Path(d).iterdir()), [p])


class _Resp:
    status = 200

    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok_payload(text="A group."):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def http_error(code, body=b'{"error":{"message":"bad"}}'):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body))


QUOTA_BODY = b'{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}'


class BackendsTest(unittest.TestCase):
    def test_openai_request_shape(self):
        d = OpenAIDescriber("sk-secret")
        with mock.patch("urllib.request.urlopen", return_value=_Resp(ok_payload("  One.  "))) as u:
            self.assertEqual(d(["alpha", "beta"]), "One.")
        req = u.call_args[0][0]
        self.assertEqual(req.full_url, "https://api.openai.com/v1/chat/completions")
        body = json.loads(req.data)
        self.assertEqual(body["model"], "gpt-4.1-mini")
        self.assertEqual(body["temperature"], 0)
        self.assertIn(DESCRIBE_PROMPT, json.dumps(body["messages"]))
        self.assertIn("alpha, beta", body["messages"][-1]["content"])
        self.assertEqual((d.prompt_tokens, d.completion_tokens), (10, 5))
        self.assertNotIn("sk-secret", repr(d))

    def test_openai_401_disables_after_first_call(self):
        d = OpenAIDescriber("k")
        with mock.patch("urllib.request.urlopen", side_effect=http_error(401)) as u:
            for _ in range(2):
                with self.assertRaises(DescriberUnavailable):
                    d(["a"])
        self.assertEqual(u.call_count, 1)
        self.assertTrue(d.disabled)

    def test_openai_other_error_is_runtime_error(self):
        d = OpenAIDescriber("k")
        with mock.patch("urllib.request.urlopen", side_effect=http_error(400)) as u:
            with self.assertRaises(RuntimeError):
                d(["a"])
        self.assertEqual(u.call_count, 1)
        self.assertFalse(d.disabled)

    def test_openai_retries_429_then_succeeds(self):
        d = OpenAIDescriber("k")
        with mock.patch("urllib.request.urlopen",
                        side_effect=[http_error(429), _Resp(ok_payload("Ok."))]) as u, \
                mock.patch("time.sleep") as sl:
            self.assertEqual(d(["a"]), "Ok.")
        self.assertEqual(u.call_count, 2)
        sl.assert_called_once_with(1)

    def test_openai_503_exhausts_retries(self):
        d = OpenAIDescriber("k")
        with mock.patch("urllib.request.urlopen", side_effect=lambda *a, **k: (_ for _ in ()).throw(http_error(503))) as u, \
                mock.patch("time.sleep"):
            with self.assertRaises(RuntimeError) as cm:
                d(["a"])
        self.assertEqual(u.call_count, 4)
        self.assertNotIsInstance(cm.exception, DescriberUnavailable)

    def test_openai_401_not_retried(self):
        d = OpenAIDescriber("k")
        with mock.patch("urllib.request.urlopen", side_effect=http_error(401)) as u, \
                mock.patch("time.sleep") as sl:
            with self.assertRaises(DescriberUnavailable):
                d(["a"])
        self.assertEqual(u.call_count, 1)
        sl.assert_not_called()

    def test_openai_404_and_quota_429_disable_without_retry(self):
        for err in (http_error(404), http_error(429, QUOTA_BODY)):
            d = OpenAIDescriber("k")
            with mock.patch("urllib.request.urlopen", side_effect=err) as u, \
                    mock.patch("time.sleep") as sl:
                for _ in range(2):
                    with self.assertRaises(DescriberUnavailable):
                        d(["a"])
            self.assertEqual(u.call_count, 1)
            sl.assert_not_called()
            self.assertTrue(d.disabled)
            self.assertEqual(d.disabled_code, err.code)

    def test_openai_401_records_disabled_code(self):
        d = OpenAIDescriber("k")
        with mock.patch("urllib.request.urlopen", side_effect=http_error(403)):
            with self.assertRaises(DescriberUnavailable):
                d(["a"])
        self.assertEqual(d.disabled_code, 403)

    def test_e5_loads_local_files_only(self):
        st = mock.MagicMock()
        st.return_value.encode.return_value = [[1.0, 0.0]]
        fake_mod = mock.MagicMock(SentenceTransformer=st)
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_mod}):
            self.assertEqual(E5Embedder()(["a"]), [[1.0, 0.0]])
        st.assert_called_once_with("intfloat/e5-base-v2", local_files_only=True)
        st.return_value.encode.assert_called_once_with(["passage: a"], normalize_embeddings=True)

    def test_build_with_unavailable_describer_completes(self):
        d = OpenAIDescriber("k")
        with mock.patch("urllib.request.urlopen", side_effect=http_error(401)) as u:
            out = build(rows(20), describe=d, clusters=2)
        self.assertEqual(u.call_count, 1)

        def walk(n):
            if "keyword" in n:
                return
            self.assertEqual(n["description_source"], "fallback")
            for c in n["children"]:
                walk(c)
        walk(out["tree"])

    def test_empty_describer_result_falls_back(self):
        for bad in ("", "   ", None):
            out = build(rows(6), describe=lambda kws, b=bad: b, clusters=1)
            top = out["tree"]["children"][0]
            self.assertEqual(top["description_source"], "fallback")
            self.assertTrue(top["description"].startswith("Terms such as: "))

    def test_write_tree_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a" / "b" / "tree.json"
            write_tree({"x": 1}, p)
            self.assertEqual(read_tree(p), {"x": 1})

    def test_write_tree_cleans_tmp_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tree.json"
            with self.assertRaises(TypeError):
                write_tree({"x": object()}, p)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_backends_import_lazily(self):
        out = subprocess.run(
            [sys.executable, "-c",
             "import tree_builder, sys; print([m for m in ('torch','sentence_transformers','sklearn') if m in sys.modules])"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).parent)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
