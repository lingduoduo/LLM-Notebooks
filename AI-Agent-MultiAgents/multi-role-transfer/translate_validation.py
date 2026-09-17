#!/usr/bin/env python3
"""Build English reading copies of the retained Experiment 10-1 evidence.

The bundles under ``validation/`` are records of what the models and Tavily
actually returned, in Chinese, and several of them are bound by SHA-256 hashes
in their manifests.  Rewriting them in place would break those hashes and, worse,
would present translated text as raw provider output.  So this script never
touches them: it writes mirrors under ``validation/translated/``.

Three kinds of Chinese string are handled differently:

* **Role/Skill system prompts** are replaced with the canonical English already
  in ``roles.py`` / ``skill_orchestrator.py``.  These are the reviewed English
  equivalents of the same documents, not a fresh translation.
* **Everything else** is looked up in ``validation/translation_map.json``, which is
  keyed by a hash of the Chinese source rather than by the source itself, so the
  map contains no Chinese.  ``validation/translation_sources.json`` records
  original source ID -> English translation for reference; it is documentation and
  is not needed to run this script.
* **Anything not covered** is left in the original Chinese, and every such JSON
  path is listed in the mirror's ``_translation`` block.  Nothing is dropped and
  nothing is silently half-translated.

Usage:
    python translate_validation.py            # write validation/translated/
    python translate_validation.py --check    # report coverage, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from judge_comparison import JUDGE_PROMPT
from roles import ROLES
from skill_orchestrator import SKILL_NAMES, _fixed_system_prompt, load_skill


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "validation"
OUTPUT_ROOT = SOURCE_ROOT / "translated"
MAP_FILE = SOURCE_ROOT / "translation_map.json"
SOURCES_FILE = SOURCE_ROOT / "translation_sources.json"
SOURCES_NOTE = 'Original source ID -> English translation, using the same IDs and English values as translation_map.json. IDs remain hashes of the original Chinese text, not of these English values. Original provider text is retained in the historical evidence bundles; earlier source-reference revisions are available in Git history.'

CJK = re.compile(r"[一-鿿]")

# Chinese-era system prompts, keyed by an opening phrase unique to each role.
ROLE_PROMPT_MARKERS = {
    "你是通用助理系统的『前台分诊』角色": "triage",
    "你是『信息检索专家』": "research",
    "你是『编程专家』": "coding",
    "你是『数据分析专家』": "data_analysis",
    "你是『写作专家』": "writing",
}
SKILL_PROMPT_MARKER = "你是共享上下文的通用 Agent"

# The judge prompt is the JUDGE_PROMPT template filled with a task and two
# candidate answers.  Translating it as one blob would duplicate work already
# done on those parts, so it is decomposed and reassembled from the English
# template instead.
JUDGE_PROMPT_MARKER = "你是独立的质量评审员"
JUDGE_PROMPT_PARTS = re.compile(
    r"\n\n用户任务：\n(?P<task>.*?)"
    r"\n\n候选 A：\n(?P<answer_a>.*?)"
    r"\n\n候选 B：\n(?P<answer_b>.*?)\n?\Z",
    re.S,
)

# Files that are pure metadata already in English; mirroring them adds nothing.
SKIP_NAMES = {"translation_map.json", "translation_sources.json", "source_drift.json"}

# A fully untranslated campaign would otherwise carry thousands of residual
# entries; the counts above the listing stay exact either way.
RESIDUAL_LISTING_LIMIT = 50


def cjk_count(text: str) -> int:
    return len(CJK.findall(text))


def source_key(text: str) -> str:
    """Stable id for a Chinese source string.

    Keying the map on this instead of on the source is what keeps
    translation_map.json free of Chinese while still resolving exact matches.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def load_translations() -> dict[str, str]:
    """Return {source_key: english}."""
    if not MAP_FILE.exists():
        return {}
    return json.loads(MAP_FILE.read_text(encoding="utf-8")).get("translations", {})


def load_sources() -> dict[str, str]:
    """Return {original_source_key: English translation}; not used for lookups."""
    if not SOURCES_FILE.exists():
        return {}
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8")).get("sources", {})


# A Skill document is identified by its YAML frontmatter.  The Chinese bundles
# contain more than one revision of each; all map to the single current English
# document, so these count as substitutions rather than translations.
SKILL_DOCUMENT = re.compile(r"\A---\s*\nname:\s*(?P<name>[a-z_]+)\s*\n")


def canonical_system_prompt(text: str) -> str | None:
    """Return the repository's English text when ``text`` is its Chinese original."""
    for marker, role in ROLE_PROMPT_MARKERS.items():
        if text.startswith(marker):
            return ROLES[role].system_prompt
    if text.startswith(SKILL_PROMPT_MARKER):
        return _fixed_system_prompt()
    match = SKILL_DOCUMENT.match(text)
    if match and match.group("name") in SKILL_NAMES:
        return load_skill(match.group("name"))
    return None


class Translator:
    def __init__(self, translations: dict[str, str]):
        self.translations = translations
        self.reset()

    def reset(self) -> None:
        self.translated = 0
        self.substituted = 0
        self.untranslated: list[dict] = []

    def string(self, text: str, path: str, kind: str | None = None) -> str:
        if not CJK.search(text):
            return text
        canonical = canonical_system_prompt(text)
        if canonical is not None:
            self.substituted += 1
            return canonical
        english = self.translations.get(source_key(text))
        if english is not None:
            self.translated += 1
            return english
        judged = self.judge_prompt(text, path)
        if judged is not None:
            return judged
        payload = self.search_payload(text, path)
        if payload is not None:
            return payload
        record = {"path": path, "cjk_chars": cjk_count(text), "preview": text[:80]}
        if kind is not None:
            record["kind"] = kind
        self.untranslated.append(record)
        return text

    def judge_prompt(self, text: str, path: str) -> str | None:
        """Reassemble a judge prompt from the English template and its parts."""
        if not text.startswith(JUDGE_PROMPT_MARKER):
            return None
        match = JUDGE_PROMPT_PARTS.search(text)
        if match is None:
            return None
        parts = {
            name: self.string(value, f"{path}<judge.{name}>")
            for name, value in match.groupdict().items()
        }
        self.translated += 1
        return JUDGE_PROMPT.format(**parts)

    def search_payload(self, text: str, path: str) -> str | None:
        """Translate the readable parts of an embedded Tavily search payload.

        The payload is JSON: an already-English ``answer`` plus per-result
        ``title``/``url``/``content``.  Titles and the query are translated so
        the trajectory reads in English; ``content`` is a verbatim excerpt from a
        third-party page and stays in the original, which is what a quoted
        source should do.  The residual is reported as ``source_excerpt``.
        """
        stripped = text.lstrip()
        if not stripped.startswith("{"):
            return None
        try:
            document = json.loads(text)
        except (ValueError, TypeError):
            return None
        if not isinstance(document, dict):
            return None
        if "results" not in document and "follow_up_questions" not in document:
            return None
        # Every sub-field goes through string() so that anything still missing
        # lands on the same work queue as the rest, instead of being counted
        # here and then invisible to residual_units().
        query = document.get("query")
        if isinstance(query, str):
            document["query"] = self.string(query, f"{path}<query>")
        for index, result in enumerate(document.get("results") or []):
            if not isinstance(result, dict):
                continue
            for field in ("title", "content"):
                value = result.get(field)
                if isinstance(value, str):
                    result[field] = self.string(
                        value, f"{path}<results[{index}].{field}>",
                        kind="source_excerpt",
                    )
        return json.dumps(document, ensure_ascii=False)

    def walk(self, node, path: str = ""):
        if isinstance(node, str):
            return self.string(node, path or ".")
        if isinstance(node, dict):
            return {key: self.walk(value, f"{path}.{key}") for key, value in node.items()}
        if isinstance(node, list):
            return [self.walk(value, f"{path}[{i}]") for i, value in enumerate(node)]
        return node


def residual_units(translations: dict[str, str]) -> dict[str, int]:
    """Every Chinese string the map cannot yet resolve, with its occurrence count.

    Shared by the coverage report and the model-backed filler so both see exactly
    the same work queue.
    """
    counts: dict[str, int] = {}

    class Collector(Translator):
        def string(self, text: str, path: str, kind: str | None = None) -> str:
            result = super().string(text, path, kind)
            if (result is text and CJK.search(text)
                    and source_key(text) not in self.translations):
                counts[text] = counts.get(text, 0) + 1
            return result

    collector = Collector(translations)
    for path in source_files():
        collector.reset()
        collector.walk(json.loads(path.read_text(encoding="utf-8")))
    return counts


def source_files() -> list[Path]:
    return sorted(
        path for path in SOURCE_ROOT.rglob("*.json")
        if OUTPUT_ROOT not in path.parents and path.name not in SKIP_NAMES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report coverage without writing the mirrors")
    args = parser.parse_args()

    translations = load_translations()
    translator = Translator(translations)
    rows = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        if not CJK.search(text):
            # Already English (metadata, or a bundle translated in place); a
            # mirror would just be a duplicate.
            continue
        document = json.loads(text)
        translator.reset()
        converted = translator.walk(document)
        relative = path.relative_to(SOURCE_ROOT)
        remaining = sum(item["cjk_chars"] for item in translator.untranslated)
        rows.append({
            "file": relative.as_posix(),
            "strings_translated": translator.translated,
            "system_prompts_substituted": translator.substituted,
            "strings_untranslated": len(translator.untranslated),
            "cjk_chars_remaining": remaining,
        })
        if args.check:
            continue
        converted = {
            "_translation": {
                "source": f"../{relative.as_posix()}",
                "warning": (
                    "English reading copy. NOT evidence: the bundle under "
                    "validation/ is the record bound by the manifest hashes. "
                    "Do not verify hashes against this file."
                ),
                "strings_translated": translator.translated,
                "system_prompts_substituted_with_repository_english":
                    translator.substituted,
                "strings_left_in_chinese": len(translator.untranslated),
                "cjk_chars_left_in_chinese": remaining,
                "left_in_chinese": translator.untranslated[:RESIDUAL_LISTING_LIMIT],
                "left_in_chinese_listing_truncated":
                    len(translator.untranslated) > RESIDUAL_LISTING_LIMIT,
            },
            **(converted if isinstance(converted, dict) else {"document": converted}),
        }
        target = OUTPUT_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    width = max(len(row["file"]) for row in rows)
    print(f"{'file'.ljust(width)}  translated  substituted  left  cjk_left")
    for row in rows:
        print(f"{row['file'].ljust(width)}  {row['strings_translated']:10d}  "
              f"{row['system_prompts_substituted']:11d}  "
              f"{row['strings_untranslated']:4d}  {row['cjk_chars_remaining']:8d}")
    if not args.check:
        print(f"\nwrote mirrors under {OUTPUT_ROOT.relative_to(ROOT)}/ "
              "(originals unmodified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
