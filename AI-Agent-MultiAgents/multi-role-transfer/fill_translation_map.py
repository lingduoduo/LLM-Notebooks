#!/usr/bin/env python3
"""Fill the remaining Chinese strings in validation/translation_map.json.

``translate_validation.py`` resolves a Chinese string either from the repository's
canonical English (system prompts, Skill documents, tool schemas) or from
``validation/translation_map.json``.  What is left is free-form model output and
verbatim page excerpts from the retained campaigns -- tens of thousands of
characters, which is why this filler exists rather than a hand-written map.

It translates only what the map cannot already resolve, writes after every batch
so an interrupted run resumes where it stopped, and never overwrites an existing
entry.  Existing entries are authoritative: hand-written and canonical
translations always win over a model's.

    export OPENAI_API_KEY=...
    python fill_translation_map.py --dry-run          # show the queue and batches
    python fill_translation_map.py --limit 50         # translate 50 units
    python fill_translation_map.py                    # translate everything left
    python translate_validation.py                   # rebuild the reading copies

The prompt asks for a faithful translation: a run that asked for "a Chinese
summary of at most 120 characters" must still say so, because the copies describe
what actually happened.
"""

from __future__ import annotations

import argparse
import json
import os

from translate_validation import (
    CJK,
    MAP_FILE,
    SOURCES_FILE,
    SOURCES_NOTE,
    load_sources,
    load_translations,
    residual_units,
    source_key,
)


SYSTEM_PROMPT = """You translate Chinese text from an AI-agent experiment log into English.

Rules:
- Translate faithfully. This is a historical record, so if the text asks for "a Chinese summary of at most 120 characters", say exactly that in English. Never restate it as English output.
- Preserve every number, unit, percentage, date, URL, identifier and piece of Markdown, JSON or LaTeX structure exactly as it appears.
- Keep Chinese sales figures in the unit the text uses. "352.1万辆" is "3.521 million units"; a table column headed "销量（万辆）" whose cell is "352.1" becomes "Sales (10k units)" with the cell left as "352.1".
- Render role and tool names verbatim: triage, research, coding, data_analysis, writing, web_search, execute_python, calculate, descriptive_stats, count_characters, load_skill, transfer_to_agent.
- Use CAAM for 中汽协 / 中国汽车工业协会, CPCA for 乘联会, MIIT for 工信部.
- Translate navigation menus and boilerplate from scraped pages literally; do not summarise or omit anything.
- Output English only. No Chinese characters may remain.

You receive a JSON object mapping an id to Chinese text. Reply with a JSON object mapping the same ids to the English translations, and nothing else."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL",
                                                        "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=None, help="defaults to OPENAI_API_KEY")
    parser.add_argument("--batch-chars", type=int, default=3000,
                        help="approximate Chinese characters per request")
    parser.add_argument("--limit", type=int, default=None,
                        help="translate at most this many units, then stop")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the queue and the batch plan without calling the model")
    return parser.parse_args()


def cjk_count(text: str) -> int:
    return len(CJK.findall(text))


def build_batches(units: list[str], batch_chars: int) -> list[list[str]]:
    """Group units into batches, keeping any single oversized unit on its own."""
    batches: list[list[str]] = []
    current: list[str] = []
    budget = 0
    for unit in units:
        size = cjk_count(unit)
        if current and budget + size > batch_chars:
            batches.append(current)
            current, budget = [], 0
        current.append(unit)
        budget += size
        if size >= batch_chars:
            batches.append(current)
            current, budget = [], 0
    if current:
        batches.append(current)
    return batches


def save(translations: dict[str, str], sources: dict[str, str]) -> None:
    """Write the map and English source reference with their original lookup IDs.

    Read reference values from the translation map so even callers supplying
    original Chinese source values cannot reintroduce them into the reference.
    """
    document = json.loads(MAP_FILE.read_text(encoding="utf-8"))
    document["translations"] = translations
    MAP_FILE.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    record = (json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
              if SOURCES_FILE.exists() else {"schema_version": 1, "sources": {}})
    record["note"] = SOURCES_NOTE
    record["sources"] = {key: translations[key] for key in sources}
    SOURCES_FILE.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")


def translate_batch(client, args, batch: list[str]) -> dict[str, str]:
    payload = {str(index): text for index, text in enumerate(batch)}
    kwargs = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if "temperature" not in str(exc).lower():
            raise
        kwargs.pop("temperature", None)
        response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or "{}"
    answer = json.loads(content)
    result = {}
    for index, text in enumerate(batch):
        value = answer.get(str(index))
        if isinstance(value, str) and value.strip():
            result[text] = value
    return result


def main() -> int:
    args = parse_args()
    translations = load_translations()
    sources = load_sources()
    counts = residual_units(translations)
    # Highest impact first: occurrences times size.
    queue = sorted(counts, key=lambda text: -counts[text] * cjk_count(text))
    if args.limit is not None:
        queue = queue[:args.limit]
    batches = build_batches(queue, args.batch_chars)

    print(f"map entries        : {len(translations)}")
    print(f"units to translate : {len(queue)}")
    print(f"chinese characters : {sum(cjk_count(text) for text in queue)}")
    print(f"batches            : {len(batches)}")
    if args.dry_run:
        for number, batch in enumerate(batches[:5], start=1):
            print(f"  batch {number}: {len(batch)} units, "
                  f"{sum(cjk_count(text) for text in batch)} chinese chars")
        if len(batches) > 5:
            print(f"  ... {len(batches) - 5} more")
        return 0
    if not queue:
        print("nothing left to translate")
        return 0

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("filling the map requires OPENAI_API_KEY (or --api-key)")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)

    added = incomplete = 0
    for number, batch in enumerate(batches, start=1):
        results = translate_batch(client, args, batch)
        for text, english in results.items():
            key = source_key(text)
            if key in translations:
                continue
            if CJK.search(english):
                incomplete += 1
                continue
            translations[key] = english
            sources[key] = english
            added += 1
        save(translations, sources)
        print(f"batch {number}/{len(batches)}: +{len(results)} "
              f"(map now {len(translations)})")
    print(f"\nadded {added} translations; {incomplete} rejected for leftover Chinese")
    print("rerun to retry anything still missing, then: python translate_validation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
