"""JSON/JSONL corpus and label loading without model dependencies."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Optional[Union[str, Path]]


class InputError(Exception):
    """A bad --corpus / --labels file, reported to the user without a traceback."""


CORPUS_FORMAT_HINT = (
    'Expected .json  : [{"doc_id": "doc_1", "title": "...", "text": "..."}, ...]\n'
    '                  (an object with a "documents" key holding that list also works)\n'
    "Expected .jsonl : one such object per line"
)

LABELS_FORMAT_HINT = (
    'Expected .json  : {"model distillation": ["doc_3", "doc_4"], "HTTP 404 error": ["doc_6"]}\n'
    "                  i.e. query -> list of relevant doc_ids"
)


def load_corpus(
    path: PathLike, default: List[Dict], format_hint: str = CORPUS_FORMAT_HINT
) -> List[Dict]:
    """Load the corpus. Supports .json (array of documents) and .jsonl (one per line)."""
    if not path:
        return default
    path = str(path)
    docs: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            docs.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise InputError(
                                f"Corpus file '{path}' has invalid JSON on line {lineno}: {e}\n"
                                f"{format_hint}"
                            ) from None
            else:
                data = json.load(f)
                docs = data["documents"] if isinstance(data, dict) else data
    except FileNotFoundError:
        raise InputError(
            f"Corpus file not found: '{path}'\n"
            f"Omit --corpus to use the built-in sample corpus "
            f"({len(default)} documents).\n"
            f"{format_hint}"
        ) from None
    except json.JSONDecodeError as e:
        raise InputError(
            f"Corpus file '{path}' is not valid JSON: {e}\n" f"{format_hint}"
        ) from None
    except (KeyError, TypeError):
        raise InputError(
            f"Unexpected structure in corpus file '{path}'.\n" f"{format_hint}"
        ) from None

    except (OSError, UnicodeError) as error:
        raise InputError(f"Cannot read '{path}': {error}\n{format_hint}") from None

    if not isinstance(docs, list):
        raise InputError(
            f"Corpus file '{path}' must contain a document list.\n{format_hint}"
        )
    if not docs:
        raise InputError(f"Corpus file is empty: '{path}'\n{format_hint}")
    missing = [
        i
        for i, d in enumerate(docs)
        if not isinstance(d, dict) or not isinstance(d.get("text"), str)
    ]
    if missing:
        raise InputError(
            f"Corpus file '{path}': {len(missing)} document(s) have no string 'text' field "
            f"(first at index {missing[0]}).\n{format_hint}"
        )
    return docs


def load_labels(
    path: PathLike, default: Dict[str, List[str]], format_hint: str = LABELS_FORMAT_HINT
) -> Dict[str, List[str]]:
    """Load evaluation labels: {query: [relevant_doc_id, ...]}."""
    if not path:
        return default
    path = str(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except FileNotFoundError:
        raise InputError(
            f"Labels file not found: '{path}'\n"
            f"Omit --labels to use the built-in labels ({len(default)} queries).\n"
            f"{format_hint}"
        ) from None
    except json.JSONDecodeError as e:
        raise InputError(
            f"Labels file '{path}' is not valid JSON: {e}\n" f"{format_hint}"
        ) from None
    except (OSError, UnicodeError) as error:
        raise InputError(f"Cannot read '{path}': {error}\n{format_hint}") from None

    if not isinstance(labels, dict) or not labels:
        raise InputError(
            f"Labels file '{path}' must be a non-empty query -> doc_ids object.\n"
            f"{format_hint}"
        )
    for query, doc_ids in labels.items():
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(doc_ids, list)
            or any(not isinstance(doc_id, str) for doc_id in doc_ids)
        ):
            raise InputError(
                f"Labels file '{path}' must map non-empty queries to lists of string doc_ids.\n{format_hint}"
            )
    return labels
