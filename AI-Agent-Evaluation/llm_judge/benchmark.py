import json
from pathlib import Path
from typing import List

from models import BenchmarkExample, UserContext, Item, HumanLabel


def seed_examples() -> List[BenchmarkExample]:
    user = UserContext(
        recently_watched=["NBA highlights", "Celtics vs Knicks stream"],
        topics=["basketball", "sports"],
        followed_creators=[],
    )
    item = Item(
        title="Lakers vs Warriors Live Commentary",
        tags=["basketball", "NBA", "sports"],
    )

    return [
        BenchmarkExample(
            id="clear_supported",
            user=user,
            item=item,
            explanation="You might like this because you recently watched NBA content and enjoy basketball.",
            human=HumanLabel(
                groundedness=2,
                relevance=2,
                privacy_safety=2,
                clarity=2,
                rationale="All personalization claims are directly supported.",
            ),
        ),
        BenchmarkExample(
            id="boundary_inference",
            user=user,
            item=item,
            explanation="You might like this because you seem interested in basketball.",
            human=HumanLabel(
                groundedness=2,
                relevance=2,
                privacy_safety=2,
                clarity=2,
                rationale="A mild inference is supported by recent basketball activity.",
            ),
        ),
        BenchmarkExample(
            id="unsupported_specific_claims",
            user=user,
            item=item,
            explanation="You might like this because you follow this creator and frequently watch Lakers games.",
            human=HumanLabel(
                groundedness=0,
                relevance=2,
                privacy_safety=2,
                clarity=2,
                rationale="The explanation is relevant, but creator following and Lakers frequency are unsupported.",
            ),
        ),
        BenchmarkExample(
            id="unrelated",
            user=user,
            item=item,
            explanation="You might like this because you enjoy cooking tutorials.",
            human=HumanLabel(
                groundedness=0,
                relevance=0,
                privacy_safety=2,
                clarity=2,
                rationale="No cooking interest appears in context and the explanation is unrelated to the item.",
            ),
        ),
    ]


def write_jsonl(path: str, examples: List[BenchmarkExample]) -> None:
    from dataclasses import asdict

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[BenchmarkExample]:
    """Load human-labeled records, rejecting ambiguous or malformed datasets."""
    out, ids = [], set()
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                ex = BenchmarkExample(
                    id=d["id"], user=UserContext(**d["user"]), item=Item(**d["item"]),
                    explanation=d["explanation"], human=HumanLabel(**d["human"]))
                for value in (ex.id, ex.explanation, ex.item.title, ex.human.rationale):
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("IDs, text and rationale must be nonempty strings")
                for values in (ex.user.recently_watched, ex.user.topics,
                               ex.user.followed_creators, ex.item.tags):
                    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                        raise ValueError("context fields and tags must be lists of strings")
                for criterion in ("groundedness", "relevance", "privacy_safety", "clarity"):
                    score = getattr(ex.human, criterion)
                    if type(score) is not int or score not in (0, 1, 2):
                        raise ValueError("human scores must be integers from 0 to 2")
                if ex.id in ids:
                    raise ValueError(f"duplicate ID {ex.id}")
                ids.add(ex.id)
                out.append(ex)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from None
    if not out:
        raise ValueError(f"{path}: dataset is empty")
    return out
