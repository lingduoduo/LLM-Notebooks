"""Validated JSON datasets, notebook fixtures, and native LoCoMo normalization."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import re

DATA = Path(__file__).resolve().parents[1] / "data"
CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def timestamp(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("timestamp must be ISO text or null")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        dt = None
        for fmt in (
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %d %B %Y",
            "%d %B, %Y",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(value, fmt)
                break
            except ValueError:
                pass
        if dt is None:
            raise ValueError("Unsupported timestamp format") from None
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat()


def normalize_observations(items):
    if not isinstance(items, list):
        raise ValueError("observations must be a list")
    result = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            item = {"content": item}
        if not isinstance(item, dict):
            raise ValueError("Each observation must be an object or string")
        content = item.get("content", item.get("text"))
        metadata = item.get("metadata", {})
        role = item.get("role", "user")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Observation content must be nonempty text")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("Observation role must be text")
        if not isinstance(metadata, dict):
            raise ValueError("Observation metadata must be an object")
        if "entities" in metadata and (
            not isinstance(metadata["entities"], list)
            or any(not isinstance(e, str) for e in metadata["entities"])
        ):
            raise ValueError("entities must be a list of strings")
        # JSON round trip both validates and takes ownership of user metadata.
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        result.append(
            {
                "content": content.strip(),
                "role": role,
                "timestamp": timestamp(item.get("timestamp", item.get("ts"))),
                "metadata": metadata,
            }
        )
    return result


def normalize_dataset(data):
    if not isinstance(data, dict):
        raise ValueError("Dataset must be an object")
    observations = normalize_observations(
        data.get("observations", data.get("conversation"))
    )
    if not observations:
        raise ValueError("Dataset has no observations")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Dataset must contain questions")
    normalized = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            raise ValueError("Question must be an object")
        text = q.get("q", q.get("question"))
        gold = q.get("gold", q.get("answer"))
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(gold, (str, int, float))
            or isinstance(gold, bool)
        ):
            raise ValueError("Questions require text and a reference answer")
        evidence = q.get("evidence", [])
        if not isinstance(evidence, list) or any(
            not isinstance(e, str) for e in evidence
        ):
            raise ValueError("Question evidence must be a list of dialogue IDs")
        normalized.append(
            {
                "id": str(q.get("id", i)),
                "q": text,
                "gold": str(gold),
                "cat": str(q.get("cat", q.get("category", "unknown"))),
                "evidence": evidence,
            }
        )
    return {
        "name": str(data.get("name", "custom")),
        "observations": observations,
        "questions": normalized,
    }


def walkthrough_dataset():
    data = json.loads((DATA / "walkthrough.json").read_text())
    return normalize_dataset(
        {
            "name": "walkthrough-16",
            "observations": data["CONVERSATION"],
            "questions": data["QUESTIONS"],
        }
    )


def synthetic_dataset(total_turns):
    if type(total_turns) is not int or total_turns < 24 or total_turns % 2:
        raise ValueError("Synthetic scale must be an even integer >= 24")
    data = json.loads((DATA / "synthetic.json").read_text())
    facts = data["FACTS"]
    slots = [
        int((i + 0.5) / len(facts) * total_turns) // 2 * 2 for i in range(len(facts))
    ]
    facts_at = dict(zip(slots, facts))
    observations = []
    distractor_index = 0
    filler_pool = []
    for pair in range(total_turns // 2):
        slot = pair * 2
        session = slot // 12
        ts = (
            datetime(2026, 6, 1, 9)
            + timedelta(days=14 * session, hours=(slot % 12) // 2)
        ).isoformat()
        if slot in facts_at:
            content = facts_at[slot]
        elif slot % 12 == 10:
            content = data["DISTRACTORS"][distractor_index % len(data["DISTRACTORS"])]
            distractor_index += 1
        else:
            if not filler_pool:
                filler_pool = list(data["FILLER"])
                random.Random(1000 + session).shuffle(filler_pool)
            content = filler_pool.pop()
        for role, text in [
            ("user", content),
            ("assistant", data["ACKS"][pair % len(data["ACKS"])]),
        ]:
            observations.append(
                {
                    "content": text,
                    "timestamp": ts,
                    "role": role,
                    "metadata": {"session": session},
                }
            )
    return normalize_dataset(
        {
            "name": f"synthetic-{total_turns}",
            "observations": observations,
            "questions": data["QUESTIONS"],
        }
    )


def load_dataset(path, sample=0, max_questions=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(sample) is not int or sample < 0:
        raise ValueError("sample must be nonnegative")
    if isinstance(raw, list):
        if sample >= len(raw):
            raise ValueError("sample index is out of range")
        raw = raw[sample]
    if not isinstance(raw, dict):
        raise ValueError("Dataset must contain objects")
    if "qa" in raw:
        conv = raw.get("conversation", {})
        observations = []
        if not isinstance(conv, dict):
            raise ValueError("LoCoMo conversation must be an object")
        sessions = sorted(
            (k for k in conv if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        )
        for session in sessions:
            ts = timestamp(conv.get(session + "_date_time"))
            if not isinstance(conv[session], list):
                raise ValueError("LoCoMo session must contain turns")
            for turn in conv[session]:
                if not isinstance(turn, dict):
                    raise ValueError("Invalid LoCoMo turn")
                speaker = turn.get("speaker", "unknown")
                content = turn.get("text", "")
                if turn.get("blip_caption"):
                    content = f"[Image: {turn['blip_caption']}] {content}".strip()
                observations.append(
                    {
                        "content": content,
                        "timestamp": ts,
                        "role": "user",
                        "metadata": {
                            "speaker": speaker,
                            "entities": [speaker],
                            "session": session,
                            "dia_id": turn.get("dia_id"),
                            "source": str(path),
                        },
                    }
                )
        questions = []
        for q in raw["qa"]:
            answer = (
                q.get("adversarial_answer")
                if q.get("category") == 5
                else q.get("answer")
            )
            if answer is None:
                raise ValueError("LoCoMo question has no reference answer")
            questions.append(
                {
                    "q": q.get("question"),
                    "gold": answer,
                    "cat": CATEGORIES.get(q.get("category"), "unknown"),
                    "evidence": q.get("evidence", []),
                }
            )
        raw = {
            "name": str(raw.get("sample_id", f"locomo-{sample}")),
            "observations": observations,
            "questions": questions,
        }
    result = normalize_dataset(raw)
    if max_questions is not None:
        if type(max_questions) is not int or max_questions < 1:
            raise ValueError("max_questions must be positive")
        result["questions"] = result["questions"][:max_questions]
    return result
