"""Prepared keyword matching shared by legacy and room classification APIs."""

import re
import unicodedata


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


class Matcher:
    """Prepare once, then reuse for multiple subjects with stable row ordering."""

    def __init__(self, parents, rows):
        # Resolve iteratively so deep trees do not hit Python's recursion limit.
        self._ancestors = {}
        for concept in parents:
            path = []
            visiting = set()
            current = concept
            while current is not None and current not in self._ancestors:
                if current not in parents:
                    raise ValueError(f"Unknown parent concept: {current}")
                if current in visiting:
                    raise ValueError(f"Cycle in taxonomy at: {current}")
                visiting.add(current)
                path.append(current)
                current = parents[current]
            ancestors = (
                self._ancestors[current] | {current}
                if current is not None
                else frozenset()
            )
            for node in reversed(path):
                self._ancestors[node] = frozenset(ancestors)
                ancestors = ancestors | {node}

        self._keywords = []
        for row in rows:
            keyword = norm(row["keyword"])
            concept = row["concept_id"]
            if not keyword:
                raise ValueError("Keyword must not be empty")
            if concept not in parents:
                raise ValueError(f"Unknown keyword concept: {concept}")
            if row["context_required"] not in ("true", "false"):
                raise ValueError(f"Invalid context_required for: {keyword}")
            pattern = re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)")
            self._keywords.append(
                (
                    pattern,
                    keyword,
                    concept,
                    row["context_required"] == "true",
                    row["source"],
                )
            )

    def match(self, subject, tags):
        tags = tags[:5]
        texts = [("subject", subject)] + [
            (f"tag_{i + 1}", tag) for i, tag in enumerate(tags)
        ]
        hits = []
        for field, raw in texts:
            text = norm(raw)
            for pattern, keyword, concept, needs_context, source in self._keywords:
                # Substring presence is necessary for a match and far cheaper.
                if keyword in text and pattern.search(text):
                    hits.append(
                        {
                            "field": field,
                            "term": keyword,
                            "concept_id": concept,
                            "context_required": needs_context,
                            "source": source,
                        }
                    )
        direct = {hit["concept_id"] for hit in hits}
        inferred = set()
        for concept in direct:
            inferred.update(self._ancestors[concept])
        return {
            "subject": subject,
            "tags": tags,
            "matches": hits,
            "direct_concepts": sorted(direct),
            "inferred_ancestors": sorted(inferred - direct),
            "requires_context_review": any(hit["context_required"] for hit in hits),
        }


def match(subject, tags, parents, rows):
    """Compatibility wrapper; use Matcher directly when classifying a batch."""
    return Matcher(parents, rows).match(subject, tags)
