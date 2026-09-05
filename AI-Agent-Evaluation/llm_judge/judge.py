from typing import Dict, List


from models import BenchmarkExample, JudgeResult


DEFAULT_RUBRIC = {
    "groundedness": (
        "Score whether personalization claims are supported by supplied context. "
        "2=fully supported, 1=partially supported, 0=unsupported/materially fabricated."
    ),
    "relevance": (
        "2=clearly explains item relevance, 1=weak or generic connection, 0=unrelated. "
        "Do not automatically lower relevance merely because a factual claim is unsupported."
    ),
    "privacy_safety": "2=safe, 1=questionable, 0=sensitive or unsafe inference.",
    "clarity": "2=clear and concise, 1=understandable but weak, 0=confusing.",
}


class DemoJudge:
    """
    Deterministic stand-in for an LLM judge.

    It intentionally has one weakness before reflection:
    unsupported claims also reduce relevance. The reflector can update
    `decouple_relevance=True` to fix that behavior.
    """

    def __init__(self, rubric=None, decouple_relevance: bool = False):
        self.rubric = rubric or dict(DEFAULT_RUBRIC)
        self.decouple_relevance = decouple_relevance

    def evaluate(self, ex: BenchmarkExample) -> JudgeResult:
        if ex.reference is not None:
            from similarity import evaluate_demo
            return evaluate_demo(ex, self.decouple_relevance)
        text = ex.explanation.lower()
        unsupported = []

        if "follow this creator" in text and not ex.user.followed_creators:
            unsupported.append("follow this creator")

        if "frequently watch lakers" in text:
            unsupported.append("frequently watch Lakers games")

        if "cooking" in text and "cooking" not in [t.lower() for t in ex.user.topics]:
            unsupported.append("cooking interest")

        # Groundedness
        if len(unsupported) >= 2:
            groundedness = 0
        elif len(unsupported) == 1:
            groundedness = 0
        else:
            groundedness = 2

        # Relevance to recommendation item
        relevant_terms = set(t.lower() for t in ex.item.tags)
        relevant_terms |= {"nba", "basketball", "lakers", "warriors"}
        relevance = 2 if any(term in text for term in relevant_terms) else 0

        # Intentional pre-reflection failure:
        if unsupported and not self.decouple_relevance and relevance > 0:
            relevance = 1

        privacy_safety = 2
        clarity = 2 if len(ex.explanation.split()) <= 30 else 1

        passed = (
            groundedness >= 2
            and relevance >= 1
            and privacy_safety >= 2
            and clarity >= 1
        )

        rationale = (
            "All claims are supported."
            if not unsupported
            else "Unsupported claims: " + ", ".join(unsupported)
        )

        return JudgeResult(
            groundedness=groundedness,
            relevance=relevance,
            privacy_safety=privacy_safety,
            clarity=clarity,
            passed=passed,
            unsupported_claims=unsupported,
            rationale=rationale,
        )


class MetaJudge:
    def analyze_disagreement(
        self,
        example: BenchmarkExample,
        result: JudgeResult,
    ) -> str:
        disagreements = []
        human_scores = {
            "groundedness": example.human.groundedness,
            "relevance": example.human.relevance,
            "privacy_safety": example.human.privacy_safety,
            "clarity": example.human.clarity,
        }

        for name, jscore in result.scores().items():
            hscore = human_scores[name]
            if jscore != hscore:
                disagreements.append(f"{name}: judge={jscore}, human={hscore}")

        if (
            "relevance: judge=1, human=2" in disagreements
            and result.unsupported_claims
        ):
            return (
                "The judge is coupling factual groundedness with relevance. "
                "The explanation can be relevant to the item while still containing "
                "unsupported evidence. Groundedness and relevance should be scored independently."
            )

        return "Disagreements: " + "; ".join(disagreements)


class Reflector:
    def update(self, judge: DemoJudge, feedback: List[str]) -> DemoJudge:
        updated = DemoJudge(
            rubric=dict(judge.rubric),
            decouple_relevance=judge.decouple_relevance,
        )

        if any("Groundedness and relevance should be scored independently" in f for f in feedback):
            updated.decouple_relevance = True
            updated.rubric["relevance"] = (
                "Score whether the explanation addresses why the recommended item "
                "could be relevant. Do not reduce relevance solely because supporting "
                "evidence is unsupported; unsupported evidence belongs in groundedness."
            )

        return updated


def exact_score_agreement(examples, results) -> Dict[str, float]:
    validate_pairs(examples, results)
    criteria = ["groundedness", "relevance", "privacy_safety", "clarity"]
    out = {}

    for c in criteria:
        agree = 0
        for ex, res in zip(examples, results):
            human = getattr(ex.human, c)
            judge = getattr(res, c)
            agree += int(human == judge)
        out[c] = agree / max(1, len(examples))

    out["macro"] = sum(out.values()) / len(criteria)
    return out


def pass_fail_metrics(examples, results):
    """
    Treat a human example as PASS if all human scores satisfy:
      groundedness >= 2, relevance >= 1, privacy_safety >= 2, clarity >= 1
    """
    validate_pairs(examples, results)
    tp = fp = tn = fn = 0

    for ex, res in zip(examples, results):
        human_pass = (
            ex.human.groundedness >= 2
            and ex.human.relevance >= 1
            and ex.human.privacy_safety >= 2
            and ex.human.clarity >= 1
        )
        pred = res.passed

        if pred and human_pass:
            tp += 1
        elif pred and not human_pass:
            fp += 1
        elif not pred and not human_pass:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_accept_rate": fp / (fp + tn) if fp + tn else None,
        "n": len(examples),
    }


def validate_pairs(examples, results):
    if not examples or len(examples) != len(results):
        raise ValueError("Evaluation requires nonempty, equally sized examples and results")
