from dataclasses import dataclass
from typing import Dict, List
import random

from judge import exact_score_agreement, pass_fail_metrics


@dataclass
class DriftDecision:
    drifted: bool
    reasons: List[str]
    metrics: Dict


def random_sample(records, n=300, seed=7):
    """
    Demo sampling strategy:
      - random production traffic
      - borderline/failure-heavy examples can be oversampled in production

    Here we simply sample deterministically.
    """
    if type(n) is not int or n < 1:
        raise ValueError("sample size must be positive")
    rng = random.Random(seed)
    if len(records) <= n:
        return list(records)
    return rng.sample(records, n)


def check_drift(examples, judge_results, agreement_threshold=0.85, max_false_accept_rate=0.10) -> DriftDecision:
    if not 0 <= agreement_threshold <= 1 or not 0 <= max_false_accept_rate <= 1:
        raise ValueError("thresholds must be between 0 and 1")
    agreement = exact_score_agreement(examples, judge_results)
    pf = pass_fail_metrics(examples, judge_results)

    reasons = []

    if agreement["macro"] < agreement_threshold:
        reasons.append(f"macro score agreement {agreement['macro']:.3f} < {agreement_threshold}")

    if agreement["groundedness"] < agreement_threshold:
        reasons.append(
            f"groundedness agreement {agreement['groundedness']:.3f} < {agreement_threshold}"
        )

    if pf["recall"] is not None and pf["recall"] < 0.90:
        reasons.append(f"pass recall {pf['recall']:.3f} < 0.90")

    if pf["false_accept_rate"] is not None and pf["false_accept_rate"] > max_false_accept_rate:
        reasons.append(f"false accept rate {pf['false_accept_rate']:.3f} > {max_false_accept_rate}")

    return DriftDecision(
        drifted=bool(reasons),
        reasons=reasons,
        metrics={
            "agreement": agreement,
            "pass_fail": pf,
        },
    )


# Backward-compatible name; sampling is uniform, not stratified.
stratified_sample = random_sample
