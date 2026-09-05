from copy import deepcopy

from models import BenchmarkExample, GenerationTrace, GenerationOutcome, JudgeResult
from llm_backend import BackendError


class DemoGenerator:
    def generate(self, ex: BenchmarkExample, attempt: int, feedback: str = "") -> str:
        if attempt == 0:
            return (
                "You might like this stream because you follow this creator "
                "and frequently watch Lakers games."
            )

        # Revision informed by judge rationale
        return (
            "You might like this stream because you've recently watched NBA "
            "content and shown interest in basketball."
        )


def generate_with_guardrail(
    example: BenchmarkExample,
    generator,
    judge,
    max_retries: int = 2,
    fallback_text: str = "Recommendation unavailable.",
) -> GenerationOutcome:
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    traces = []
    feedback = ""

    for attempt in range(max_retries + 1):
        explanation = ""
        try:
            explanation = generator.generate(example, attempt, feedback)
            candidate = deepcopy(example)
            candidate.explanation = explanation
            result = judge.evaluate(candidate)
        except BackendError as exc:
            result = JudgeResult(0, 0, 0, 0, False, rationale=f"Evaluation unavailable: {exc}")

        traces.append(
            GenerationTrace(
                attempt=attempt,
                explanation=explanation,
                judge=result,
            )
        )

        if result.passed:
            return GenerationOutcome(
                served_text=explanation,
                passed=True,
                traces=traces,
                used_fallback=False,
            )

        feedback = result.rationale

    return GenerationOutcome(
        served_text=fallback_text,
        passed=False,
        traces=traces,
        used_fallback=True,
    )
