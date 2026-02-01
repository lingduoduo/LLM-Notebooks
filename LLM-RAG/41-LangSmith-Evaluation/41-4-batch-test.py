import os
import re
from pathlib import Path
from typing import List, Optional

import dotenv
from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import EvaluationResult, evaluate

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import RunnableLambda

dotenv.load_dotenv()

# -----------------------------
# Config
# -----------------------------
DATASET_NAME = "ds-dependable-formula-100"
EXPERIMENT_PREFIX = "regression-test"
MAX_CONCURRENCY = 5
NUM_REPETITIONS = 1

CHAIN_FILENAME = "41-3-define-chain.py"  # update if different


# -----------------------------
# Evaluators
# -----------------------------
def keyword_match_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Keyword-match accuracy evaluator."""
    prediction: str = (run.outputs or {}).get("answer", "") or ""
    expected: List[str] = (example.metadata or {}).get("expected_keywords", []) or []

    if not expected:
        return EvaluationResult(key="keyword_match", score=1.0, comment="No expected keywords provided")

    matched = [kw for kw in expected if kw in prediction]
    score = len(matched) / len(expected)

    return EvaluationResult(
        key="keyword_match",
        score=score,
        comment=f"Matched {len(matched)}/{len(expected)} keywords",
    )


def length_appropriate_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Checks whether answer length is within a reasonable range."""
    prediction: str = (run.outputs or {}).get("answer", "") or ""
    length = len(prediction)

    # Reasonable range: 10~150 chars
    if 10 <= length <= 150:
        score = 1.0
    elif 5 <= length < 10 or 150 < length <= 200:
        score = 0.5
    else:
        score = 0.0

    return EvaluationResult(
        key="length_appropriate",
        score=score,
        comment=f"Answer length: {length} chars",
    )


_SCORE_RE = re.compile(r"([01](?:\.\d+)?)")  # grabs 0, 1, 0.8, 1.0, etc.


def _parse_score(text: str) -> Optional[float]:
    """Extract a numeric score from LLM output robustly."""
    if not text:
        return None
    m = _SCORE_RE.search(text.strip())
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, val))


def semantic_relevance_evaluator(run: Run, example: Example, llm: BaseLLM) -> EvaluationResult:
    """LLM-as-a-judge semantic relevance evaluator."""
    question: str = (example.inputs or {}).get("question", "") or (example.inputs or {}).get("inputs_1", "") or ""
    prediction: str = (run.outputs or {}).get("answer", "") or ""

    judge_prompt = f"""Score how relevant the customer-support answer is to the user's question on a 0.0–1.0 scale.

USER QUESTION:
{question}

SUPPORT ANSWER:
{prediction}

RUBRIC:
- 1.0: Fully answers the question; accurate and complete
- 0.8: Mostly answers; minor omissions
- 0.6: Partially related; key info missing
- 0.4: Mentions topic but doesn't really answer
- 0.2: Barely relevant
- 0.0: Completely unrelated

Return ONLY a single number with one decimal place (e.g., 0.8).
"""

    try:
        resp = llm.invoke(judge_prompt)
        text = (getattr(resp, "content", "") or "").strip()
        score = _parse_score(text)
        if score is None:
            return EvaluationResult(
                key="semantic_relevance",
                score=0.0,
                comment=f"Could not parse score from judge output: {text[:80]}",
            )
        return EvaluationResult(
            key="semantic_relevance",
            score=score,
            comment=f"Semantic relevance score: {score}",
        )
    except Exception as e:
        return EvaluationResult(
            key="semantic_relevance",
            score=0.0,
            comment=f"Judge failed: {type(e).__name__}: {e}",
        )


# -----------------------------
# Helpers
# -----------------------------
def load_test_chain() -> object:
    """Load test_chain from the sibling chain file."""
    import importlib.util

    base_dir = Path(__file__).resolve().parent
    chain_path = base_dir / CHAIN_FILENAME
    if not chain_path.exists():
        raise FileNotFoundError(f"Chain file not found: {chain_path}")

    spec = importlib.util.spec_from_file_location("chain_module", str(chain_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {chain_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "test_chain"):
        raise AttributeError(f"{CHAIN_FILENAME} must define `test_chain`")

    return module.test_chain


def ensure_dataset_exists(client: Client, dataset_name: str) -> None:
    """Fail fast if dataset does not exist in this LangSmith workspace."""
    try:
        client.read_dataset(dataset_name=dataset_name)
    except Exception as e:
        raise RuntimeError(
            f"Dataset '{dataset_name}' not found or inaccessible. "
            f"Check LANGSMITH_API_KEY / org, and dataset name. Root error: {e}"
        ) from e


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    client = Client()
    ensure_dataset_exists(client, DATASET_NAME)

    test_chain = load_test_chain()
    print("✓ Loaded test chain successfully")

    # Wrap input schema: supports either {"question": ...} or {"inputs_1": ...}
    target = RunnableLambda(lambda x: {"question": x.get("question") or x.get("inputs_1")}) | test_chain

    # Create ONE judge instance and reuse it (big speed/cost win)
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print(f"Starting evaluation: {EXPERIMENT_PREFIX}")

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            keyword_match_evaluator,
            length_appropriate_evaluator,
            lambda run, example: semantic_relevance_evaluator(run, example, judge_llm),
        ],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=MAX_CONCURRENCY,
        num_repetitions=NUM_REPETITIONS,
    )

    # Materialize once (also triggers iteration of lazy generator)
    rows = list(results)
    print("Evaluation finished.")
    print(f"Total result rows: {len(rows)}")

    # Best-effort: print experiment URL if available (varies by SDK version)
    exp_url = getattr(results, "url", None)
    if exp_url:
        print(f"Experiment URL: {exp_url}")


if __name__ == "__main__":
    main()
