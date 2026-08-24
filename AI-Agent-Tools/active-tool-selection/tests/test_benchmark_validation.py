"""Regression tests for benchmark accounting and CLI validation."""

from __future__ import annotations

import pytest

import benchmark
from demo_comparison import build_parser
from semantic_router import SemanticRouter


def test_offline_benchmark_reports_actual_average_retrieved_count() -> None:
    result = benchmark.evaluate_offline(benchmark.build_catalog(), top_k=100)
    actual_average = sum(
        len(task["retrieved"]) for task in result["per_task"]
    ) / len(result["per_task"])

    assert result["strategies"]["retrieval"]["tools_in_context"] == actual_average
    assert actual_average < 100


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_non_positive_top_k(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--top-k", value])


def test_cli_rejects_negative_catalog_size() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--num-tools", "-1"])


def test_offline_benchmark_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        benchmark.evaluate_offline(benchmark.build_catalog(), top_k=0)


def test_catalog_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="num_tools cannot be negative"):
        benchmark.build_catalog(-1)


@pytest.mark.parametrize("top_k", [0, -1])
def test_router_rejects_non_positive_top_k(top_k: int) -> None:
    router = SemanticRouter(benchmark.build_catalog())

    with pytest.raises(ValueError, match="top_k must be positive"):
        router.retrieve("read a file", top_k)


def test_all_tools_recall_reflects_missing_gold_tools() -> None:
    tasks = [
        {
            "name": "Missing tool",
            "task": "Use a capability that is not installed",
            "gold_tools": ["not_in_catalog"],
        }
    ]

    result = benchmark.evaluate_offline(benchmark.build_catalog(), 5, tasks)

    assert result["strategies"]["all-tools"]["recall"] == 0.0
