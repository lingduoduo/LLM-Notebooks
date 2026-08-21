"""Collection-pipeline robustness: output paths, retry bookkeeping, malformed rows."""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

import generate_data as gd


def _args(**overrides):
    base = dict(
        model="m",
        max_tokens=10,
        temperature=0.3,
        max_retries=1,
        request_timeout=5,
        answer_suffix="",
        reasoning_effort="",
        reasoning_max_tokens=0,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _Response:
    def __init__(self, content="Final Answer: 42", reasoning="let me check"):
        message = types.SimpleNamespace(content=content, reasoning=reasoning)
        self.choices = [types.SimpleNamespace(message=message)]
        self.usage = None


class _FlakyClient:
    """Fails `fail_times` times, then succeeds."""

    def __init__(self, fail_times):
        self.remaining_failures = fail_times
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise RuntimeError("transient 503")
        return _Response()


def test_successful_retry_clears_earlier_error():
    """A recovered problem must not stay flagged as an API error in the summary."""
    record = asyncio.run(
        gd.distill_one(
            _FlakyClient(fail_times=1),
            {"id": "p1", "question": "q", "answer": 42},
            _args(),
            asyncio.Semaphore(1),
        )
    )
    assert record["verified"] is True
    assert record["error"] is None


def test_exhausted_retries_keep_the_last_error():
    record = asyncio.run(
        gd.distill_one(
            _FlakyClient(fail_times=99),
            {"id": "p1", "question": "q", "answer": 42},
            _args(max_retries=1),
            asyncio.Semaphore(1),
        )
    )
    assert record["verified"] is False
    assert "RuntimeError" in record["error"]


def test_bare_output_filenames_do_not_crash(tmp_path, monkeypatch):
    """--sft_output out.jsonl (no directory part) must not FileNotFoundError."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_data.py",
            "--input", "empty.jsonl",
            "--raw_output", "raw.jsonl",
            "--sft_output", "sft.jsonl",
        ],
    )
    asyncio.run(gd.main())
    assert (tmp_path / "raw.jsonl").exists()
    assert (tmp_path / "sft.jsonl").exists()


def test_malformed_problem_row_fails_before_collecting(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "p1", "question": "q", "answer": 1}) + "\n"
        + json.dumps({"id": "p2", "question": "no answer field"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        gd.load_problems(str(bad))
    message = str(excinfo.value)
    assert "line 2" in message
    assert "answer" in message


def test_valid_problem_file_still_loads():
    problems = gd.load_problems(str(Path(__file__).parent / "problems.jsonl"))
    assert len(problems) == 24
    assert all(set(gd.REQUIRED_FIELDS) <= set(p) for p in problems)
