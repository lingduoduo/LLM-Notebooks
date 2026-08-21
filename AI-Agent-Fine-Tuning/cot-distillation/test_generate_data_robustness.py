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


class _ScriptedClient:
    """Returns the scripted contents in order; a None entry raises instead."""

    def __init__(self, contents):
        self.contents = iter(contents)
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        content = next(self.contents)
        if content is None:
            raise RuntimeError("transient 503")
        message = types.SimpleNamespace(content=content, reasoning="r")
        usage = types.SimpleNamespace(
            model_dump=lambda: {"prompt_tokens": 5, "completion_tokens": 50}
        )
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)], usage=usage
        )


def _collect(contents, **overrides):
    return asyncio.run(
        gd.distill_one(
            _ScriptedClient(contents),
            {"id": "p", "question": "q", "answer": 42},
            _args(max_retries=2, **overrides),
            asyncio.Semaphore(1),
        )
    )


def test_wrong_answer_is_resampled():
    """A wrong answer must trigger a retry, not just an exception."""
    record = _collect(["Final Answer: 7", "Final Answer: 42"])
    assert record["verified"] is True
    assert record["attempts"] == 2


def test_exhausted_resampling_keeps_the_last_trajectory():
    record = _collect(["Final Answer: 7", "Final Answer: 8", "Final Answer: 9"])
    assert record["verified"] is False
    assert record["attempts"] == 3
    assert record["content"] == "Final Answer: 9"
    assert record["error"] is None  # A wrong answer is not an API error


def test_usage_sums_every_billed_attempt():
    """Rejected samples are billed, so they must show up in the usage totals."""
    record = _collect(["Final Answer: 7", "Final Answer: 42"])
    assert record["usage"]["completion_tokens"] == 100
    assert record["usage"]["prompt_tokens"] == 10


def test_answer_verified_on_first_try_costs_one_attempt():
    record = _collect(["Final Answer: 42"])
    assert record["attempts"] == 1
    assert record["usage"]["completion_tokens"] == 50


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Long reasoning with 111 and 222\n\nFinal Answer: 337", 337.0),
        ("Step 1: compute 12\nStep 2: check 999\nThe answer is 337.", 337.0),
        ("$$\\boxed{337}$$", 337.0),
        ("no numbers here", None),
        ("The answer is 1,234.", 1234.0),
        # "-?[\\d,]+" used to match a bare comma, so prose punctuation parsed to None
        ("just, commas, here", None),
        ("Final Answer: -42", -42.0),
    ],
)
def test_answer_parsing(text, expected):
    assert gd.extract_predicted_number(text) == expected


def test_fallback_ignores_numbers_above_the_final_line():
    """The old parser scanned the whole text and would return 999 here."""
    text = "The answer is 337.\nSee step 999 for the derivation, which has no answer."
    assert gd.extract_predicted_number(text) == 999.0  # final line's last number
    # ...but a number buried mid-reasoning is never reached when the final line has one:
    assert gd.extract_predicted_number("junk 111\nFinal Answer: 337") == 337.0
