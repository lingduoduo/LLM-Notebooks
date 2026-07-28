"""Tests for the auxiliary `files` payload of the code interpreter.

Regression coverage for silent data corruption: `_is_base64` accepted any
alphanumeric string whose length was a multiple of four, so ordinary text such
as "data", "test" or "name" was treated as base64 and decoded into binary
garbage before the program ever read it. Nothing was logged and no error was
returned.
"""

import base64

import pytest

from multilang_executor import LanguageExecutor, ExecutionStatus

READ_BACK = "print(open('payload.txt', 'rb').read().decode())"


@pytest.fixture
def executor():
    return LanguageExecutor()


class TestPlainTextIsPreserved:
    @pytest.mark.parametrize("content", ["data", "test", "name", "abcd", "hello world"])
    async def test_text_reaches_the_program_verbatim(self, executor, content):
        result = await executor.execute_code(
            code=READ_BACK, language="python", files={"payload.txt": content}
        )

        assert result["status"] == ExecutionStatus.SUCCESS, result.get("stderr")
        assert result["stdout"].rstrip("\n") == content

    async def test_text_that_looks_like_base64_is_still_text(self, executor):
        """A literal base64-looking string must not be decoded implicitly."""
        result = await executor.execute_code(
            code=READ_BACK, language="python", files={"payload.txt": "aGVsbG8="}
        )

        assert result["stdout"].rstrip("\n") == "aGVsbG8="


class TestExplicitBase64:
    async def test_prefixed_content_is_decoded(self, executor):
        encoded = base64.b64encode(b"hello binary").decode()

        result = await executor.execute_code(
            code=READ_BACK,
            language="python",
            files={"payload.txt": f"base64:{encoded}"},
        )

        assert result["stdout"].rstrip("\n") == "hello binary"

    async def test_malformed_payload_is_reported_not_silently_written(self, executor):
        result = await executor.execute_code(
            code=READ_BACK,
            language="python",
            files={"payload.txt": "base64:!!!not-base64!!!"},
        )

        assert result["status"] == ExecutionStatus.ERROR
        assert "payload.txt" in result["error"]
        assert "base64" in result["error"].lower()
