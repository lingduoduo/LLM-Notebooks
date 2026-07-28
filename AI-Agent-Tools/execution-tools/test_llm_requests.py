"""Tests for how requests are shaped for the configured OpenAI model.

Found by running the offline demo with a real key present: every LLM call
failed with

    Unsupported parameter: 'max_tokens' is not supported with this model.
    Use 'max_completion_tokens' instead.

GPT-5 reasoning models renamed the output-token cap. Since the default model
is now gpt-5.6, sending `max_tokens` broke approval, summarization and error
analysis outright -- and because approval fails closed, every gated operation
was refused with an API error rather than a verdict.
"""

import pytest

import llm_helper as llm_module
from config import Config
from llm_helper import LLMHelper


class FakeCompletions:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


class FakeReply:
    def __init__(self, content):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


@pytest.fixture
def helper(monkeypatch):
    """An LLMHelper wired to a fake client, with a configurable model."""

    def make(model):
        instance = LLMHelper()
        completions = FakeCompletions(FakeReply('{"approved": true, "reason": "ok"}'))
        chat = type("Chat", (), {"completions": completions})()
        instance.client = type("Client", (), {"chat": chat})()
        instance.model = model
        instance.provider = "openai"
        return instance, completions

    return make


class TestTokenLimitParameter:
    @pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-terra", "gpt-5"])
    def test_reasoning_models_use_max_completion_tokens(self, helper, model):
        instance, completions = helper(model)

        instance.request_approval("code_execution", {"code": "x = 1"})

        sent = completions.calls[0]
        assert "max_completion_tokens" in sent
        assert "max_tokens" not in sent
        assert sent["max_completion_tokens"] == Config.MAX_TOKENS

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1-mini"])
    def test_other_models_keep_max_tokens(self, helper, model):
        instance, completions = helper(model)

        instance.request_approval("code_execution", {"code": "x = 1"})

        sent = completions.calls[0]
        assert "max_tokens" in sent
        assert "max_completion_tokens" not in sent

    def test_summarization_uses_the_same_parameter_choice(self, helper):
        instance, completions = helper("gpt-5.6")

        instance.summarize_output("virtual_terminal", "lots of output")

        assert "max_completion_tokens" in completions.calls[0]

    def test_error_analysis_uses_the_same_parameter_choice(self, helper):
        instance, completions = helper("gpt-5.6")

        instance.analyze_error("virtual_terminal", "ls /nope", "No such file")

        assert "max_completion_tokens" in completions.calls[0]

    def test_syntax_verification_uses_the_same_parameter_choice(self, helper):
        instance, completions = helper("gpt-5.6")
        completions.reply = FakeReply('{"valid": true, "errors": [], "warnings": []}')

        instance.verify_code_syntax("const x = 1;", "javascript")

        assert "max_completion_tokens" in completions.calls[0]


class TestApprovalStillParsesTheVerdict:
    def test_approval_returns_the_models_decision(self, helper):
        instance, completions = helper("gpt-5.6")
        completions.reply = FakeReply('{"approved": false, "reason": "destructive"}')

        approved, reason = instance.request_approval("terminal_command", {"command": "rm -rf /"})

        assert approved is False
        assert reason == "destructive"

    def test_fenced_json_is_still_accepted(self, helper):
        instance, completions = helper("gpt-5.6")
        completions.reply = FakeReply('```json\n{"approved": true, "reason": "safe"}\n```')

        approved, reason = instance.request_approval("code_execution", {"code": "x = 1"})

        assert approved is True
        assert reason == "safe"


class TestTokenParameterHelper:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-5.6", "max_completion_tokens"),
            ("gpt-5", "max_completion_tokens"),
            ("o3-mini", "max_completion_tokens"),
            ("gpt-4o", "max_tokens"),
            (None, "max_tokens"),
        ],
    )
    def test_parameter_name_by_model(self, model, expected):
        assert llm_module.token_limit_parameter(model) == expected
