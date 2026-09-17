"""The English demo must enforce the length limit stated in its task."""

import json
from types import SimpleNamespace

import pytest

from demo import COMPOSITE_TASK, save_evidence


@pytest.mark.parametrize("length, accepted", [(360, True), (361, False), (0, False)])
def test_english_investor_summary_length_limit(tmp_path, length, accepted):
    draft = "A" * length
    orch = SimpleNamespace(
        activity=[], handoffs=[], api_calls=[], steps_used=1,
        terminated_by_limit=False,
        history=[{"role": "assistant", "tool_calls": [{
            "function": {"name": "count_characters",
                         "arguments": json.dumps({"text": draft})},
        }]}],
    )
    evidence = save_evidence(
        tmp_path / "evidence.json", orch, draft,
        model="offline", base_url="https://api.openai.com/v1", task=COMPOSITE_TASK,
    )
    assert "English" in COMPOSITE_TASK and "360 characters" in COMPOSITE_TASK
    assert evidence["acceptance_gates"]["investor_summary_within_360_characters"] is accepted
