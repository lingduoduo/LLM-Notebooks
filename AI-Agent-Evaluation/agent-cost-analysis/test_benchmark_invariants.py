"""Regression tests for the English agent-cost benchmark's data invariants.

The benchmark's conclusions only hold when its fixtures stay valid:
  - the stable system prompt is long enough to be cache eligible
  - every tool result parses as JSON, because the model is fed raw JSON text
  - commerce amounts stay in CNY, matching the scenario's payment channel
"""
import json
import re

import agent


def test_stable_prompt_is_cache_eligible():
    assert agent._ntok(agent.STABLE_SYSTEM_PROMPT) >= agent.MIN_CACHEABLE_PREFIX_TOKENS


def test_tool_results_are_valid_json():
    for _, _, result in agent.TOOL_RESULTS:
        assert isinstance(json.loads(result), dict)


def test_scenario_uses_cny_without_dollar_amounts():
    corpus = "\n".join([agent.STABLE_SYSTEM_PROMPT, *agent.TOOL_SUMMARIES.values()])
    assert "CNY" in corpus
    assert not re.search(r"\$\s*\d", corpus)


def test_benchmark_validation_passes():
    agent.validate_benchmark()
