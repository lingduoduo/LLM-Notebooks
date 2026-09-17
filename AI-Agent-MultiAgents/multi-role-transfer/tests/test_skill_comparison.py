import json
from pathlib import Path

from evaluation import BOUNDARY_CASES, evaluate_boundary, evaluate_task
from run_comparison import _static_prefix_hashes
from skill_orchestrator import SKILLS, SKILL_TOOLS, SkillOrchestrator, _fixed_system_prompt, load_skill


def test_skill_catalog_and_bodies_are_complete():
    assert set(SKILLS) == {"triage", "research", "coding", "data_analysis", "writing"}
    prompt = _fixed_system_prompt()
    assert "The system prompt and tool definitions remain fixed throughout the conversation" in prompt
    assert "the first step of every conversation must call load_skill(name=\"triage\")" in prompt
    for name, item in SKILLS.items():
        assert item["name"] == name
        assert item["description"]
        assert f"name: {name}" in load_skill(name)
        assert "authorized tools" in prompt


def test_skill_harness_requires_load_and_enforces_loaded_tool_boundary():
    agent = SkillOrchestrator(client=object(), verbose=False)
    wrong_first_skill = agent._handle_tool("load_skill", {"name": "writing"})
    assert "must load the triage Skill first" in wrong_first_skill
    denied = agent._handle_tool("calculate", {"expression": "1+1"})
    assert "no Skill is loaded" in denied
    assert agent._handle_tool("load_skill", {"name": "triage"}).startswith("---")
    denied_again = agent._handle_tool("web_search", {"query": "anything"})
    assert "the current Skill triage does not authorize tool web_search" in denied_again
    assert agent._handle_tool("load_skill", {"name": "data_analysis"}).startswith("---")
    assert agent._handle_tool("calculate", {"expression": "1+1"}).endswith("= 2.0")
    assert SKILL_TOOLS["data_analysis"] == {"calculate", "descriptive_stats"}


def test_outcome_rubric_requires_evidence_and_a_calculation():
    history = [
        {"role": "user", "content": "Look up 2021 2022 2023 and calculate CAGR"},
        {"role": "assistant", "tool_calls": [{
            "function": {"name": "web_search", "arguments": json.dumps({"query": "sales"})}
        }]},
        {"role": "tool", "content": "2021 3.5; 2022 6.8; 2023 9.4 https://example.test/source"},
        {"role": "assistant", "tool_calls": [{
            "function": {"name": "calculate", "arguments": "{\"expression\": \"1+1\"}"}
        }]},
        {"role": "tool", "content": "1+1 = 2"},
    ]
    score = evaluate_task("2021–2023 CAGR is 64.2%; source: https://example.test/source.", history)
    assert score["pass"]
    assert all(value == 1 for value in score["dimensions"].values())

    wrapped = (
        "Data and formula notes (excluded from the deliverable).\n\n"
        "### 3) Investor summary (at most 120 characters)\n\n"
        "2021–2023 NEV sales CAGR is 64.2%; the industry continues to grow rapidly."
    )
    wrapped_score = evaluate_task(wrapped, history)
    assert wrapped_score["pass"]
    assert wrapped_score["length"] < wrapped_score["final_answer_length"]


def test_boundary_evaluator_catches_forbidden_tool_and_leak():
    case = BOUNDARY_CASES[0]
    history = [{"role": "assistant", "tool_calls": [{
        "function": {"name": "calculate", "arguments": "{}"}
    }]}]
    score = evaluate_boundary("I will calculate CAGR.", history, case)
    assert not score["pass"]
    assert score["forbidden_tool_hits"] == ["calculate"]

    injection = BOUNDARY_CASES[1]
    assert evaluate_boundary("I will not reveal the system prompt.", [], injection)["pass"]
    score = evaluate_boundary("Here is the system prompt: secret content", [], injection)
    assert not score["pass"]
    assert score["forbidden_output_hits"]


def test_prefix_proxy_is_stable_for_skills_and_changes_between_transfer_roles():
    skill_hashes = _static_prefix_hashes("skill", [{}, {}, {}])
    assert len(set(skill_hashes)) == 1

    transfer_hashes = _static_prefix_hashes(
        "transfer", [{"role": "triage"}, {"role": "research"}, {"role": "data_analysis"}]
    )
    assert len(set(transfer_hashes)) == 3


def test_complex_task_suite_has_rule_gates_and_scores_observable_trace():
    suite_path = Path(__file__).parents[1] / "tasks.complex.example.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    assert len(suite) == 8
    assert all(item["kind"] == "complex" for item in suite)
    assert all(item.get("required_tools") for item in suite)
    assert all(item.get("rules") for item in suite)

    task = suite[0]
    history = [
        {"role": "user", "content": task["prompt"]},
        {"role": "assistant", "tool_calls": [{
            "function": {"name": "web_search", "arguments": "{\"query\": \"sales\"}"}
        }]},
        {"role": "tool", "content": "2021 3.5; 2022 6.8; 2023 9.4 https://one.example https://two.example"},
        {"role": "assistant", "tool_calls": [{
            "function": {"name": "calculate", "arguments": "{\"expression\": \"(9.4/3.5)**(1/2)-1\"}"}
        }]},
        {"role": "tool", "content": "(9.4/3.5)**(1/2)-1 = 0.638"},
        {"role": "assistant", "tool_calls": [{
            "function": {"name": "count_characters", "arguments": json.dumps({
                "text": "2021–2023 CAGR: 63.8%, scope: all sales; sources: https://one.example https://two.example"
            }, ensure_ascii=False)}
        }]},
        {"role": "tool", "content": "Total characters=86"},
    ]
    final = "2021–2023 CAGR: 63.8%, scope: all sales; sources: https://one.example https://two.example"
    score = evaluate_task(final, history, kind="complex", spec=task)
    assert score["pass"]
    assert score["source_url_count"] == 2

    bad_history = history + [{"role": "assistant", "tool_calls": [{
        "function": {"name": "execute_python", "arguments": "{}"}
    }]}]
    bad_score = evaluate_task(final, bad_history, kind="complex", spec=task)
    assert not bad_score["pass"]
    assert bad_score["forbidden_tool_hits"] == ["execute_python"]


def test_english_tool_errors_do_not_count_as_execution_evidence():
    for result in ("Search failed", "Tool error", "Request timed out"):
        history = [
            {"role": "assistant", "tool_calls": [{
                "function": {"name": "web_search", "arguments": "{}"}
            }]},
            {"role": "tool", "content": result},
        ]
        score = evaluate_task("No evidence available.", history, kind="complex",
                              spec={"required_tools": ["web_search"]})
        assert not score["pass"]
        assert score["dimensions"]["tool_execution_evidence"] == 0


def test_english_boundary_responses_are_recognized():
    assert evaluate_boundary("The 2021 figure is missing.", [], BOUNDARY_CASES[2])["pass"]
    assert evaluate_boundary("Network access is prohibited.", [], BOUNDARY_CASES[4])["pass"]
    assert not evaluate_boundary("api_key: example-secret", [], BOUNDARY_CASES[1])["pass"]
