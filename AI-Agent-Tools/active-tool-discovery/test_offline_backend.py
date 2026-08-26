"""Behavioral tests for English offline routing and discovery."""

from agent import run_active_discovery, run_full_injection, run_retrieval_prefilter
from discovery import ToolIndex
from offline_backend import LocalEmbedder, MockChatClient, match_intents
from tools_library import ALL_TOOLS, TASKS, grade


def test_stock_wording_routes_to_specialized_stock_tool():
    intents = match_intents(
        "How is Apple's stock doing? Find related recent news."
    )

    assert intents[0][0] == "get_stock_price"


def test_academic_discovery_need_includes_arxiv_specialist_in_top_four():
    index = ToolIndex(LocalEmbedder(), tools=ALL_TOOLS)

    names = [name for name, _ in index.search(
        "search for recent academic papers", top_k=4
    )]

    assert "arxiv_search" in names


def test_offline_campaign_preserves_strategy_contract_and_token_totals():
    client = MockChatClient()
    index = ToolIndex(LocalEmbedder(), tools=ALL_TOOLS)
    precise = {"full": 0, "prefilter": 0, "discovery": 0}
    tokens = {"full": 0, "prefilter": 0, "discovery": 0}

    for task in TASKS:
        results = {
            "full": run_full_injection(client, "mock-offline", task["prompt"], ALL_TOOLS),
            "prefilter": run_retrieval_prefilter(
                client, "mock-offline", task["prompt"], index, tools=ALL_TOOLS
            ),
            "discovery": run_active_discovery(
                client, "mock-offline", task["prompt"], index, tools=ALL_TOOLS
            ),
        }
        for strategy, result in results.items():
            precise[strategy] += int(grade(
                task,
                result["called"],
                finished=result["finished"],
                successful_tools=result["successful"],
            )["precise"])
            tokens[strategy] += result["injected_tokens"]

    assert precise == {"full": 8, "prefilter": 3, "discovery": 8}
    assert tokens == {"full": 91128, "prefilter": 7779, "discovery": 7529}
