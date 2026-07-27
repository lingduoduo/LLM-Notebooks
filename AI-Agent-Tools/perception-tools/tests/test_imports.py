"""Package import and base-model tests."""

from perception_tools import server
from perception_tools.base import ActionResponse, is_url
from perception_tools.filesystem_tools import grep_search, read_file, summarize_text
from perception_tools.multimodal_tools import parse_image, parse_video, read_document, read_webpage
from perception_tools.private_data_tools import get_calendar_events, search_notion
from perception_tools.public_data_tools import (
    convert_currency,
    get_stock_price,
    get_weather,
    search_arxiv,
    search_wayback,
    search_wikipedia,
)
from perception_tools.search_tools import download_file, search_knowledge_base, search_web


def test_primary_modules_and_callables_import():
    callables = (
        grep_search,
        read_file,
        summarize_text,
        parse_image,
        parse_video,
        read_document,
        read_webpage,
        get_calendar_events,
        search_notion,
        convert_currency,
        get_stock_price,
        get_weather,
        search_arxiv,
        search_wayback,
        search_wikipedia,
        download_file,
        search_knowledge_base,
        search_web,
    )
    assert all(callable(item) for item in callables)
    assert server.mcp is not None


def test_action_response_and_url_helpers():
    response = ActionResponse(
        success=True,
        message="Test message",
        metadata={"test": "value"},
    )
    assert response.success is True
    assert is_url("https://example.com") is True
    assert is_url("/path/to/file") is False
