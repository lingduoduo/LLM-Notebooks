import asyncio
import json

from perception_tools.wiki_enhanced import get_article_history


def test_year_only_date_returns_error_payload():
    result = asyncio.run(get_article_history("Python", "2025"))
    payload = json.loads(result.text)
    assert payload["success"] is False
    msg = str(payload["message"])
    assert "date must be" in msg or "Failed" in msg
