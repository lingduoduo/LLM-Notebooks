"""Regression: negative max_length must not drop the last character."""
import json
from pathlib import Path

import pytest

from perception_tools.filesystem_tools import read_file


@pytest.mark.asyncio
async def test_negative_max_length_keeps_full_content(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_text("hello world", encoding="utf-8")
    r = await read_file(str(path), max_length=-1)
    payload = json.loads(r.text if hasattr(r, "text") else r)
    msg = payload.get("message", payload)
    if isinstance(msg, dict) and "content" in msg:
        content = msg["content"]
    elif isinstance(msg, str):
        content = msg
    else:
        content = str(msg)
    assert "hello world" in content
