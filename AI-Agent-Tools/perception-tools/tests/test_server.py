import asyncio

from perception_tools.cli import TOOLS
from perception_tools.server import mcp


def test_server_registers_the_cli_tool_set():
    server_tools = asyncio.run(mcp.list_tools())
    server_names = {tool.name for tool in server_tools}
    cli_names = {tool.name for tool in TOOLS}

    assert server_names == cli_names
