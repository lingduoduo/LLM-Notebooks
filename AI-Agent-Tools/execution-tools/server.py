"""MCP server for execution tools.

Tool definitions and handlers come from `tool_registry`, which `cli.py` also
consumes, so the two entry points cannot describe different tool sets.
"""

import asyncio
import json
from typing import Any

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from tool_registry import build_registry


# Initialize server
server = Server("execution-tools")

# Instantiate every tool once, keyed by name.
REGISTRY = build_registry()


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name=spec.name,
            description=spec.description,
            inputSchema=spec.input_schema,
        )
        for spec in REGISTRY.values()
    ]


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    """Handle tool calls."""
    if arguments is None:
        arguments = {}

    try:
        spec = REGISTRY.get(name)
        if spec is None:
            raise ValueError(f"Unknown tool: {name}")

        result = await spec.handler(**arguments)

        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Tool execution failed: {str(e)}"
                }, indent=2)
            )
        ]


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="execution-tools",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
