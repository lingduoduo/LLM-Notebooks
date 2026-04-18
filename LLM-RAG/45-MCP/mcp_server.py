# mcp_server.py
"""
FastAPI MCP server for tool discovery and execution.
Provides REST endpoints for discovering and calling tools from an MCP registry.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config import MCP_PROTOCOL_VERSION
from tools import registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Server", description="Model Context Protocol Server")


# -------------------------
# Pydantic Models
# -------------------------
class ToolCallRequest(BaseModel):
    """Request model for calling a tool."""

    name: str = Field(..., description="Tool name", min_length=1, max_length=256)
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tool name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Tool name must be a non-empty string")
        return v.strip()


class ToolInfo(BaseModel):
    """Information about a single tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]


class ToolsListResponse(BaseModel):
    """Response model for tools listing."""

    protocol: str
    version: str
    tools: list[ToolInfo]


class ToolCallResponse(BaseModel):
    """Response model for tool execution."""

    ok: bool
    tool_name: str
    result: Any = None
    error: str | None = None


# -------------------------
# API Endpoints
# -------------------------
@app.get("/mcp/tools", response_model=ToolsListResponse)
def list_tools() -> ToolsListResponse:
    """
    Discover available tools.
    
    Returns:
        ToolsListResponse: Protocol version and available tools
    """
    try:
        tools = registry.list_tools()
        logger.info(f"Listed {len(tools)} tools")
        
        # Convert to ToolInfo models
        tool_infos = [
            ToolInfo(
                name=t["name"],
                description=t["description"],
                input_schema=t["input_schema"],
            )
            for t in tools
        ]
        
        return ToolsListResponse(
            protocol="mcp",
            version=MCP_PROTOCOL_VERSION,
            tools=tool_infos,
        )
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tools",
        )


@app.post("/mcp/call", response_model=ToolCallResponse)
def call_tool(req: ToolCallRequest) -> ToolCallResponse:
    """
    Execute a tool with given arguments.
    
    Args:
        req: ToolCallRequest with tool name and arguments
    
    Returns:
        ToolCallResponse: Execution result or error
    """
    logger.info(f"Tool call request: {req.name}")
    
    try:
        result = registry.call_tool(req.name, req.arguments)
        logger.info(f"Tool {req.name} executed successfully")
        return ToolCallResponse(
            ok=True,
            tool_name=req.name,
            result=result,
        )
    except ValueError as e:
        # Expected validation errors
        logger.warning(f"Tool call validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error executing tool {req.name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}",
        )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    tools = registry.list_tools()
    return {"status": "ok" if tools else "degraded", "tools_registered": len(tools)}
