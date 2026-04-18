# mcp_server.py
"""
FastAPI MCP server for tool discovery and execution.
Provides REST endpoints for discovering and calling tools from an MCP registry.
"""
from __future__ import annotations

import asyncio
import secrets
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from starlette.responses import StreamingResponse

from config import (
    MCP_PROTOCOL_VERSION,
    MCP_API_KEY,
    MCP_AUTH_HEADER,
    MCP_SERVER_DESCRIPTION,
    MCP_SERVER_NAME,
    MCP_SERVER_TRANSPORTS,
    SSE_HEARTBEAT_SECONDS,
)
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


class ServerEndpointInfo(BaseModel):
    """Named server endpoint published in the manifest."""

    path: str
    method: str
    description: str


class ServerCapabilities(BaseModel):
    """Capability flags exposed by this MCP server."""

    tool_discovery: bool
    tool_invocation: bool
    event_streaming: bool
    authentication_required: bool


class ServerManifestResponse(BaseModel):
    """Unified manifest for discovery, transports, and available tools."""

    protocol: str
    version: str
    server_name: str
    server_description: str
    transports: list[str]
    capabilities: ServerCapabilities
    endpoints: Dict[str, ServerEndpointInfo]
    tools: list[ToolInfo]


class ToolCallResponse(BaseModel):
    """Response model for tool execution."""

    ok: bool
    tool_name: str
    request_id: str
    result: Any = None
    error: str | None = None


# -------------------------
# API Endpoints
# -------------------------
def _build_tool_infos() -> list[ToolInfo]:
    """Build typed tool metadata for MCP responses."""
    return [
        ToolInfo(
            name=t["name"],
            description=t["description"],
            input_schema=t["input_schema"],
        )
        for t in registry.list_tools()
    ]


def _build_manifest() -> ServerManifestResponse:
    """Build a unified server manifest to reduce client-specific adaptation logic."""
    tool_infos = _build_tool_infos()
    return ServerManifestResponse(
        protocol="mcp",
        version=MCP_PROTOCOL_VERSION,
        server_name=MCP_SERVER_NAME,
        server_description=MCP_SERVER_DESCRIPTION,
        transports=list(MCP_SERVER_TRANSPORTS),
        capabilities=ServerCapabilities(
            tool_discovery=True,
            tool_invocation=True,
            event_streaming=True,
            authentication_required=bool(MCP_API_KEY),
        ),
        endpoints={
            "manifest": ServerEndpointInfo(
                path="/mcp/manifest",
                method="GET",
                description="Unified MCP server manifest with capabilities and tools",
            ),
            "tools": ServerEndpointInfo(
                path="/mcp/tools",
                method="GET",
                description="List available tools",
            ),
            "call": ServerEndpointInfo(
                path="/mcp/call",
                method="POST",
                description="Invoke a tool by name",
            ),
            "events": ServerEndpointInfo(
                path="/mcp/events",
                method="GET",
                description="SSE stream for server readiness and capability events",
            ),
        },
        tools=tool_infos,
    )


def _format_sse_event(event: str, data: Dict[str, Any]) -> str:
    """Format a server-sent event message."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _request_id(request: Request) -> str:
    """Read or create a request id for traceability."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


def _verify_api_key(request: Request) -> None:
    """Apply optional shared-key authentication when MCP_API_KEY is configured."""
    if not MCP_API_KEY:
        return

    provided = request.headers.get(MCP_AUTH_HEADER, "")
    if not secrets.compare_digest(provided, MCP_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing MCP API key",
        )


@app.get("/mcp/tools", response_model=ToolsListResponse)
def list_tools(request: Request) -> ToolsListResponse:
    _verify_api_key(request)
    try:
        tool_infos = _build_tool_infos()
        logger.info("Listed %s tools", len(tool_infos))
        return ToolsListResponse(
            protocol="mcp",
            version=MCP_PROTOCOL_VERSION,
            tools=tool_infos,
        )
    except Exception as e:
        logger.error("Error listing tools: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tools",
        )


@app.get("/mcp/manifest", response_model=ServerManifestResponse)
def get_manifest(request: Request) -> ServerManifestResponse:
    """Return a unified MCP manifest describing capabilities, transports, and tools."""
    _verify_api_key(request)
    try:
        manifest = _build_manifest()
        logger.info("Returned MCP manifest for server %s", manifest.server_name)
        return manifest
    except Exception as e:
        logger.error("Error building manifest: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build manifest",
        )


@app.get("/mcp/events")
async def stream_events(request: Request) -> StreamingResponse:
    """Stream server readiness and tool metadata over SSE."""
    _verify_api_key(request)

    async def event_generator() -> AsyncGenerator[str, None]:
        manifest = _build_manifest()
        yield _format_sse_event(
            "ready",
            {
                "protocol": manifest.protocol,
                "version": manifest.version,
                "server_name": manifest.server_name,
                "request_id": _request_id(request),
            },
        )
        yield _format_sse_event(
            "capabilities",
            manifest.capabilities.model_dump(),
        )
        yield _format_sse_event(
            "tools",
            {"count": len(manifest.tools), "tools": [tool.model_dump() for tool in manifest.tools]},
        )

        while not await request.is_disconnected():
            await asyncio.sleep(SSE_HEARTBEAT_SECONDS)
            yield _format_sse_event("ping", {"status": "ok"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/mcp/call", response_model=ToolCallResponse)
def call_tool(req: ToolCallRequest, request: Request) -> ToolCallResponse:
    _verify_api_key(request)
    request_id = _request_id(request)
    logger.info("Tool call request: %s request_id=%s", req.name, request_id)
    
    try:
        result = registry.call_tool(req.name, req.arguments)
        logger.info("Tool %s executed successfully request_id=%s", req.name, request_id)
        return ToolCallResponse(
            ok=True,
            tool_name=req.name,
            request_id=request_id,
            result=result,
        )
    except ValueError as e:
        # Expected validation errors
        logger.warning("Tool call validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        # Unexpected errors
        logger.error("Unexpected error executing tool %s: %s", req.name, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}",
        )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    tools = registry.list_tools()
    return {"status": "ok" if tools else "degraded", "tools_registered": len(tools)}
