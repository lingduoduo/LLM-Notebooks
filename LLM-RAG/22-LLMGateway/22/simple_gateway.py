#!/usr/bin/env python3
"""LLM Intelligent Gateway"""
import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Iterable
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="LLM Intelligent Gateway", version="1.0.0")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[ChatMessage]
    stream: Optional[bool] = False

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Engine configuration
ENGINES = {
    "fast": {"model": "gpt-3.5-turbo", "tier": "Fast"},
    "balanced": {"model": "gpt-4", "tier": "Balanced"},
    "premium": {"model": "o3-mini", "tier": "Premium"},
}
MODEL_TO_ENGINE = {engine["model"].lower(): engine for engine in ENGINES.values()}

COMPLEX_WORDS = (
    "design",
    "architecture",
    "analysis",
    "system",
    "algorithm",
    "optimization",
)
MEDIUM_WORDS = ("explain", "principle", "method", "process")
SIMPLE_WORDS = ("what is", "definition", "translate")
STREAM_DONE_MESSAGE = "data: [DONE]\n\n"


def serialize_messages(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    """Convert request messages once so downstream calls can reuse them."""
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def fallback_content(question: str, engine: dict[str, str]) -> str:
    """Build a local mock response when the API is unavailable."""
    tier = engine["tier"]
    if tier == "Fast":
        return f"This is a quick answer to a simple question: {question[:20]}..."
    if tier == "Balanced":
        return (
            "This is a more detailed answer for a moderately complex question. "
            f"Question: {question}. This requires more comprehensive analysis and explanation."
        )
    return (
        f"This is an in-depth analysis for a complex question: {question}. "
        "It needs to be addressed from multiple angles, including technical architecture, "
        "implementation plans, performance optimization, and other key considerations."
    )


def build_completion_payload(content: str, model: str) -> dict[str, Any]:
    """Return a response payload compatible with chat completions."""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def build_stream_chunk(content: str) -> str:
    data = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def select_engine(question: str) -> dict[str, str]:
    """Intelligently select an engine"""
    content = question.lower()

    if any(word in content for word in COMPLEX_WORDS) or len(question) > 50:
        engine_type = "premium"
    elif any(word in content for word in MEDIUM_WORDS) or len(question) > 20:
        engine_type = "balanced"
    else:
        engine_type = "fast"

    engine = ENGINES[engine_type]
    logger.info("Selected engine: %s (%s)", engine["model"], engine["tier"])
    return engine


def resolve_engine(requested_model: str, question: str) -> dict[str, str]:
    """Resolve either an explicit engine/model request or auto-selection."""
    normalized_model = requested_model.strip().lower()

    if normalized_model in ("", "auto"):
        return select_engine(question)

    if normalized_model in ENGINES:
        engine = ENGINES[normalized_model]
        logger.info("Using requested engine type: %s -> %s", normalized_model, engine["model"])
        return engine

    engine = MODEL_TO_ENGINE.get(normalized_model)
    if engine:
        logger.info("Using requested model: %s", engine["model"])
        return engine

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported model '{requested_model}'. Use 'auto', an engine type "
            f"({', '.join(ENGINES.keys())}), or a configured model name."
        ),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completion requests"""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Request must include at least one message.")

    user_question = request.messages[-1].content
    engine = resolve_engine(request.model, user_question)
    serialized_messages = serialize_messages(request.messages)
    handler = stream_response if request.stream else complete_response

    try:
        if request.stream:
            return StreamingResponse(
                handler(serialized_messages, user_question, engine),
                media_type="text/event-stream",
            )
        return await handler(serialized_messages, user_question, engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def create_chat_completion(
    messages: list[dict[str, str]], engine: dict[str, str], *, stream: bool = False
):
    """Execute the OpenAI call outside the event loop."""
    if not client:
        return None

    kwargs: dict[str, Any] = {
        "model": engine["model"],
        "messages": messages,
        "stream": stream,
    }
    if not stream:
        kwargs["timeout"] = 25

    return await asyncio.to_thread(client.chat.completions.create, **kwargs)


async def stream_response(
    messages: list[dict[str, str]], user_question: str, engine: dict[str, str]
):
    """Streaming response"""
    try:
        response = await create_chat_completion(messages, engine, stream=True)
        if response is not None:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield build_stream_chunk(chunk.choices[0].delta.content)
            yield STREAM_DONE_MESSAGE
            return
    except Exception as exc:
        logger.warning("Streaming API call failed, falling back to simulated stream: %s", exc)

    # Simulated response
    for part in fallback_content(user_question, engine).split():
        yield build_stream_chunk(f"{part} ")
        await asyncio.sleep(0.2)
    yield STREAM_DONE_MESSAGE


async def complete_response(
    messages: list[dict[str, str]], user_question: str, engine: dict[str, str]
):
    """Full (non-streaming) response"""
    logger.info("Processing request using %s model...", engine["tier"])

    try:
        response = await create_chat_completion(messages, engine)
        if response is not None:
            return {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content,
                        },
                        "finish_reason": response.choices[0].finish_reason,
                    }
                ],
            }
    except Exception as e:
        logger.warning("API call failed, using simulated response: %s", e)

    return build_completion_payload(fallback_content(user_question, engine), engine["model"])


@app.get("/health")
async def health():
    return {"status": "healthy", "engines": len(ENGINES)}


@app.get("/stats")
async def stats():
    return {"engine_types": list(ENGINES.keys()), "api_available": client is not None}


if __name__ == "__main__":
    logger.info("LLM Intelligent Gateway starting up")
    logger.info("Address: http://localhost:8000")
    logger.info("Engines: %s", len(ENGINES))
    logger.info("API: %s", "Available" if client else "Simulated")
    uvicorn.run(app, host="0.0.0.0", port=8000)
