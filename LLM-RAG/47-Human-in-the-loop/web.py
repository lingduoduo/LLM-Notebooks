# web.py
"""
Web interface for Human-in-the-Loop Agent System.
Provides REST API and web UI for HITL interactions.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt

from config import (
    ENABLE_WEB_INTERFACE, WEB_HOST, WEB_PORT,
    APPROVAL_TIMEOUT_HOURS
)
from logger import get_logger, audit_logger

logger = get_logger(__name__)

# In-memory storage for pending approvals (in production, use Redis/database)
pending_approvals: Dict[str, Dict[str, Any]] = {}
agent = None
cleanup_task: asyncio.Task | None = None


def get_agent():
    """Create the HITL agent lazily so importing web.py does not require API keys."""
    global agent
    if agent is None:
        from agent import HITLAgent

        agent = HITLAgent()
    return agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background approval cleanup for the web app lifecycle."""
    global cleanup_task
    logger.info("HITL Web API starting up")
    cleanup_task = asyncio.create_task(cleanup_expired_approvals())
    try:
        yield
    finally:
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
            cleanup_task = None


app = FastAPI(
    title="HITL Agent API",
    description="Human-in-the-Loop Agent System with web interface",
    version="1.0.0",
    lifespan=lifespan,
)

# Templates and static files are optional for API-only/test usage.
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# Pydantic models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    user_id: Optional[str] = Field(default=None, description="User identifier")


class ApprovalRequest(BaseModel):
    """Approval request model."""
    approval_id: str = Field(..., description="Approval request ID")
    decision: str = Field(..., description="Decision: accept, reject, or edit")
    edit_args: Optional[Dict[str, Any]] = Field(default=None, description="Edit arguments if decision is edit")


class ChatResponse(BaseModel):
    """Chat response model."""
    thread_id: str
    status: str  # "completed", "waiting_approval", "error"
    response: Optional[str] = None
    approval_id: Optional[str] = None
    approval_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ApprovalResponse(BaseModel):
    """Approval response model."""
    status: str
    message: str
    thread_id: Optional[str] = None


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    if not (BASE_DIR / "templates" / "index.html").exists():
        return HTMLResponse("<h1>HITL Agent API</h1><p>Web template not installed.</p>")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Process a chat message and return response or approval request.

    This endpoint handles the main conversation flow and will return
    either a direct response or an approval request that needs human input.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        logger.info(f"Chat request: thread={thread_id}, user={request.user_id}, message='{request.message}'")

        agent_instance = get_agent()
        graph_config = {"configurable": {"thread_id": thread_id}}
        input_msg = {"messages": [HumanMessage(content=request.message)]}

        try:
            result = await asyncio.to_thread(agent_instance.graph.invoke, input_msg, graph_config)
            last_message = result["messages"][-1]
            response_text = getattr(last_message, "content", str(last_message))
            return ChatResponse(
                thread_id=thread_id,
                status="completed",
                response=response_text,
            )

        except GraphInterrupt as interrupt_exc:
            approval_data: Dict[str, Any] = interrupt_exc.args[0] if interrupt_exc.args else {}
            approval_data.setdefault("thread_id", thread_id)
            approval_data.setdefault("user_id", request.user_id)
            approval_data.setdefault("timestamp", datetime.now().isoformat())

            approval_id = str(uuid.uuid4())
            pending_approvals[approval_id] = {
                "thread_id": thread_id,
                "user_id": request.user_id,
                "approval_data": approval_data,
                "created_at": datetime.now(),
            }

            logger.info(f"Approval request created: {approval_id}")

            return ChatResponse(
                thread_id=thread_id,
                status="waiting_approval",
                approval_id=approval_id,
                approval_data=approval_data,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        audit_logger.log_agent_action(
            thread_id,
            request.user_id,
            "chat_error",
            {"error": str(e), "message": request.message}
        )

        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/api/approve", response_model=ApprovalResponse)
async def approve_purchase(request: ApprovalRequest):
    """
    Process a human approval decision.

    This endpoint handles the human decision on pending approval requests
    and resumes the agent execution.
    """
    try:
        if request.approval_id not in pending_approvals:
            raise HTTPException(status_code=404, detail="Approval request not found")

        approval_info = pending_approvals[request.approval_id]
        thread_id = approval_info["thread_id"]
        user_id = approval_info["user_id"]

        logger.info(f"Approval decision: {request.decision} for thread {thread_id}")

        # Process decision
        if request.decision == "accept":
            message = "Purchase approved and completed successfully."
        elif request.decision == "reject":
            message = "Purchase request rejected."
        elif request.decision == "edit":
            if not request.edit_args:
                raise HTTPException(status_code=400, detail="Edit arguments required for edit decision")
            message = f"Purchase approved with modifications: {request.edit_args}"
        else:
            raise HTTPException(status_code=400, detail=f"Invalid decision: {request.decision}")

        # Log approval decision
        audit_logger.log_purchase_approval(
            thread_id,
            user_id,
            approval_info["approval_data"]["item"],
            approval_info["approval_data"]["price"],
            approval_info["approval_data"]["vendor"],
            request.decision
        )

        # Remove from pending approvals
        del pending_approvals[request.approval_id]

        # In a real implementation, you'd resume the graph execution here
        # For demo purposes, we just return success

        return ApprovalResponse(
            status="success",
            message=message,
            thread_id=thread_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approval processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Approval processing failed: {str(e)}")


@app.get("/api/approvals")
async def list_pending_approvals():
    """List all pending approval requests."""
    return {
        "pending_approvals": [
            {
                "approval_id": aid,
                "thread_id": info["thread_id"],
                "user_id": info["user_id"],
                "item": info["approval_data"]["item"],
                "price": info["approval_data"]["price"],
                "vendor": info["approval_data"]["vendor"],
                "created_at": info["created_at"].isoformat()
            }
            for aid, info in pending_approvals.items()
        ]
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pending_approvals": len(pending_approvals)
    }


@app.delete("/api/approvals/{approval_id}")
async def cancel_approval(approval_id: str):
    """Cancel a pending approval request."""
    if approval_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="Approval request not found")

    approval_info = pending_approvals[approval_id]
    thread_id = approval_info["thread_id"]

    # Log cancellation
    audit_logger.log_agent_action(
        thread_id,
        approval_info["user_id"],
        "approval_cancelled",
        {"approval_id": approval_id}
    )

    del pending_approvals[approval_id]

    return {"status": "cancelled", "message": "Approval request cancelled"}


async def cleanup_expired_approvals():
    """Background task to clean up expired approval requests."""
    while True:
        try:
            current_time = datetime.now()
            expired_ids = []

            for approval_id, info in list(pending_approvals.items()):
                created_at = info["created_at"]
                age_hours = (current_time - created_at).total_seconds() / 3600

                if age_hours > APPROVAL_TIMEOUT_HOURS:
                    expired_ids.append(approval_id)
                    logger.warning(f"Approval request expired: {approval_id}")

                    # Log expired approval
                    audit_logger.log_agent_action(
                        info["thread_id"],
                        info["user_id"],
                        "approval_expired",
                        {"approval_id": approval_id, "age_hours": age_hours}
                    )

            # Remove expired approvals
            for approval_id in expired_ids:
                del pending_approvals[approval_id]

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired approval requests")

        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

        # Run cleanup every hour
        await asyncio.sleep(3600)


def create_templates_and_static():
    """Create basic templates and static files if they don't exist."""
    templates_dir = BASE_DIR / "templates"
    static_dir = BASE_DIR / "static"
    css_dir = static_dir / "css"
    js_dir = static_dir / "js"

    for directory in (templates_dir, css_dir, js_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Create basic HTML template
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HITL Agent Interface</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <h1>Human-in-the-Loop Agent</h1>
        <div id="chat-container">
            <div id="messages"></div>
            <input type="text" id="message-input" placeholder="Type your message...">
            <button id="send-button">Send</button>
        </div>
    </div>
    <script src="/static/js/app.js"></script>
</body>
</html>
"""

    css_content = """
body {
    font-family: Arial, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

h1 {
    color: #333;
    text-align: center;
}

#chat-container {
    margin-top: 20px;
}

#messages {
    height: 400px;
    border: 1px solid #ddd;
    padding: 10px;
    overflow-y: auto;
    margin-bottom: 10px;
    background: #fafafa;
}

.message {
    margin-bottom: 10px;
    padding: 8px;
    border-radius: 4px;
}

.user-message {
    background: #007bff;
    color: white;
    text-align: right;
}

.agent-message {
    background: #e9ecef;
    color: #333;
}

#message-input {
    width: 80%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

#send-button {
    width: 18%;
    padding: 10px;
    background: #28a745;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

#send-button:hover {
    background: #218838;
}
"""

    js_content = """
document.addEventListener('DOMContentLoaded', function() {
    const messagesDiv = document.getElementById('messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');

    let threadId = null;

    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.textContent = content;
        messagesDiv.appendChild(messageDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        addMessage(message, 'user');
        messageInput.value = '';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    thread_id: threadId
                })
            });

            const data = await response.json();

            if (data.status === 'completed') {
                addMessage(data.response, 'agent');
            } else if (data.status === 'waiting_approval') {
                addMessage(`Approval required: ${data.approval_data.message}`, 'agent');
                // In a real implementation, show approval UI here
            } else if (data.error) {
                addMessage(`Error: ${data.error}`, 'agent');
            }

            threadId = data.thread_id;

        } catch (error) {
            addMessage(`Network error: ${error.message}`, 'agent');
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
"""

    (templates_dir / "index.html").write_text(html_content, encoding="utf-8")
    (css_dir / "style.css").write_text(css_content, encoding="utf-8")
    (js_dir / "app.js").write_text(js_content, encoding="utf-8")


if __name__ == "__main__":
    if ENABLE_WEB_INTERFACE:
        create_templates_and_static()
        logger.info(f"Starting web interface on {WEB_HOST}:{WEB_PORT}")
        uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
    else:
        logger.warning("Web interface is disabled. Set ENABLE_WEB_INTERFACE=true to enable.")
