"""
tools.py: specialist tool implementations and OpenAI function-calling schemas.

Design principles for Experiment 10-1:
- Every tool used by the experiment performs real work; searches never use canned answers.
- research.web_search: live Tavily searches with attributable URLs and excerpts.
- coding.execute_python: execute Python and capture stdout in a subprocess with a timeout.
- data_analysis.calculate / descriptive_stats: real, safe calculations.
- writing.count_characters: actual character counts for Chinese and English text.

Each tool has the signature func(**kwargs) -> str, returning a string for the conversation history.
"""

from __future__ import annotations

import ast
import json
import operator
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional


# Keep live campaigns bounded when a provider stalls.  The value is configurable
# for readers running in a slower network, while the default is short enough that
# one unavailable search cannot consume the whole paired comparison.
TAVILY_TIMEOUT_SECONDS = float(os.environ.get("TAVILY_TIMEOUT_SECONDS", "20"))
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "5"))
TAVILY_CONTENT_CHARS = int(os.environ.get("TAVILY_CONTENT_CHARS", "1400"))


# ---------------------------------------------------------------------------
# research role: web_search - live Tavily search
# ---------------------------------------------------------------------------

def web_search(query: str, receipt_sink: Optional[Callable[[dict], None]] = None) -> str:
    """Run a real Tavily web search and return attributable source excerpts."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("web_search requires TAVILY_API_KEY; no mock fallback is allowed")
    body = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": TAVILY_MAX_RESULTS,
        "include_answer": True,
        "include_raw_content": False,
    }
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=TAVILY_TIMEOUT_SECONDS) as response:
            status = response.status
            raw_response = response.read().decode("utf-8", "replace")
            payload = json.loads(raw_response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"Tavily HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Tavily request failed: {exc}") from None
    if receipt_sink:
        receipt_sink({
            "kind": "tavily_search",
            "request": {
                "method": "POST",
                "url": "https://api.tavily.com/search",
                "headers": {"Content-Type": "application/json"},
                "body": {key: value for key, value in body.items() if key != "api_key"},
            },
            "response": {
                "http_status": status,
                "raw_body": raw_response,
            },
            "duration_seconds": round(time.monotonic() - started, 3),
        })
    results = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            # Search snippets are evidence pointers, not a second context
            # window.  Bound their size so repeated role transitions do not
            # make later API requests quadratic in prompt length.
            "content": str(item.get("content") or "")[:TAVILY_CONTENT_CHARS],
            "score": item.get("score"),
        })
    if not results:
        return json.dumps({
            "provider": "tavily",
            "query": query,
            "answer": payload.get("answer"),
            "results": [],
        }, ensure_ascii=False)
    return json.dumps({
        "provider": "tavily",
        "query": query,
        "answer": payload.get("answer"),
        "results": results,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# coding role: execute_python - run code and capture stdout with a timeout
# ---------------------------------------------------------------------------

def execute_python(code: str, timeout: int = 10) -> str:
    """Write source to a temporary file, run it in a subprocess with a timeout, and return stdout."""
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "snippet.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return f"Execution timed out (>{timeout}s)"
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return (
                f"Code execution failed: exit code {proc.returncode}\n"
                f"stderr:\n{err}\n"
                f"Captured output:\n{out}"
            )
        return out if out else "(Code executed, but produced no print output)"


# ---------------------------------------------------------------------------
# data_analysis role: calculate (safe expression evaluation) + descriptive_stats
# ---------------------------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Safely evaluate arithmetic, powers, and modulo without Python's built-in eval."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expression contains unsupported operations; only + - * / ** % and parentheses are allowed.")


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression, such as (949.5/352.1)**(1/2)-1."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except Exception as exc:  # noqa: BLE001
        return f"Calculation failed: {exc}"
    return f"{expression} = {result}"


def descriptive_stats(numbers: List[float]) -> str:
    """Return basic descriptive statistics (mean, maximum, minimum, and range) for a list of numbers."""
    if not numbers:
        return "Input is empty; cannot calculate statistics."
    nums = [float(x) for x in numbers]
    n = len(nums)
    mean = sum(nums) / n
    return (
        f"Sample size={n}, mean={mean:.4f}, minimum={min(nums)}, "
        f"maximum={max(nums)}, range={max(nums) - min(nums)}"
    )


# ---------------------------------------------------------------------------
# writing role: count_characters - character counts for Chinese and English text
# ---------------------------------------------------------------------------

def count_characters(text: str) -> str:
    """Count total characters and Chinese characters to help control text length."""
    if text is None:
        text = ""
    total = len(text)
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return f"Total characters={total}, Chinese characters={chinese}"


# ---------------------------------------------------------------------------
# Tool registry: name -> (implementation function, OpenAI schema)
# ---------------------------------------------------------------------------

# OpenAI function-calling schema for each tool.
TOOL_SCHEMAS: Dict[str, dict] = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web using Tavily and return source excerpts with URLs. Use for data, facts, and reference material.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords or question"},
                },
                "required": ["query"],
            },
        },
    },
    "execute_python": {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return its print output. Useful for scripts and program logic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to execute; use print to output results"},
                },
                "required": ["code"],
            },
        },
    },
    "calculate": {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Safely evaluate a mathematical expression supporting + - * / ** % and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression, such as (949.5/352.1)**(1/2)-1"},
                },
                "required": ["expression"],
            },
        },
    },
    "descriptive_stats": {
        "type": "function",
        "function": {
            "name": "descriptive_stats",
            "description": "Calculate basic descriptive statistics (mean, maximum, minimum, and range) for a list of numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numbers",
                    },
                },
                "required": ["numbers"],
            },
        },
    },
    "count_characters": {
        "type": "function",
        "function": {
            "name": "count_characters",
            "description": "Count total characters and Chinese characters to help control text length.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to count"},
                },
                "required": ["text"],
            },
        },
    },
}

# Tool name -> implementation function
TOOL_IMPLEMENTATIONS: Dict[str, Callable[..., str]] = {
    "web_search": web_search,
    "execute_python": execute_python,
    "calculate": calculate,
    "descriptive_stats": descriptive_stats,
    "count_characters": count_characters,
}
