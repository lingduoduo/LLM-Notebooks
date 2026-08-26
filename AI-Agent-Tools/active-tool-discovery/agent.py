"""Agent loops for three tool-discovery strategies (text/ReAct protocol).

Why use text injection and text-parsed tool calls instead of native OpenAI
function calling? This experiment reproduces the book's setup: inject 120+
tool schemas into the system prompt at once (tens of thousands of tokens) and
observe instruction-following degradation in a very long context. Native
function calling strongly constrains and optimizes tool selection, so errors
remain rare even with hundreds of tools. Plain-text schemas and model-emitted
JSON calls reproduce the control mechanism and make degradation observable.

At each step the model outputs exactly one JSON object:
    {"thought": "...", "tool": "tool_name", "arguments": {...}}
When the task is complete:
    {"thought": "...", "tool": "finish", "arguments": {"answer": "..."}}

1) run_full_injection: control group; all 126 schemas appear in the system
   prompt. injected_tokens is the token count of that catalog text.
2) run_retrieval_prefilter: second control; a one-shot semantic search of the
   initial query injects only top-n candidates. It saves tokens but cannot
   anticipate cross-domain needs that emerge during execution.
3) run_active_discovery: treatment; the system prompt contains a few base tools
   plus discover_tools. Calls to discover_tools(need) retrieve 3-5 candidates,
   append their schemas as a user message (preserving the system-prefix KV
   cache), and update the available-tool status bar.
"""

import json
import re
from typing import Dict, List

import tiktoken

from discovery import ToolIndex  # noqa: F401  (used for type hints)
from tools_library import ALL_TOOLS, BASE_TOOL_NAMES, TOOL_IMPLS

try:
    _ENC = tiktoken.get_encoding("o200k_base")  # gpt-4o family encoding
except Exception:
    _ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Tool-catalog rendering and token counting
# ---------------------------------------------------------------------------

def render_tool(tool: Dict) -> str:
    """Render one tool as the complete JSON schema injected into the prompt."""
    return json.dumps(tool["function"], ensure_ascii=False, indent=2)


def render_tools(tools: List[Dict]) -> str:
    return "\n".join(render_tool(t) for t in tools)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text)) if text else 0


# discover_tools meta-tool (also presented to the model as text)
DISCOVER_TOOL = {
    "type": "function",
    "function": {
        "name": "discover_tools",
        "description": ("Discover new tools when no suitable specialized tool is available. "
                        "Describe the required capability in natural language as `need`; "
                        "semantic retrieval returns and loads the best-matching tools."),
        "parameters": {"type": "object",
                       "properties": {"need": {"type": "string"}}, "required": ["need"]},
    },
}

FINISH_TOOL_DESC = "- finish(answer: string): Call after all subtasks are complete to give the final answer."


_PROTOCOL = (
    "At every step, output exactly one JSON object and no extra text, in this format:\n"
    '{"thought": "brief reasoning", "tool": "tool_name", "arguments": {"key": "value"}}\n'
    "The system executes the tool and returns its result before your next step.\n"
    "Only after every subtask has been completed with an appropriate tool, output:"
    '{"thought": "...", "tool": "finish", "arguments": {"answer": "final answer"}}\n'
    "Choose the best specialized tool for each subtask, not a broad generic search tool."
)


def _extract_json(text: str):
    """Extract the first JSON object from a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _tool_succeeded(result: str) -> bool:
    """Return whether a mock/tool result represents a successful execution."""
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return False
    if (not isinstance(payload, dict) or payload.get("error")
            or payload.get("success") is False):
        return False
    return str(payload.get("status", "ok")).lower() not in {"error", "failed", "failure"}


def _valid_action(action) -> bool:
    if (not isinstance(action, dict) or not isinstance(action.get("tool"), str)
            or not isinstance(action.get("arguments"), dict)):
        return False
    args = action["arguments"]
    if action["tool"] == "finish":
        return isinstance(args.get("answer"), str) and bool(args["answer"].strip())
    if action["tool"] == "discover_tools":
        return isinstance(args.get("need"), str) and bool(args["need"].strip())
    return True


def _run_loop(client, model, system_prompt, task_prompt, available_names,
              on_discover=None, max_steps=10):
    """
    Text-based ReAct loop.
    available_names: names currently callable, excluding discover_tools/finish;
      grows dynamically after discover_tools calls in active-discovery mode.
    Returns (called_tools, successful_tools, trace, finished).
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}]
    called: List[str] = []
    successful: List[str] = []
    trace: List[str] = []
    finished = False

    for _ in range(max_steps):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0)
        except Exception as e:
            # Some reasoning models (such as gpt-5.x) support only the default
            # temperature=1; retry without an explicit temperature.
            if "temperature" in str(e):
                resp = client.chat.completions.create(
                    model=model, messages=messages)
            else:
                raise
        content = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})

        action = _extract_json(content)
        if not _valid_action(action):
            trace.append(f"[format error] Model did not output valid JSON: {content[:80]!r}")
            messages.append({"role": "user",
                             "content": "Your response is not valid JSON. Output only the required JSON object."})
            continue

        name = action.get("tool")
        args = action.get("arguments") or {}

        if name == "finish":
            trace.append(f"[finish] {str(args.get('answer',''))[:100]}")
            finished = True
            break

        if name == "discover_tools" and on_discover is not None:
            need = args.get("need", "")
            result_text, new_names = on_discover(need)
            called.append(name)
            trace.append(f"[discover_tools] need='{need}' -> {new_names}")
            available_names.update(new_names)
            messages.append({"role": "user", "content": result_text})
            continue

        # Regular tool call.
        if name not in available_names:
            # Unavailable because it was not discovered/prefiltered or was
            # hallucinated. Do not count it as called because it never ran.
            trace.append(f"[unavailable] {name}")
            hint = ("That tool is currently unavailable. "
                    + ("First use discover_tools to find the required capability." if on_discover else
                       "Choose an existing tool from the catalog."))
            messages.append({"role": "user", "content": hint})
            continue

        called.append(name)
        impl = TOOL_IMPLS.get(name)
        result = impl(args) if impl else json.dumps({"error": f"unknown tool {name}"})
        if _tool_succeeded(result):
            successful.append(name)
        trace.append(f"[call] {name}({json.dumps(args, ensure_ascii=False)})")
        messages.append({"role": "user", "content": f"Tool {name} returned: {result}"})

    return called, successful, trace, finished


# ---------------------------------------------------------------------------
# Control group: full injection
# ---------------------------------------------------------------------------

def run_full_injection(client, model, task_prompt: str, tools: List[Dict] = None,
                       max_steps: int = 10) -> Dict:
    tools = tools if tools is not None else ALL_TOOLS
    tools_text = render_tools(tools) + "\n" + FINISH_TOOL_DESC
    injected = count_tokens(tools_text)
    system = (
        f"You are an intelligent assistant. Below is the complete catalog of {len(tools)} available tools. "
        "Choose the most appropriate tools for the task and handle every subtask.\n\n"
        "[TOOL CATALOG]\n" + tools_text + "\n\n" + _PROTOCOL
    )
    available = {t["function"]["name"] for t in tools}
    called, successful, trace, finished = _run_loop(
        client, model, system, task_prompt, available, max_steps=max_steps
    )
    return {"mode": "full_injection", "injected_tokens": injected,
            "num_tools_exposed": len(tools), "called": called,
            "successful": successful,
            "trace": trace, "finished": finished}


# ---------------------------------------------------------------------------
# Second control: retrieval prefilter (the book's retrieval-based prefilter)
# Perform one semantic search against the initial query and inject only top-n
# candidates. This saves tokens but cannot anticipate cross-domain needs that
# emerge during execution. A missed tool for a later subtask cannot be called.
# ---------------------------------------------------------------------------

def run_retrieval_prefilter(client, model, task_prompt: str, index, top_n: int = 10,
                            tools: List[Dict] = None, max_steps: int = 10) -> Dict:
    tools = tools if tools is not None else ALL_TOOLS
    tbn = {t["function"]["name"]: t for t in tools}
    hits = index.search(task_prompt, top_k=top_n)
    picked = [name for name, _ in hits if name in tbn]
    picked_tools = [tbn[n] for n in picked]
    tools_text = render_tools(picked_tools) + "\n" + FINISH_TOOL_DESC
    injected = count_tokens(tools_text)
    system = (
        f"You are an intelligent assistant. The system pre-retrieved {len(picked_tools)} potentially relevant tools. "
        "Choose from them to complete the task. If a subtask has no suitable tool in the catalog, say so.\n\n"
        "[TOOL CATALOG]\n" + tools_text + "\n\n" + _PROTOCOL
    )
    available = set(picked)
    called, successful, trace, finished = _run_loop(
        client, model, system, task_prompt, available, max_steps=max_steps
    )
    return {"mode": "retrieval_prefilter", "injected_tokens": injected,
            "num_tools_exposed": len(picked_tools), "prefiltered": picked,
            "called": called, "successful": successful,
            "trace": trace, "finished": finished}


# ---------------------------------------------------------------------------
# Treatment group: active discovery
# ---------------------------------------------------------------------------

def run_active_discovery(client, model, task_prompt: str, index, top_k=4,
                         tools: List[Dict] = None, max_steps: int = 10) -> Dict:
    tools = tools if tools is not None else ALL_TOOLS
    tbn = {t["function"]["name"]: t for t in tools}
    base_tools = [tbn[n] for n in BASE_TOOL_NAMES]
    base_text = (render_tools(base_tools) + "\n"
                 + render_tool(DISCOVER_TOOL) + "\n" + FINISH_TOOL_DESC)

    discovered_names = set()          # Specialized tools loaded in this run.
    injected_schema_blocks: List[str] = []  # Every schema block appended to history.
    available = set(BASE_TOOL_NAMES)

    def on_discover(need: str):
        hits = index.search(need, top_k=top_k)
        names, lines = [], []
        for name, score in hits:
            if name in BASE_TOOL_NAMES:
                continue
            names.append(name)
            schema_text = render_tool(tbn[name])
            lines.append(schema_text + f"   (similarity {score:.3f})")
            injected_schema_blocks.append(schema_text)
            if name not in discovered_names:
                discovered_names.add(name)
        status = f"\n\n[STATUS BAR | AVAILABLE TOOLS] {sorted(available | set(names))}"
        body = ("discover_tools matched and loaded these specialized tools; they are ready to call:\n"
                + "\n".join(lines) + status)
        return body, names

    system = (
        "You are an intelligent assistant with only the small set of base tools listed below. "
        "When a task requires a missing capability, call discover_tools first and describe the need in natural language. "
        "The system will return and load matching specialized tools for you to call. "
        "For multi-part tasks, discover each distinct capability separately and verify that every subtask is complete before finishing.\n\n"
        "[BASE TOOLS]\n" + base_text + "\n\n" + _PROTOCOL
    )
    called, successful, trace, finished = _run_loop(
        client, model, system, task_prompt, available,
        on_discover=on_discover, max_steps=max_steps
    )

    injected = count_tokens(base_text) + count_tokens("\n".join(injected_schema_blocks))
    return {"mode": "active_discovery", "injected_tokens": injected,
            "num_tools_exposed": len(BASE_TOOL_NAMES) + 1 + len(discovered_names),
            "discovered": sorted(discovered_names),
            "called": called, "successful": successful,
            "trace": trace, "finished": finished}
