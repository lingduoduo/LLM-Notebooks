"""Offline backend for running the entire pipeline without an OpenAI key.

It validates the mechanism, measures token use and latency, and reproduces the
comparison structure of the three strategies at no cost.

LocalEmbedder provides deterministic local hash bag-of-words embeddings.
MockChatClient is a scripted model compatible with the OpenAI client interface;
it decomposes tasks by keyword and follows the text ReAct protocol.

The mock is a strong heuristic router, not a realistic small model, so it does
not reproduce long-context instruction degradation or generic-tool misuse.
Offline mode faithfully reproduces token counts, one-shot prefilter limitations,
and active discovery's ability to load missing tools on demand. Configure a real
model to observe real behavior against a long tool catalog (see README.md).
"""

import hashlib
import json
import re
from types import SimpleNamespace
from typing import Dict, List, Tuple

from tools_library import TOOLS_BY_NAME

_DIM = 512


# ---------------------------------------------------------------------------
# 1) Local embedding backend
# ---------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    """Tokenize words plus CJK unigrams and bigrams for bag-of-words matching."""
    text = text.lower()
    toks: List[str] = []
    for w in re.findall(r"[a-z0-9]+", text):
        toks.append(w)
    han = re.findall(r"[\u4e00-\u9fff]", text)
    toks += han
    toks += [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return toks


class LocalEmbedder:
    """Deterministic offline hash bag-of-words embeddings."""

    name = "local-hash-%d" % _DIM

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            vec = [0.0] * _DIM
            for tok in _tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % _DIM] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


# ---------------------------------------------------------------------------
# 2) Scripted mock model
# ---------------------------------------------------------------------------
# Intent rules map keywords to the appropriate specialized tool and capability.
# Order matters: forecast must precede general weather.
INTENT_RULES: List[Tuple[str, str, str]] = [
    (r"stock|shares?|equity", "get_stock_price", "retrieve a stock's live price and change"),
    (r"ethereum|bitcoin|crypto|\beth\b|\bbtc\b", "get_crypto_price", "retrieve a cryptocurrency's live price"),
    (r"yen|exchange rate|convert.*(usd|jpy|eur)|currency", "get_forex_rate", "retrieve a currency exchange rate"),
    (r"papers?|arxiv|literature|quantum computing|research progress", "arxiv_search", "search for recent academic papers"),
    (r"download", "download_file", "download a URL to local storage"),
    (r"contributors?|contributions?", "github_list_contributors", "retrieve GitHub contributor statistics"),
    (r"charts?|visuali[sz]ation|plot|graph", "render_chart", "render a data visualization"),
    (r"forecast|future|sunday|this week|tomorrow|next week", "get_weather_forecast", "retrieve a multi-day weather forecast"),
    (r"weather", "get_current_weather", "retrieve current city weather"),
    (r"calendar|schedule|event|remind", "create_calendar_event", "create a calendar event"),
    (r"news|sentiment|reports?|coverage|opinion", "search_news", "search relevant recent news"),
]


def match_intents(prompt: str) -> List[Tuple[str, str]]:
    """Return deduplicated (specialized tool, capability need) pairs in order."""
    needed: List[Tuple[str, str]] = []
    seen = set()
    for pat, tool, phrase in INTENT_RULES:
        if re.search(pat, prompt, re.IGNORECASE) and tool not in seen:
            needed.append((tool, phrase))
            seen.add(tool)
    # If forecast matched, do not separately require current weather.
    if "get_weather_forecast" in seen and "get_current_weather" in seen:
        needed = [(t, p) for t, p in needed if t != "get_current_weather"]
    return needed


_ARG_HINTS = {
    "symbol": "AAPL", "location": "Beijing", "query": "search", "url": "https://example.com/f.pdf",
    "path": "/tmp/paper.pdf", "owner": "pytorch", "repo": "pytorch", "base": "USD",
    "quote": "JPY", "title": "Outdoor hike", "start": "2026-07-19T09:00", "end": "2026-07-19T12:00",
    "days": 3, "data": "[]", "chart_type": "bar", "code": "print('ok')", "max_results": 3,
}


def _fill_args(tool_name: str) -> Dict:
    tool = TOOLS_BY_NAME.get(tool_name)
    if not tool:
        return {}
    props = tool["function"]["parameters"]["properties"]
    args = {}
    for key, spec in props.items():
        if key in _ARG_HINTS:
            args[key] = _ARG_HINTS[key]
        elif spec.get("type") == "integer":
            args[key] = 1
        else:
            args[key] = "auto"
    return args


def _extract_json(text: str):
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _json(thought: str, tool: str, arguments: Dict) -> str:
    return json.dumps({"thought": thought, "tool": tool, "arguments": arguments},
                      ensure_ascii=False)


class MockChatClient:
    """Deterministic scripted model compatible with an OpenAI client subset."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model=None, messages=None, temperature=0, **kw):
        content = self._respond(messages or [])
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def _respond(self, messages: List[Dict]) -> str:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        task_prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        full_text = "\n".join(m.get("content", "") for m in messages)
        has_discover = "discover_tools" in system

        # Available tools are names appearing in injected or discovered text.
        available = set(re.findall(r'"name":\s*"([a-zA-Z_][a-zA-Z0-9_]*)"', full_text))
        available.discard("discover_tools")

        prior = []
        for m in messages:
            if m["role"] == "assistant":
                a = _extract_json(m.get("content", ""))
                if a and "tool" in a:
                    prior.append(a)
        called_ok = {a["tool"] for a in prior if a["tool"] in available}
        discover_needs = [((a.get("arguments") or {}).get("need", ""))
                          for a in prior if a.get("tool") == "discover_tools"]
        attempted = [a["tool"] for a in prior
                     if a["tool"] not in available and a["tool"] not in ("discover_tools", "finish")]

        for tool, phrase in match_intents(task_prompt):
            if tool in called_ok:
                continue
            if tool in available:
                return _json(f"Call specialized tool {tool}", tool, _fill_args(tool))
            # Target tool is currently unavailable.
            if has_discover:
                if discover_needs.count(phrase) >= 1:
                    continue  # Already discovered but still absent; skip it.
                return _json(f"I need a tool that can {phrase}; discover it first", "discover_tools",
                             {"need": phrase})
            else:
                if attempted.count(tool) >= 1:
                    continue  # Not in the catalog; abandon after one attempt.
                return _json(f"The task requires {tool}; attempt the call", tool, _fill_args(tool))
        return _json("All subtasks have been handled", "finish", {"answer": "Completed all feasible subtasks."})
