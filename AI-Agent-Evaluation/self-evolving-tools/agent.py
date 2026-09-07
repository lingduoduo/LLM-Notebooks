"""Alita-style self-evolving agent.

Only five base tools are predefined:
    web_search / read_webpage / code_interpreter / create_tool / search_tools
No domain tools are included. The agent must analyze the task, identify missing
capabilities, find libraries/APIs, read documentation, test in the sandbox,
package the solution in the library, and use the new tool to complete the task.
For similar tasks, first use search_tools to reuse tools instead of rebuilding.

Uses function calling through the OpenAI SDK, defaulting to gpt-5.6-luna.
Select an OpenAI-compatible provider with LLM_PROVIDER=openai|moonshot|ark.
If its key is absent but OPENROUTER_API_KEY is set, fall back to OpenRouter.
"""

import json
import os

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import base_tools
from tool_manager import ToolLibrary, normalize_schema

# --------------------------------------------------------------------------- #
# LLM clients: OpenAI, Moonshot, and ARK use OpenAI-compatible interfaces
# --------------------------------------------------------------------------- #
_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", None, "gpt-5.6-luna"),
    "moonshot": ("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1", "kimi-k3"),
    "ark": ("ARK_API_KEY", "https://ark.cn-beijing.volces.com/api/v3", "doubao-seed-1-6-250615"),
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _to_openrouter_model(model: str) -> str:
    """Map common model names to the OpenRouter namespace."""
    if not model:
        return "openai/gpt-5.6-luna"
    if "/" in model:
        return model
    if model.startswith("gpt-"):
        return "openai/" + model
    if model.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    return "openai/gpt-5.6-luna"


def build_client():
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    key_env, base_url, default_model = _PROVIDERS.get(provider, _PROVIDERS["openai"])
    model = os.environ.get("LLM_MODEL", default_model)
    api_key = os.environ.get(key_env)
    # Fall back to OpenRouter when the provider key is missing but its key is available
    if not api_key and os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
        return client, _to_openrouter_model(model)
    if not api_key:
        raise RuntimeError(
            f"missing {key_env} in environment (provider={provider})；"
            f"OPENROUTER_API_KEY is also unset (OpenRouter can serve as a fallback)."
        )
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return client, model


SYSTEM_PROMPT = """You are an Alita-style self-evolving agent. You have only five base tools:
web_search, read_webpage, code_interpreter, create_tool, and search_tools.
You have no ready-made domain tools for stock prices, subtitles, or similar tasks.
Your mission is to **build reusable tools** for missing capabilities, becoming more
capable with use instead of assembling a one-off answer each time.

Strictly follow this pipeline:

Step 0 (reuse first): Call **search_tools** to check for a tool that can do the task.
    - If found, call that packaged tool directly to obtain data and answer.
      Do **not** call web_search or create_tool to rebuild it. Tool reuse is required.
    - If no tool matches, follow evolution steps 1-5 below.

Evolution workflow (when no suitable library tool exists):
1. Use web_search to find **open-source Python libraries** callable from code.
   Search with terms such as "open source python library" or "python package"
   instead of "API", since online APIs often require registering for keys.
   Prefer mature libraries that install with pip, require no keys, and fetch
   public data automatically; these exist for many data types, including markets.
2. Use read_webpage to read candidate README files, PyPI pages, or documentation
   to learn installation and usage. Use it only for documentation, not to scrape
   the numbers for the final answer.
3. Use code_interpreter to **actually execute code** in the sandbox and verify
   the library, installing dependencies with pip_install.
   **Strict constraint**: prefer Python libraries that need no API key and can
   be called offline after pip installation. Skip online APIs requiring key
   registration or apikey/token parameters; choose a free, keyless library.
   Verification succeeds only when test code uses print to output **real price
   numbers**, not placeholders, errors, or empty output. Only printed real data
   counts as verification. Never fabricate data, and do not give up easily.
4. After verification, use create_tool to package a **generic, reusable** tool.
   Parameterize it, for example by ticker rather than hard-coding one stock.
   Use a generic name such as get_stock_price rather than get_nvidia_price, and
   a general description so future searches can find it.
   Define def run(**kwargs) in code and return structured results. The tool must
   **actually call the verified library** to retrieve fresh data at runtime.
   **Always include test_args** with create_tool: the system executes
   run(**test_args) before registration and only admits tools that run successfully,
   preventing broken tools from contaminating the library.
   After verification you **must** perform this create_tool step before answering.
   Do not skip packaging and answer directly.
5. Call the tool you just created and answer the user with **real data**.

Mandatory requirements (violations count as failure):
- For live or structured data tasks, do **not** answer solely using numbers from a
  page fetched by read_webpage. Follow find library -> test -> package -> call tool
  to make the capability reusable and reliable.
- **Never fabricate data**: do not hard-code prices, dates, or other data in tool
  code, and do not use simulated, sample, or mock data. Tools must obtain current
  data through the library at runtime. Do **not** call create_tool before printing
  real numbers through code_interpreter.
- If no usable free library can be found, explain the failure honestly and do not
  invent a numerical answer.
- Give the final answer in English and identify the data source (library/tool)."""


# --------------------------------------------------------------------------- #
# Function-calling schemas for base tools
# --------------------------------------------------------------------------- #
BASE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web with DuckDuckGo and return titles, URLs, and snippets to find open-source libraries or public APIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "num_results": {"type": "integer", "description": "Number of results (1-10)", "default": 6},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Fetch a page and extract its main text to read README files or API documentation.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Webpage URL"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Execute Python in a subprocess sandbox to verify a solution. Optionally install dependencies with pip_install. Returns stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "pip_install": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional list of packages to install with pip before execution",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_tool",
            "description": "Package a verified capability as a standard tool and persist it in the library. Code must define def run(**kwargs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool name (valid Python identifier)"},
                    "description": {"type": "string", "description": "Tool purpose description for future retrieval"},
                    "parameters": {
                        "type": "object",
                        "description": "Tool parameter JSON Schema (type=object, properties, required)",
                    },
                    "code": {"type": "string", "description": "Tool implementation; must define def run(**kwargs) and return a JSON-serializable result"},
                    "test_args": {
                        "type": "object",
                        "description": "Example arguments for pre-save validation: run(**test_args) executes before registration. "
                                       "Only successful tools are admitted. Strongly recommended to keep broken tools out of the library.",
                    },
                },
                "required": ["name", "description", "parameters", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tools",
            "description": "Search existing library tools by keyword for reuse. Call this before searching the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search keywords, such as 'stock price'"}},
                "required": ["query"],
            },
        },
    },
]


class SelfEvolvingAgent:
    def __init__(self, verbose: bool = True, allow_create: bool = True, model: str | None = None):
        self.client, self.model = build_client()
        if model:  # CLI/caller model overrides take precedence over LLM_MODEL
            self.model = model
        self.library = ToolLibrary()
        self.verbose = verbose
        # Control tool creation during self-evolution; False removes create_tool.
        # Use this for comparisons where the agent can only reuse tools or fail the task.
        self.allow_create = allow_create
        self.trajectory = []  # Record actions to demonstrate tool reuse
        self._verified_real_data = False  # Whether code_interpreter printed real data during this task
        self._created_tool = False         # Whether a tool was created during this task
        self._used_library_tool = False    # Whether a packaged library tool was used during this task
        # Unlock library tools only after search_tools finds them or create_tool creates them.
        # Then expose them as callable functions, enforcing retrieval before reuse.
        self._unlocked = set()

    # ------------------------------------------------------------------ #
    def _tools(self):
        """Expose the base tools plus tools found or created during the current task."""
        base = BASE_TOOL_SCHEMAS
        if not self.allow_create:  # Disable tool creation by omitting create_tool from the exposed tools
            base = [s for s in base if s["function"]["name"] != "create_tool"]
        dynamic = []
        for rec in self.library.list_tools():
            if rec["name"] not in self._unlocked:
                continue
            dynamic.append(
                {
                    "type": "function",
                    "function": {
                        "name": rec["name"],
                        "description": "[Packaged tool] " + rec["description"],
                        "parameters": normalize_schema(rec["parameters"]),
                    },
                }
            )
        return base + dynamic

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------ #
    def _dispatch(self, name: str, args: dict) -> dict:
        """Execute a tool call and record it in the trajectory."""
        self.trajectory.append(name)
        if name == "web_search":
            return base_tools.web_search(args.get("query", ""), args.get("num_results", 6))
        if name == "read_webpage":
            return base_tools.read_webpage(args.get("url", ""))
        if name == "code_interpreter":
            res = base_tools.code_interpreter(args.get("code", ""), args.get("pip_install"))
            # Count real-data verification only when execution succeeds with nonempty output
            if res.get("success") and res.get("stdout", "").strip():
                self._verified_real_data = True
            return res
        if name == "create_tool":
            if not self.allow_create:
                return {"success": False, "error": "Tool creation is disabled for this run (--no-create)."}
            code = args.get("code", "")
            # Anti-hallucination guard 1: require printed real data before packaging a tool
            if not self._verified_real_data:
                return {
                    "success": False,
                    "error": "Real data has not been verified. First use code_interpreter to call the library and print "
                             "real numbers, then package the tool after verification. Do not package unverified or fabricated data.",
                }
            # Anti-hallucination guard 2: reject signs of simulated, sample, or hard-coded data
            lowered = code.lower()
            if any(k in lowered for k in ("mock", "\u6a21\u62df", "\u793a\u4f8b\u6570\u636e", "sample data", "fake", "dummy")):
                return {
                    "success": False,
                    "error": "Tool code appears to contain simulated, sample, or hard-coded data. Tools must retrieve real "
                             "data through a library at runtime. Replace it with a real library call and resubmit.",
                }
            res = self.library.create_tool(
                args.get("name", ""), args.get("description", ""),
                args.get("parameters", {}), code, args.get("test_args"),
            )
            if res.get("success"):
                self._created_tool = True
                self._unlocked.add(res["name"])  # Unlock newly created tools immediately for use during this task
            return res
        if name == "search_tools":
            res = self.library.search_tools(args.get("query", ""))
            for t in res.get("tools", []):  # Unlock search matches as callable functions for reuse
                self._unlocked.add(t["name"])
            return res
        # Otherwise, call a packaged tool for reuse
        if self.library.get_tool(name) is not None:
            self._used_library_tool = True
            return self.library.execute_tool(name, args)
        return {"success": False, "error": f"unknown tool: {name}"}

    # ------------------------------------------------------------------ #
    def run(self, task: str, max_steps: int = 20) -> str:
        self._verified_real_data = False
        self._created_tool = False
        self._used_library_tool = False
        self._unlocked = set()
        nudges = 0  # Limit reminders to package the tool to avoid an infinite loop
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        self._log(f"\n{'='*70}\n[Task] {task}\n{'='*70}")

        for step in range(max_steps):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                tools=self._tools(), tool_choice="auto", temperature=0,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                # Evolution guard: if real data was verified but no tool was created or reused,
                # require create_tool before answering to persist the new capability.
                if (
                    self._verified_real_data
                    and not self._created_tool
                    and not self._used_library_tool
                    and nudges < 2
                ):
                    nudges += 1
                    self._log("\n[Evolution guard] Real data verified but no tool packaged; reminding the model to call create_tool first.")
                    messages.append(
                        {
                            "role": "user",
                            "content": "You verified the solution with real data but have not packaged it as a reusable tool. "
                            "**Call create_tool first** with a generic name, ticker parameter, and real internal library calls. "
                            "Then call the new tool to obtain real data before answering.",
                        }
                    )
                    continue
                self._log(f"\n[Final answer]\n{msg.content}")
                return msg.content or ""

            for tc in msg.tool_calls:
                fname = tc.function.name
                try:
                    fargs = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fargs = {}
                self._log(f"\n[step {step+1}] Calling tool -> {fname}  args={_short(fargs)}")
                result = self._dispatch(fname, fargs)
                self._log(f"           Result: {_short(result)}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str)[:8000],
                    }
                )

        return "(Maximum step limit reached)"


def _short(obj, n: int = 240) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + f"...(+{len(s)-n} chars)"
