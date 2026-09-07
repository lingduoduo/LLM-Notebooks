# Supplementary Case: Agent Finds and Validates Tools from the Web (Alita Style)

> Companion code for "Deep Understanding of AI Agents" · ★★★
> Core concept: **"Minimum predefined, maximum self-evolution"**.

## Purpose

The capability ceiling of most agents is determined by "human-predefined tools". This experiment takes the opposite approach: the agent **has no domain-specific tools predefined**, only five generic "meta-tools". When it encounters a task it cannot handle, it will search the web for **open-source libraries/APIs**, **read documentation**, **test in a sandbox**, **package the viable solution as a new tool and store it in the tool library**, and then use the new tool to complete the task—evolving like Alita. When encountering a similar task again, it will first **reuse** the already-built tool from the library instead of reinventing the wheel.

The entire process emphasizes **hallucination control**: all numbers and conclusions must come from real search results, documentation, or code execution output.

## Five Base Tools (No Domain-Specific Tools)

| Tool | Purpose | Implementation |
| --- | --- | --- |
| `web_search` | Search for open-source libraries / APIs | DuckDuckGo, **no API key required** (lite + html dual endpoints, with backoff retry) |
| `read_webpage` | Read README / API documentation | requests + BeautifulSoup to extract main text |
| `code_interpreter` | Actually execute code in a sandbox to verify the solution | **Subprocess sandbox** + timeout; can `pip_install` to a temporary directory |
| `create_tool` | Package a verified function as a standard tool and persist it | Write to `tool_library/<name>.json` (metadata + code) |
| `search_tools` | Search the tool library by name/description for **reuse** | Keyword matching |

## Self-Evolution Pipeline

```text
Analyze task
  → search_tools (first check if a reusable tool already exists in the library)
      Hit ─────────────────► Directly call that tool to answer (tool reuse)
      Miss ↓
  → web_search  find open-source Python libraries requiring no API key
  → read_webpage read README / PyPI documentation
  → code_interpreter actually run in sandbox, print real data (can pip install dependencies)
  → create_tool package as a "generic, parameterized" standard tool
      └─ Pre-save validation: syntax compilation + actually run run() once with test_args, only register if it passes
  → Call the new tool, answer with real data
```

To suppress hallucinations and "laziness", several **guardrails** are built into the code:

- Without `code_interpreter` printing real data, `create_tool` is **prohibited**;
- If the code for `create_tool` contains words like `mock / simulated / sample data / fake`, it is **rejected** from the library;
- When real data has been verified but the agent tries to skip packaging and answer directly, it is forced to `create_tool` first;
- A tool in the library must first be hit by `search_tools` (or just created) to be "unlocked" as callable—thus enforcing the "retrieve before reuse" flow.

There is also a **"pre-save validation" gate** (corresponding to the "Test" step in the pipeline of Figure 8-7, and also addressing the chapter's warning about "tool quality degradation"—bad tools propagate errors to subsequent tasks through reuse): before `create_tool` persists the tool to disk, it will

- First perform a **syntax compilation check**; code with syntax errors is blocked from the library;
- If the caller provides `test_args` (a set of example input parameters), it will **actually run `run(**test_args)` once** in the sandbox; only if it successfully returns a result is it allowed into the library. The system prompt requires the model to provide `test_args` when creating a tool, thus keeping "broken tools that don't run" out of the tool library, rather than waiting for them to crash when reused by a subsequent task.

## Running

```bash
# From the repository root: use the shared Chapter 8 environment
uv sync --locked --python 3.12 --extra ch8
# Apple Silicon macOS needs macOS 14+ for the locked bitsandbytes wheel;
# older macOS users should use the single-project compatibility path below.

# Activate it before changing directories:
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows cmd: .venv\Scripts\activate.bat

# pip fallback when uv is not installed:
# python -m pip install -e ".[ch8]"

cd AI-Agent-Evaluation/self-evolving-tools

# Single-project compatibility path, still supported during migration:
# python -m pip install -r requirements.txt

cp env.example .env        # Fill in OPENAI_API_KEY (default model gpt-5.6-luna)
# Fallback: if no OPENAI_API_KEY but OPENROUTER_API_KEY is set, automatically switch to OpenRouter (maps to openai/gpt-5.6-luna, etc.)
python demo.py             # Run the two default "evolution + reuse" tasks (requires API + internet)
python demo.py --fresh     # Clear tool_library/ first, then run, reproducing "evolution from scratch" (recommended for repeated demos)
python demo.py --offline   # Offline mechanism self-check: no API/network required, verify the evolution loop itself
python demo.py --help      # View all parameters
```

**Command-line arguments** (`--help`):

| Argument | Purpose |
| --- | --- |
| `--task task description` | Custom task; can be repeated multiple times to run several tasks in sequence. If not given, runs the default NVDA/AAPL two tasks |
| `--offline` | Offline mechanism self-check: does not call LLM/network, directly drives the "search miss → create tool → pre-save validation → register → reuse" loop |
| `--fresh` | Clears `tool_library/` before running, reproducing "evolution from scratch" |
| `--no-create` | Disables the ability to create tools (removes `create_tool`), used for comparison demos showing "without evolution ability, can only reuse/cannot complete" |
| `--model model name` | Override the LLM model name (higher priority than the `LLM_MODEL` environment variable) |
| `--output path` | Write tasks, answers, action traces, and reuse conclusions to this JSON file |

> The tool library is **persisted** to `tool_library/`. If `get_stock_price` was already packaged in a previous run, running again directly will have task one hit `search_tools` and reuse it at step 0, so you won't see the full "evolution" process; add `--fresh` to reproduce evolution.

**Without an API key / no internet access**, use `python demo.py --offline` for a mechanism self-check—it uses a purely offline, deterministic tool (calculating the number of days between two dates) to run through the complete loop: task one `search_tools` miss → `create_tool` (with pre-save validation) → register → call; task two `search_tools` hit → **direct reuse**, no reinventing the wheel; and additionally demonstrates that the pre-save validation gate will reject a "broken tool that doesn't run" from entering the library. The self-check runs in a temporary directory and **will not pollute** your real `tool_library/`. A real offline run output:

```text
[Validation Gate] Attempting to register a broken tool that will crash (with test_args)...
  Result: success=False  ->  Tool registration pre-validation failed: run(**test_args) did not return successfully...
  ✅ Pre-save validation blocked the broken tool (not stored), consistent with 'Don't save bad programs'.
[step 1] search_tools -> 0 hits (tool library empty, no hit)
[step 2] create_tool(days_between) -> success=True validated=True (pre-save validation actually ran run() once)
[step 3] days_between(...) -> {'start': '2020-01-01', 'end': '2020-03-01', 'days': 60}
[step 1] search_tools -> 1 hit: ['days_between'] (reuse!)
[step 2] days_between(...) -> {'start': '2021-01-01', 'end': '2021-12-31', 'days': 364}
Task one trace: ['search_tools', 'create_tool', 'days_between']
Task two trace: ['search_tools', 'days_between']
Did task two reuse the tool created by task one (did not re-create_tool): Yes ✅
Did the pre-save validation gate block the broken tool: Yes ✅
```

`demo.py` will run two tasks consecutively:

1. **NVDA** (demonstrates evolution): Starting from zero base tools, search → read documentation → sandbox test → package `get_stock_price` tool → provide NVIDIA's real stock price and weekly change.
2. **AAPL** (demonstrates reuse): `search_tools` hits the just-created `get_stock_price`, **directly reuses** it, no re-searching/creating.

> You can also switch to other OpenAI-compatible providers: `LLM_PROVIDER=moonshot|ark` (with corresponding `MOONSHOT_API_KEY` / `ARK_API_KEY`), or use `LLM_MODEL` to override the model name. Search uses DuckDuckGo, no search key required.

## A Real Run Trace (Excerpt, Real Internet + Real OpenAI Calls)

**Task One · NVDA** (Self-Evolution, Note Error Recovery):

```text
[step 1] search_tools("stock price")      -> 0 hits (tool library empty)
[step 2] web_search("open source python library stock price") -> yfinance · PyPI ...
[step 3] read_webpage(pypi.org/project/yfinance) / github.com/ranaroussi/yfinance
[step 4] code_interpreter(...)  -> stdout empty, note: "No real data printed, not considered verification passed"
[step 5] code_interpreter(...)  -> "Latest stock price: 205.91..., Change: 1.54"   ← Real data, verification passed
[step 6] create_tool("get_stock_price", parameterized ticker/period, internally calls yfinance)
[step 7] get_stock_price(ticker="NVDA") -> {latest_price: 205.71, change_percentage: 1.44}
[Final Answer] NVIDIA (NVDA) latest stock price $205.71, +1.44% compared to one week ago. Data source: yfinance.
```

**Task Two · AAPL** (Tool Reuse, No Re-Searching/Creating):

```text
[step 1] search_tools("stock price")  -> hit get_stock_price (reuse!)
[step 2] get_stock_price(ticker="AAPL") -> {latest_price: 330.48, change_percentage: 4.51}
[Final Answer] Apple (AAPL) latest stock price $330.48, +4.51% compared to one week ago.
Task two trace = ['search_tools', 'get_stock_price']  → No web_search / create_tool ✅ Reuse confirmed
```

(Numbers change in real-time with market data, different each run; above is the result of one real run.)

## Conclusion

- Starting from **zero domain tools**, with only five meta-tools, the agent autonomously discovered `yfinance`, packaged a generic `get_stock_price` tool, and provided **real** stock prices and changes.
- The second task **hit and reused** the already-built tool via `search_tools`, without re-searching/reinventing—the tool library makes the agent "stronger with use".
- Empty output reminder + anti-mock guard + "verify before package" effectively **suppressed hallucinations**: in one successful run, the model's first test code forgot to `print`, was reminded by the note, self-corrected, and ultimately answered based on real execution results.

## Regarding Task One in the Book (YouTube Subtitles)

The book's "Task One: YouTube subtitle understanding, answer 100000000" depends on `youtube-transcript-api` + a specific video. Internet connectivity, access controls, or video takedowns can cause instability, so this repository **uses a reliably reproducible real-time financial task to actually verify the mechanism**.
To reproduce the YouTube scenario, the same pipeline applies: let the agent `web_search` find `youtube-transcript-api` → read documentation → sandbox test → `create_tool` to package a subtitle fetching tool.

## ⚠️ Security Boundary Reminder (Must Read)

This experiment **executes model-generated code** and **installs third-party packages from the network**, which inherently carries risks:

- **Supply chain risk**: `code_interpreter` will `pip install` the package selected by the model. In real/production environments, package sources must have **whitelisting/auditing/pinned versions and hashes** to guard against typosquatting and malicious packages.
- **Code execution isolation**: The sandbox here is only **demonstration-grade** (subprocess isolation + timeout), **not a security sandbox**. Production environments should use containers / gVisor / seccomp / network-namespace isolation / read-only filesystems / resource limits for strong isolation, and ideally disable networking or only allow whitelisted domains.
- **Self-evolving tool library requires human review**: Tools persisted by `create_tool` will be reused by subsequent tasks, effectively turning "model-written code" into a permanent capability. It is recommended to perform manual/automated review of tools entering the library, and record sources and audit logs.
- This directory by default includes `tool_library/*.json` and `.sandbox_packages/` in `.gitignore` (runtime artifacts).
