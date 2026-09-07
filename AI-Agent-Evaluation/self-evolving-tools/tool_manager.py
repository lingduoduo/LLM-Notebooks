"""Tool library management: create_tool packages and persists tools, search_tools
retrieves tools for reuse, and the library executes packaged tools.

This is the core of Alita-style self-evolution:
- After verifying a solution with code_interpreter, the agent calls create_tool
  to store a standard tool in tool_library/, including its name, description,
  JSON Schema parameters, and Python code.
- For similar tasks, the agent should first find and reuse an existing tool with
  search_tools instead of searching the web and writing code again.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LIBRARY_DIR = PROJECT_DIR / "tool_library"
SANDBOX_PKG_DIR = PROJECT_DIR / ".sandbox_packages"


def normalize_schema(params) -> dict:
    """Normalize model-supplied parameters into a valid function-calling JSON Schema.

    Models often provide only a properties mapping, omitting the top-level
    {"type": "object"}. Handle this to prevent an invalid-schema HTTP 400 from
    interrupting the workflow when the tool is exposed to OpenAI.
    """
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    if params.get("type") == "object":
        out = dict(params)
        out["properties"] = params.get("properties") or {}
        return out
    if "properties" in params:  # properties exists, but type is missing or incorrect
        out = dict(params)
        out["type"] = "object"
        out["properties"] = params.get("properties") or {}
        return out
    # Treat the entire dictionary as a properties mapping
    return {"type": "object", "properties": params}


class ToolLibrary:
    """Minimal filesystem tool library: each JSON file stores metadata and code."""

    def __init__(self, library_dir: Path = LIBRARY_DIR):
        self.dir = Path(library_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- create_tool ----------------------------- #
    def create_tool(self, name: str, description: str, parameters: dict, code: str,
                    test_args: dict | None = None) -> dict:
        """Package a function as a standard tool and persist it.

        The code must define run(**kwargs) and return a JSON-serializable result.
        Parameters use a function-calling JSON Schema (type=object, properties,
        required).

        Pre-save validation corresponds to the Test step in Figure 8-7 and the
        chapter warning about tool quality degradation:
        - Compile the code and reject syntax errors.
        - If test_args are supplied, execute run(**test_args) in the sandbox.
          Register only after a successful result, preventing broken tools from
          entering the library and propagating errors through later reuse.
        """
        name = name.strip()
        if not name.isidentifier():
            return {"success": False, "error": f"invalid tool name: {name!r} (must be a valid identifier)"}
        if "def run" not in code:
            return {"success": False, "error": "tool code must define a function `def run(**kwargs)`"}
        # Pre-save validation 1: compile syntax and reject invalid code
        try:
            compile(code, f"<tool {name}>", "exec")
        except SyntaxError as e:
            return {"success": False, "error": f"tool code has a syntax error: {e}"}

        record = {
            "name": name,
            "description": description,
            "parameters": normalize_schema(parameters),
            "code": code,
        }

        # Pre-save validation 2: execute run() with test_args and reject failures
        validated = False
        if test_args is not None:
            val = self._run_record(record, test_args)
            if not val.get("success"):
                return {
                    "success": False,
                    "error": "Tool registration pre-validation failed: run(**test_args) did not return successfully. Fix the code or test_args "
                             "and resubmit. Unvalidated tools are not stored, so later tasks cannot reuse broken tools.",
                    "validation": val,
                }
            validated = True

        (self.dir / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "success": True,
            "message": f"tool '{name}' created and saved to tool_library/"
                       + (" (pre-save validation passed)" if validated else " (no test_args supplied; runtime validation skipped)"),
            "name": name,
            "validated": validated,
        }

    # ----------------------------- search_tools ---------------------------- #
    def search_tools(self, query: str) -> dict:
        """Search tool names and descriptions by keyword and return matches for reuse."""
        query = (query or "").strip().lower()
        terms = [t for t in query.replace(",", " ").split() if t]
        hits = []
        for rec in self.list_tools():
            name = str(rec.get("name") or "")
            desc = str(rec.get("description") or "")
            haystack = (name + " " + desc).lower()
            score = sum(1 for t in terms if t in haystack)
            if score > 0 or not terms:
                hits.append((score, rec))
        hits.sort(key=lambda x: -x[0])
        return {
            "success": True,
            "query": query,
            "count": len(hits),
            "tools": [
                {
                    "name": str(r.get("name") or ""),
                    "description": str(r.get("description") or ""),
                    "parameters": r.get("parameters") or {},
                }
                for _, r in hits
            ],
        }

    # ------------------------------ helpers -------------------------------- #
    def list_tools(self) -> list:
        recs = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    recs.append(data)
            except Exception:  # noqa: BLE001
                continue
        return recs

    def get_tool(self, name: str) -> dict | None:
        p = self.dir / f"{name}.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None

    # -------------------------- execute a wrapped tool --------------------- #
    def execute_tool(self, name: str, arguments: dict, timeout: int = 60) -> dict:
        """Execute a packaged tool in a subprocess sandbox and capture its JSON result.

        Inject the code and run(**args). PYTHONPATH points to .sandbox_packages
        so dependencies installed during creation remain available.
        """
        rec = self.get_tool(name)
        if rec is None:
            return {"success": False, "error": f"tool '{name}' not found in library"}
        return self._run_record(rec, arguments, timeout)

    def _run_record(self, rec: dict, arguments: dict, timeout: int = 60) -> dict:
        """Execute run(**arguments) from a tool record in a sandbox subprocess.

        Accept the record, including code, directly instead of reading from disk,
        enabling pre-save validation before the tool is persisted.
        """
        SANDBOX_PKG_DIR.mkdir(exist_ok=True)
        driver = (
            rec["code"]
            + "\n\nif __name__ == '__main__':\n"
            "    import json as _json, sys as _sys\n"
            "    _args = _json.loads(_sys.argv[1])\n"
            "    _out = run(**_args)\n"
            "    print('__TOOL_RESULT__' + _json.dumps(_out, default=str))\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SANDBOX_PKG_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=SANDBOX_PKG_DIR) as f:
            f.write(driver)
            script = f.name
        try:
            r = subprocess.run(
                [sys.executable, script, json.dumps(arguments)],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            if r.returncode != 0:
                return {"success": False, "error": "tool crashed", "stderr": r.stderr[-3000:]}
            for line in r.stdout.splitlines():
                if line.startswith("__TOOL_RESULT__"):
                    raw = line[len("__TOOL_RESULT__"):]
                    try:
                        return {"success": True, "result": json.loads(raw)}
                    except json.JSONDecodeError as e:
                        return {
                            "success": False,
                            "error": f"invalid result marker: {e}",
                            "stdout": r.stdout[-2000:],
                        }
            return {"success": False, "error": "no result marker", "stdout": r.stdout[-2000:]}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"timeout after {timeout}s"}
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass
