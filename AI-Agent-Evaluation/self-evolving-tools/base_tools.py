"""The non-library subset of the five base tools:
web_search, read_webpage, and code_interpreter.

Design principles: minimum predefined capabilities, maximum self-evolution.
- No domain tools such as get_stock_price or get_youtube_transcript are included.
- The agent uses web_search to find libraries/APIs, read_webpage for documentation,
  and code_interpreter to verify solutions by executing code in a subprocess.
- Outputs come from real network or execution results to reduce hallucinations.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Target directory for pip install --target; installed packages persist here.
# Packaged tools can import them later using the same PYTHONPATH.
PROJECT_DIR = Path(__file__).resolve().parent
SANDBOX_PKG_DIR = PROJECT_DIR / ".sandbox_packages"


# --------------------------------------------------------------------------- #
# Tool 1: web_search -- DuckDuckGo, no API key required
# --------------------------------------------------------------------------- #
def web_search(query: str, num_results: int = 6) -> dict:
    """Search the web with DuckDuckGo for free, without an API key.

    Following the chapter4/perception-tools approach:
    - Prefer lite.duckduckgo.com for more stable responses and less throttling.
    - Fall back to html.duckduckgo.com.
    - Retry with backoff when DDG returns HTTP 202 (rate limiting), so transient
      network issues do not immediately fail the search.
    """
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "search query is empty", "results": []}

    try:  # Default null or nonnumeric model arguments to the standard value
        num_results = max(1, min(int(num_results or 6), 10))
    except (TypeError, ValueError):
        num_results = 6
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15"
        )
    }

    last_err = None
    # Retry each of the two endpoints several times
    for endpoint in ("https://lite.duckduckgo.com/lite/", "https://html.duckduckgo.com/html/"):
        for attempt in range(3):
            try:
                resp = requests.post(
                    endpoint, data={"q": query, "kl": "wt-wt"}, headers=headers, timeout=15
                )
                if resp.status_code == 202:  # DDG rate-limiting signal
                    raise RuntimeError("rate limited (202)")
                resp.raise_for_status()
                results = _parse_ddg(endpoint, resp.text, num_results)
                if results:
                    return {"success": True, "query": query, "count": len(results), "results": results}
                last_err = "no results parsed"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
            time.sleep(1.5 * (attempt + 1))  # Backoff

    return {"success": False, "error": f"search failed: {last_err}", "results": []}


def _parse_ddg(endpoint: str, html: str, num_results: int) -> list:
    """Parse the two DuckDuckGo page layouts."""
    soup = BeautifulSoup(html, "html.parser")
    results = []

    if "html.duckduckgo" in endpoint:
        for div in soup.find_all("div", class_="result")[:num_results]:
            a = div.find("a", class_="result__a")
            if not a:
                continue
            snip = div.find("a", class_="result__snippet")
            results.append(
                {
                    "title": a.get_text(strip=True),
                    "url": a.get("href", ""),
                    "snippet": snip.get_text(strip=True) if snip else "",
                }
            )
    else:  # Lite layout: results are ordinary <a href="http..."> links
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text(strip=True)
            if href.startswith("http") and text:
                results.append({"title": text, "url": href, "snippet": ""})
            if len(results) >= num_results:
                break
    return results


# --------------------------------------------------------------------------- #
# Tool 2: read_webpage -- fetch a page and extract its main text
# --------------------------------------------------------------------------- #
def read_webpage(url: str, max_chars: int = 6000) -> dict:
    """Fetch a page and extract plain text for reading README files or API docs."""
    if not url or not url.startswith(("http://", "https://")):
        return {"success": False, "error": "invalid url"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"fetch failed: {e}", "url": url}

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    truncated = len(text) > max_chars
    return {
        "success": True,
        "url": url,
        "title": soup.title.get_text(strip=True) if soup.title else "",
        "text": text[:max_chars],
        "truncated": truncated,
    }


# --------------------------------------------------------------------------- #
# Tool 3: code_interpreter -- execute Python in a subprocess sandbox
# --------------------------------------------------------------------------- #
def code_interpreter(code: str, pip_install: list | None = None, timeout: int = 60) -> dict:
    """Execute Python in a separate subprocess to verify libraries/APIs found online.

    - pip_install: third-party packages to install first into .sandbox_packages
      with --target. This leaves the system environment intact and makes packages
      importable by the subprocess through PYTHONPATH.
    - timeout: terminate execution after the limit to prevent hangs or loops.

    Security boundary: this demonstration provides process isolation and a timeout,
    not a security sandbox. Production requires stronger isolation such as
    containers, gVisor, or network-disabled namespaces, plus package auditing to
    address supply chain risks.
    """
    SANDBOX_PKG_DIR.mkdir(exist_ok=True)
    logs = []

    # Add the package directory to PYTHONPATH; system site-packages remain available
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SANDBOX_PKG_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    # 1) Install packages on demand with pip install --target
    if pip_install:
        for pkg in pip_install:
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet",
                     "--target", str(SANDBOX_PKG_DIR), pkg],
                    capture_output=True, text=True, timeout=180, env=env,
                )
                if r.returncode != 0:
                    logs.append(f"[pip install {pkg}] FAILED: {r.stderr.strip()[-500:]}")
                else:
                    logs.append(f"[pip install {pkg}] ok")
            except Exception as e:  # noqa: BLE001
                logs.append(f"[pip install {pkg}] error: {e}")

    # 2) Execute code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=SANDBOX_PKG_DIR) as f:
        f.write(code)
        script = f.name
    try:
        r = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        out = r.stdout[-8000:]
        result = {
            "success": r.returncode == 0,
            "stdout": out,
            "stderr": r.stderr[-4000:],
            "returncode": r.returncode,
            "pip_logs": logs,
        }
        # Remind the model that no printed output means no real data for an answer or tool.
        if r.returncode == 0 and not out.strip():
            result["note"] = (
                "Code execution succeeded but stdout is empty: no real data was printed. "
                "Verification has not passed. Revise the code to call the library and print real numbers."
            )
        return result
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"timeout after {timeout}s", "pip_logs": logs}
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def run_python_snippet(code: str, timeout: int = 60) -> dict:
    """Run a script in the same sandbox for tool_manager, without pip installation."""
    return code_interpreter(code, pip_install=None, timeout=timeout)
