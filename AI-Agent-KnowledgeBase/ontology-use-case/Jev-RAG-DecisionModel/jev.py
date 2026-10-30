from __future__ import annotations

import hashlib
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tree import Node

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
CHOICE_KEY = "most_relevant"
BINARY_RULE = ("Present: the candidate is actually present in or relevant to the query. "
               "Absent/negated: it is absent, only mentioned negated, or unrelated.")

Transport = Callable[[dict], dict]


class JevError(RuntimeError):
    pass


def _state(query: str, candidates: Sequence[Node]) -> dict:
    lines = "\n".join(f"{i}. {n.description}" for i, n in enumerate(candidates, 1))
    return {"query": query, "candidates": lines}


def build_binary_request(query: str, candidates: Sequence[Node], question: str) -> dict:
    # Criteria are repeated per question, so keep them terse (a flat request has one question
    # per unit); the full rule is stated once in the shared state.
    criteria = {"true": "Present", "false": "Absent/negated"}
    questions = {
        f"c{i}": {"type": "noul", "instructions": question.format(n=i), "criteria": dict(criteria)}
        for i in range(1, len(candidates) + 1)
    }
    state = {**_state(query, candidates), "rule": BINARY_RULE}
    return {"state": state, "questions": questions}


def build_choice_request(query: str, candidates: Sequence[Node], question: str) -> dict:
    return {
        "state": _state(query, candidates),
        "questions": {CHOICE_KEY: {"type": "choice", "instructions": question,
                                   "criteria": {n.id: n.description for n in candidates}}},
    }


def _check(response: dict) -> dict:
    if isinstance(response, dict) and response.get("error"):
        err = response["error"]
        if isinstance(err, dict):
            raise JevError(f"{err.get('code')}: {err.get('message')}")
        raise JevError(str(err))
    return response


def _answer(response: dict, key: str, qtype: str) -> dict:
    _check(response)
    answers = (response.get("answers") if isinstance(response, dict) else None) or {}
    if not isinstance(answers, dict) or key not in answers:
        raise JevError(f"response is missing answer {key!r}")
    ans = answers[key]
    if not isinstance(ans, dict):
        raise JevError(f"answer {key!r} is not an object: {ans!r}")
    if ans.get("type") != qtype:
        raise JevError(f"answer {key!r} has type {ans.get('type')!r}, expected {qtype!r}")
    return ans


def _prob(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise JevError(f"{what} is not a probability: {value!r}")
    return float(value)


def _noul(response: dict, key: str) -> float:
    return _prob(_answer(response, key, "noul").get("noul"), f"answer {key!r} noul")


def _probabilities(response: dict, key: str) -> dict:
    probs = _answer(response, key, "choice").get("probabilities")
    if not isinstance(probs, dict):
        raise JevError(f"answer {key!r} has no probabilities map: {probs!r}")
    return probs


def parse_binary(response: dict, candidates: Sequence[Node]) -> dict[str, float]:
    return {n.id: _noul(response, f"c{i}") for i, n in enumerate(candidates, 1)}


def parse_choice(response: dict, candidates: Sequence[Node]) -> dict[str, float]:
    probs = _probabilities(response, CHOICE_KEY)
    return {n.id: _prob(probs.get(n.id, 0.0), f"probability of {n.id!r}") for n in candidates}


def validate_response(body: dict, response: dict) -> None:
    """Raise JevError unless `response` answers every question in `body` well-formed."""
    _check(response)
    for key, q in (body.get("questions") or {}).items():
        if q.get("type") == "noul":
            _noul(response, key)
        elif q.get("type") == "choice":
            _probabilities(response, key)


class HttpTransport:
    def __init__(self, api_key: str, timeout: float = 60):
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, body: dict) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        delays = [1, 2, 4]
        attempt = 0
        while True:
            req = urllib.request.Request(
                ENDPOINT, data=data, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status, raw = resp.status, resp.read()
            except urllib.error.HTTPError as e:
                status, raw = e.code, e.read()
            except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, resets
                if attempt >= len(delays):
                    raise JevError(f"network error: {getattr(e, 'reason', None) or e!r}") from e
                time.sleep(delays[attempt])
                attempt += 1
                continue
            if status == 200:
                try:
                    return json.loads(raw.decode("utf-8"))
                except ValueError as e:
                    raise JevError(f"invalid JSON in 200 response: {raw[:200]!r}") from e
            if (status == 429 or status >= 500) and attempt < len(delays):
                time.sleep(delays[attempt])
                attempt += 1
                continue
            raise JevError(f"{status}: {_message(raw)}")


def _message(raw: bytes) -> str:
    try:
        err = json.loads(raw.decode("utf-8")).get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    except (ValueError, AttributeError):
        pass
    return raw.decode("utf-8", "replace")[:200]


class CachedTransport:
    """Caches responses on disk; only responses that pass `validate` are stored."""

    def __init__(self, inner: Transport, cache_dir: Path,
                 validate: Callable[[dict, dict], None] = validate_response):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.validate = validate
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def __call__(self, body: dict) -> dict:
        key = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with self._lock:
                self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self.misses += 1
        response = self.inner(body)
        try:
            self.validate(body, response)
        except JevError:
            return response  # not cached; the caller's parse raises
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return response


class JevJudge:
    """Judge over the Jev decision model. `requests` counts requests sent, cache hits included."""

    def __init__(self, transport: Transport, model: str = MODEL,
                 max_per_request: int = 50, max_workers: int = 8):
        self.transport = transport
        self.model = model
        self.max_per_request = max_per_request
        self.max_workers = max_workers
        self.requests = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def binary(self, query: str, candidates: Sequence[Node], question: str) -> dict[str, float]:
        return self._run(candidates, lambda chunk: build_binary_request(query, chunk, question),
                         parse_binary)

    def choice(self, query: str, candidates: Sequence[Node], question: str) -> dict[str, float]:
        # One request, never split: probabilities are only comparable within one choice.
        candidates = list(candidates)
        return self._run(candidates, lambda chunk: build_choice_request(query, chunk, question),
                         parse_choice, size=max(len(candidates), 1))

    def _run(self, candidates, build, parse, size: int | None = None) -> dict[str, float]:
        candidates = list(candidates)
        if not candidates:
            return {}
        size = size or self.max_per_request
        chunks = [candidates[i:i + size] for i in range(0, len(candidates), size)]

        def one(chunk):
            body = {"model": self.model, **build(chunk)}
            response = self.transport(body)
            with self._lock:
                self.requests += 1
                usage = response.get("usage") or {} if isinstance(response, dict) else {}
                self.cost_usd += float(usage.get("cost") or 0.0)
            return parse(response, chunk)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            parts = list(pool.map(one, chunks))
        merged: dict[str, float] = {}
        for p in parts:
            merged.update(p)
        return merged
