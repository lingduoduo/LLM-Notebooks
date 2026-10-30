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
from dataclasses import dataclass
from pathlib import Path

from tree import Node

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"  # unpinned alias; the resolved version is in each response
RETRY_DELAYS = (1, 2, 4)  # seconds between attempts for 429/5xx/network errors
USER_AGENT = "jev-decisionrag-prototype/0.1"
USD_PER_INPUT_TOKEN = 0.042e-6  # estimate used when a response reports no usage.cost
CHOICE_KEY = "most_relevant"
BINARY_RULE = ("Present: the candidate is actually present in or relevant to the query. "
               "Absent/negated: it is absent, only mentioned negated, or unrelated.")

Transport = Callable[[dict], dict]


class JevError(RuntimeError):
    pass


@dataclass(frozen=True)
class Provider:
    name: str
    env_key: str
    endpoint: str
    model: str


# In order of preference: the first provider whose key is set is used.
PROVIDERS = (
    Provider("typesafe", "TYPESAFE_API_KEY", TYPESAFE_ENDPOINT, TYPESAFE_MODEL),
    Provider("openrouter", "OPENROUTER_API_KEY", ENDPOINT, MODEL),
)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    if not _is_number(value) or not 0.0 <= value <= 1.0:
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
            for opt, value in _probabilities(response, key).items():
                _prob(value, f"answer {key!r} probability of {opt!r}")


class HttpTransport:
    def __init__(self, api_key: str, timeout: float = 60,
                 endpoint: str = ENDPOINT):
        self.endpoint = endpoint
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json",
                         "User-Agent": USER_AGENT}

    def __repr__(self) -> str:  # never expose the key
        return f"HttpTransport(endpoint={self.endpoint!r})"

    def _send_once(self, data: bytes) -> tuple[int, bytes]:
        req = urllib.request.Request(self.endpoint, data=data, method="POST",
                                     headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def __call__(self, body: dict) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        for attempt in range(len(RETRY_DELAYS) + 1):
            last = attempt == len(RETRY_DELAYS)
            try:
                status, raw = self._send_once(data)
            except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, resets
                if last:
                    raise JevError(f"network error: {getattr(e, 'reason', None) or e!r}") from e
            else:
                if status == 200:
                    try:
                        return json.loads(raw.decode("utf-8"))
                    except ValueError as e:
                        raise JevError(f"invalid JSON in 200 response: {raw[:200]!r}") from e
                if last or not (status == 429 or status >= 500):
                    raise JevError(f"{status}: {_message(raw)}")
            time.sleep(RETRY_DELAYS[attempt])


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
        try:
            cached = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
        else:
            with self._lock:
                self.hits += 1
            return json.loads(cached)
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


def _cost_of(response: dict) -> tuple[float, bool]:
    """(USD, estimated?): the reported usage.cost, else an estimate from input tokens."""
    usage = response.get("usage") or {}
    if _is_number(usage.get("cost")):
        return float(usage["cost"]), False
    tokens = usage.get("input_tokens")
    return (tokens * USD_PER_INPUT_TOKEN if _is_number(tokens) else 0.0), True


class JevJudge:
    """Judge over the Jev decision model. `requests` counts requests sent, cache hits included."""

    def __init__(self, transport: Transport, model: str = MODEL,
                 max_per_request: int = 50, max_workers: int = 8,
                 provider: str = ""):
        self.provider = provider
        self.transport = transport
        self.model = model
        self.max_per_request = max_per_request
        self.max_workers = max_workers
        self.requests = 0
        self.cost_usd = 0.0
        self.cost_estimated = False
        self.models_seen: set[str] = set()
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
            response = self.transport({"model": self.model, **build(chunk)})
            resp = response if isinstance(response, dict) else {}
            cost, estimated = _cost_of(resp)
            with self._lock:
                self.requests += 1
                self.cost_usd += cost
                self.cost_estimated |= estimated
                if isinstance(resp.get("model"), str):
                    self.models_seen.add(resp["model"])
            return parse(response, chunk)

        if len(chunks) == 1:
            parts = [one(chunks[0])]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                parts = list(pool.map(one, chunks))
        merged: dict[str, float] = {}
        for p in parts:
            merged.update(p)
        return merged
