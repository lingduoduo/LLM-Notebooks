"""Explicit offline and live providers; usage belongs to each returned response."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse


def tokens(text):
    # Words for Latin text; individual characters and bigrams for Chinese.
    words = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text.lower())
    return words + [
        a + b
        for a, b in zip(words, words[1:])
        if "\u3400" <= a <= "\u9fff" and "\u3400" <= b <= "\u9fff"
    ]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a) * sum(y * y for y in b)) or 1
    )


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    source: str = "offline"

    @property
    def usage(self):
        return {k: v for k, v in asdict(self).items() if k.endswith("_tokens")}


class LexicalEncoder:
    """Deterministic hashing baseline; not a learned semantic embedding."""

    def __init__(self, dimension=256):
        if type(dimension) is not int or dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.fingerprint = f"lexical-sha256-v1:{dimension}"

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in tokens(text):
                index = (
                    int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
                    % self.dimension
                )
                vector[index] += 1
            norm = math.sqrt(sum(x * x for x in vector)) or 1
            result.append([x / norm for x in vector])
        return result


class OfflineCompletion:
    """Extractive smoke-test answerer; never accesses reference answers."""

    source = "offline-extractive"

    def complete(self, prompt, *, json_mode=False):
        if json_mode:
            raise ValueError("Offline mode uses lexical F1, not a simulated LLM judge")
        context = prompt.split("<evidence>\n", 1)[1].split("\n</evidence>", 1)[0]
        question = prompt.rsplit("<question>\n", 1)[1].split("\n</question>", 1)[0]
        query = set(tokens(question))
        ranked = sorted(
            enumerate(context.splitlines()),
            key=lambda pair: (-len(query & set(tokens(pair[1]))), pair[0]),
        )
        lines = [line for _, line in ranked[:3] if query & set(tokens(line))]
        return Completion(
            "\n".join(lines) if lines else "Information not found", source=self.source
        )


def _endpoint(base_url, path):
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("base URL must be an HTTP(S) endpoint without credentials")
    if parsed.scheme == "http" and parsed.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise ValueError("Remote API endpoints must use HTTPS")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain query parameters or fragments")
    return base_url.rstrip("/") + "/" + path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def post_json(url, key, payload, timeout=60):
    request = urllib.request.Request(
        url,
        json.dumps(payload, allow_nan=False).encode(),
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(
            request, timeout=timeout
        ) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        # Never persist provider response bodies or request headers containing credentials.
        status = getattr(exc, "code", None)
        raise RuntimeError(
            f"API request failed ({status or type(exc).__name__})"
        ) from None


class ChatCompletion:
    source = "live"

    def __init__(self, model, base_url, api_key, timeout=60):
        if not api_key:
            raise ValueError("An answer/judge API key is required in live mode")
        self.model, self.base_url, self.api_key, self.timeout = (
            model,
            base_url,
            api_key,
            timeout,
        )
        self.url = _endpoint(base_url, "chat/completions")

    def complete(self, prompt, *, json_mode=False):
        raw = post_json(
            self.url,
            self.api_key,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object" if json_mode else "text"},
            },
            self.timeout,
        )
        try:
            text = raw["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError()
            usage = raw.get("usage") or {}
            values = [
                usage.get(k)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ]
            if any(v is not None and (type(v) is not int or v < 0) for v in values):
                raise ValueError()
            return Completion(text, *values, source=self.source)
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError("Invalid chat completion response") from None


class APIEncoder:
    def __init__(self, model, base_url, api_key, batch_size=16):
        if not api_key:
            raise ValueError("An embedding API key is required")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Invalid embedding batch size")
        self.model, self.base_url, self.api_key, self.batch_size = (
            model,
            base_url,
            api_key,
            batch_size,
        )
        self.url = _endpoint(base_url, "embeddings")
        self.fingerprint = f"api:{base_url.rstrip('/')}:{model}"
        self.dimension = None
        self.usage = []

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            raw = post_json(
                self.url, self.api_key, {"model": self.model, "input": batch}
            )
            rows = raw.get("data", [])
            if len(rows) != len(batch) or sorted(
                r.get("index", -1) for r in rows
            ) != list(range(len(batch))):
                raise ValueError("Embedding response indices do not match input")
            for row in sorted(rows, key=lambda r: r["index"]):
                vector = row.get("embedding")
                if (
                    not isinstance(vector, list)
                    or not vector
                    or any(
                        type(x) not in (int, float) or not math.isfinite(x)
                        for x in vector
                    )
                ):
                    raise ValueError("Invalid embedding vector")
                if self.dimension is None:
                    self.dimension = len(vector)
                if len(vector) != self.dimension:
                    raise ValueError("Embedding dimension changed")
                result.append(vector)
            self.usage.append(raw.get("usage") or {})
        return result


def parse_judge_score(raw):
    text = raw.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, re.S)
        if not match:
            raise ValueError("Invalid judge JSON")
        text = match.group(1)
    try:
        score = json.loads(text)["score"]
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError(
            "Judge must return a finite numeric score between 0 and 1"
        ) from None
    return float(score)


def lexical_f1(gold, prediction):
    expected, actual = Counter(tokens(gold)), Counter(tokens(prediction))
    overlap = sum((expected & actual).values())
    return (
        2 * overlap / (sum(expected.values()) + sum(actual.values()))
        if expected or actual
        else 1.0
    )


class LocalDecisions:
    source = "offline-heuristic"
    model = "local-v1"

    def __init__(self):
        self.calls = []

    def decide(self, operation, state, questions):
        values = {k: float(state.get("heuristics", {}).get(k, 0.5)) for k in questions}
        self.calls.append(
            {
                "operation": operation,
                "source": self.source,
                "usage": {},
                "decisions": values,
            }
        )
        return values


class JevDecisions:
    source = "jev"

    def __init__(
        self, api_key=None, model="jev-latest", base_url="https://api.typesafe.ai"
    ):
        from typesafe_sdk import TypeSafeClient, RetryPolicy

        key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("TYPESAFE_API_KEY is required for real Jev decisions")
        _endpoint(base_url, "")
        self.model = model
        self.base_url = base_url
        self.client = TypeSafeClient(
            api_key=key,
            model=model,
            base_url=base_url,
            retry=RetryPolicy(max_retries=1, timeout=30),
        )
        self.calls = []

    def decide(self, operation, state, questions):
        from typesafe_sdk import Noul

        # Local heuristic hints are private to the offline provider.
        state = {k: v for k, v in state.items() if k != "heuristics"}
        try:
            response = self.client.system_one(
                state=json.dumps({"operation": operation, **state}, ensure_ascii=False),
                questions={k: Noul(instructions=v) for k, v in questions.items()},
                timeout=30,
            )
            raw = response.model_dump()
            values = {k: raw["answers"][k]["noul"] for k in questions}
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in values.values()
            ):
                raise ValueError()
        except Exception:
            raise RuntimeError(
                "Jev decision request failed; no silent fallback was applied"
            ) from None
        self.calls.append(
            {
                "operation": operation,
                "source": self.source,
                "usage": raw.get("usage") or {},
                "decisions": values,
            }
        )
        return values

    def close(self):
        self.client.close()
