from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from judge import Judge
from tree import Node

SEPARATE_WORKERS = 8  # per-unit calls in flight in separate mode


@dataclass(frozen=True)
class NavigationConfig:
    keep_threshold: float = 0.3
    k: int = 4
    floor: float = 0.1
    tag_threshold: float = 0.5
    r: int | None = None
    m: int = 20
    use_choice: bool = True
    binary_question: str = "Is Candidate {n} relevant to the query?"
    choice_question: str = "Which candidate is most relevant to the query?"
    returnable: Callable[[Node], bool] = lambda n: n.is_leaf


@dataclass
class Ranked:
    id: str
    p_bin: float | None
    p_mul: float | None = None
    source: str = "inferred"


@dataclass
class Result:
    ranked: list[Ranked]
    trace: list[dict]
    latency_s: float


class _Stage:
    """Times one stage and counts its judge requests."""

    def __init__(self, judge):
        self.judge = judge
        self.calls = 0

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.r0 = getattr(self.judge, "requests", None)
        return self

    def call(self, method: str, *args):
        self.calls += 1
        return getattr(self.judge, method)(*args)

    def __exit__(self, *exc):
        self.latency_s = time.perf_counter() - self.t0
        r1 = getattr(self.judge, "requests", None)
        if isinstance(self.r0, int) and isinstance(r1, int):
            self.requests = r1 - self.r0
        else:
            self.requests = self.calls

    def record(self, entry: dict) -> dict:
        return {**entry, "latency_s": self.latency_s, "requests": self.requests}


def navigate(root: Node, query: str, judge: Judge, config: NavigationConfig) -> Result:
    t0 = time.perf_counter()
    candidates = list(root.children)
    nodes: dict[str, Node] = {}
    scores: dict[str, float] = {}
    trace: list[dict] = []
    while candidates:
        with _Stage(judge) as st:
            p = st.call("binary", query, candidates, config.binary_question)
        scores.update(p)
        nodes.update({c.id: c for c in candidates})
        internal = sorted((c for c in candidates if not c.is_leaf), key=lambda c: -p[c.id])
        keep = [c for c in internal if p[c.id] >= config.keep_threshold][: config.k]
        if not keep and internal and p[internal[0].id] >= config.floor:
            keep = [internal[0]]
        kept_ids = {c.id for c in keep}
        trace.append(st.record({
            "judged": [c.id for c in candidates],
            "p": dict(p),
            "kept": [c.id for c in keep],
            "pruned": [c.id for c in internal if c.id not in kept_ids],
        }))
        candidates = [child for c in keep for child in c.children]
    units = [n for n in nodes.values() if config.returnable(n)]
    ranked = _tag_and_choose(units, scores, query, judge, config, trace)
    return Result(ranked, trace, time.perf_counter() - t0)


def score_units(units: Sequence[Node], query: str, judge: Judge, config: NavigationConfig,
                *, separate: bool = False) -> Result:
    t0 = time.perf_counter()
    trace: list[dict] = []
    scores: dict[str, float] = {}
    with _Stage(judge) as st:
        if separate:
            with ThreadPoolExecutor(max_workers=SEPARATE_WORKERS) as pool:
                parts = list(pool.map(
                    lambda u: judge.binary(query, [u], config.binary_question), units))
            st.calls += len(parts)
            for part in parts:  # map keeps unit order, so scores match the sequential run
                scores.update(part)
        else:
            scores.update(st.call("binary", query, list(units), config.binary_question))
    trace.append(st.record({"judged": [u.id for u in units], "p": dict(scores),
                            "kept": [], "pruned": []}))
    ranked = _tag_and_choose(list(units), scores, query, judge, config, trace)
    return Result(ranked, trace, time.perf_counter() - t0)


def _tag_and_choose(units: list[Node], scores: dict[str, float], query: str, judge: Judge,
                    config: NavigationConfig, trace: list[dict]) -> list[Ranked]:
    th = config.tag_threshold
    tags = [
        u for u in units
        if scores[u.id] >= th and not any(c.id in scores and scores[c.id] >= th for c in u.children)
    ]
    tags.sort(key=lambda u: -scores[u.id])  # stable sort: ties keep judged order
    if config.r is not None:
        tags = tags[: config.r]
    ranked = [Ranked(u.id, scores[u.id]) for u in tags]
    if config.use_choice and len(tags) >= 2:
        short = tags[: config.m]
        with _Stage(judge) as st:
            q = st.call("choice", query, short, config.choice_question)
        p_mul = {u.id: q.get(u.id, 0.0) for u in short}
        head = sorted(ranked[: len(short)], key=lambda x: -p_mul[x.id])
        for x in head:
            x.p_mul = p_mul[x.id]
        ranked = head + ranked[len(short):]
        trace.append(st.record({"choice": [u.id for u in short], "p": p_mul}))
    return ranked
