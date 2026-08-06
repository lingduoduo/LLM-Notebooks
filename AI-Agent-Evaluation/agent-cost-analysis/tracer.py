"""A lightweight tracing and observability layer.

One agent task is a trace, and each LLM or tool call is a span. Spans record
their logical step, type, token usage, latency, and cost.

Usage:
    tracer = Tracer(client)
    resp = tracer.chat(step="turn-1", tool="query_order",
                       model=..., messages=..., temperature=0)
    tracer.print_breakdown()

Offline reuse without model calls:
    tracer = Tracer.from_records(records, pricing=..., name=...)
    # records contains previously captured per-step token usage.
"""

import math
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional

from config import Pricing, default_pricing


def _percentile(values: List[float], q: float) -> float:
    """Return a nearest-rank percentile without requiring NumPy."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    # Nearest rank is ceil(q / 100 * N); ordinary rounding is incorrect at ties.
    rank = max(1, min(len(xs), math.ceil(q / 100.0 * len(xs))))
    return xs[rank - 1]


@dataclass
class Span:
    """One traced call, normally an LLM call."""
    step: str                 # Logical step such as "turn-2".
    tool: str                 # Associated tool or action.
    kind: str = "llm"         # Span type: llm or tool.
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    # Cumulative tool-result tokens in this turn's input; -1 means unknown.
    tool_ctx_tokens: int = -1
    latency_s: float = 0.0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_tokens, 0)


class Tracer:
    """Wrap an OpenAI client and record usage, latency, and cost."""

    def __init__(self, client=None, name: str = "trace",
                 pricing: Optional[Pricing] = None):
        self.client = client
        self.name = name
        self.pricing = pricing or default_pricing()
        self.spans: List[Span] = []

    # ---------- Collection ----------
    def chat(self, step: str, tool: str, tool_ctx_tokens: int = -1, **kwargs):
        """Make a traced chat.completions call and return the original response."""
        t0 = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.time() - t0

        usage = resp.usage
        # Some OpenAI-compatible providers omit usage (null); match from_records coercion.
        if usage is None:
            span = Span(
                step=step,
                tool=tool,
                kind="llm",
                tool_ctx_tokens=tool_ctx_tokens,
                latency_s=latency,
                cost_usd=0.0,
            )
            self.spans.append(span)
            return resp

        # Read cached_tokens defensively because compatible providers may omit it.
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        span = Span(
            step=step,
            tool=tool,
            kind="llm",
            prompt_tokens=prompt_tokens,
            cached_tokens=cached,
            completion_tokens=completion_tokens,
            tool_ctx_tokens=tool_ctx_tokens,
            latency_s=latency,
            cost_usd=self.pricing.cost_usd(
                prompt_tokens, cached, completion_tokens),
        )
        self.spans.append(span)
        return resp

    # ---------- Offline reuse ----------
    @classmethod
    def from_records(cls, records: List[dict], name: str = "trace",
                     pricing: Optional[Pricing] = None) -> "Tracer":
        """Rebuild a trace from recorded usage and recalculate its cost."""
        tr = cls(client=None, name=name, pricing=pricing)
        for r in records:
            span = Span(
                step=r.get("step", ""),
                tool=r.get("tool", ""),
                kind=r.get("kind", "llm"),
                # Missing/null tool_ctx → -1 (unknown); keep explicit 0 (known-zero).
                prompt_tokens=int(r.get("prompt_tokens") or 0),
                cached_tokens=int(r.get("cached_tokens") or 0),
                completion_tokens=int(r.get("completion_tokens") or 0),
                tool_ctx_tokens=(-1 if r.get("tool_ctx_tokens") is None
                                 else int(r.get("tool_ctx_tokens"))),
                latency_s=float(r.get("latency_s") or 0.0),
            )
            span.cost_usd = tr.pricing.cost_usd(
                span.prompt_tokens, span.cached_tokens, span.completion_tokens)
            tr.spans.append(span)
        return tr

    def to_records(self) -> List[dict]:
        """Export raw per-step token usage for offline reuse."""
        out = []
        for s in self.spans:
            d = asdict(s)
            d.pop("cost_usd", None)   # Costs are recalculated rather than persisted.
            out.append(d)
        return out

    # ---------- Aggregation ----------
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.spans)

    def total_cached_tokens(self) -> int:
        return sum(s.cached_tokens for s in self.spans)

    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.spans)

    def total_uncached_prompt_tokens(self) -> int:
        return sum(s.uncached_prompt_tokens for s in self.spans)

    def total_tool_ctx_tokens(self) -> int:
        return sum(s.tool_ctx_tokens for s in self.spans if s.tool_ctx_tokens >= 0)

    def total_latency(self) -> float:
        return sum(s.latency_s for s in self.spans)

    def cache_rate(self) -> float:
        pin = self.total_prompt_tokens()
        return self.total_cached_tokens() / pin if pin else 0.0

    def component_costs(self) -> dict:
        """Split total cost into uncached input, cached input, and output."""
        p = self.pricing
        uncached_in = self.total_uncached_prompt_tokens()
        cached_in = self.total_cached_tokens()
        out = self.total_completion_tokens()
        return {
            "uncached_input_cost": uncached_in / 1_000_000 * p.input_per_m,
            "cached_input_cost": cached_in / 1_000_000 * p.cached_per_m,
            "output_cost": out / 1_000_000 * p.output_per_m,
            "uncached_input_tokens": uncached_in,
            "cached_input_tokens": cached_in,
            "output_tokens": out,
            "tool_ctx_tokens": self.total_tool_ctx_tokens(),
        }

    def cost_distribution(self) -> dict:
        """Return the per-step cost distribution (p50/p95/p99)."""
        costs = [s.cost_usd for s in self.spans]
        n = len(costs)
        return {
            "n": n,
            "mean": (sum(costs) / n) if n else 0.0,
            "p50": _percentile(costs, 50),
            "p95": _percentile(costs, 95),
            "p99": _percentile(costs, 99),
            "max": max(costs) if costs else 0.0,
        }

    # ---------- Reporting ----------
    def print_breakdown(self, title: Optional[str] = None):
        """Print a per-step cost breakdown, components, and distribution."""
        print()
        print(f"===== Cost breakdown: {title or self.name} =====")
        header = (
            f"{'Step':<8} {'Tool/action':<20} {'Input':>8} {'Cached':>8} "
            f"{'Tool tok':>8} {'Output':>8} {'Latency':>8} {'Cost ($)':>12}"
        )
        print(header)
        print("-" * len(header))
        for s in self.spans:
            tctx = s.tool_ctx_tokens if s.tool_ctx_tokens >= 0 else "-"
            print(
                f"{s.step:<8} {s.tool:<20} {s.prompt_tokens:>8} {s.cached_tokens:>8} "
                f"{str(tctx):>8} {s.completion_tokens:>8} {s.latency_s:>8.2f} "
                f"{s.cost_usd:>12.6f}"
            )
        print("-" * len(header))
        tctx_total = self.total_tool_ctx_tokens() if any(
            s.tool_ctx_tokens >= 0 for s in self.spans) else "-"
        print(
            f"{'Total':<8} {'':<20} {self.total_prompt_tokens():>8} "
            f"{self.total_cached_tokens():>8} {str(tctx_total):>8} "
            f"{self.total_completion_tokens():>8} "
            f"{self.total_latency():>8.2f} {self.total_cost():>12.6f}"
        )

        # Identify the most expensive step.
        if self.spans:
            worst = max(self.spans, key=lambda s: s.cost_usd)
            total = self.total_cost()
            share = worst.cost_usd / total * 100 if total else 0
            print(
                f"\nMost expensive step -> {worst.step} / {worst.tool}: "
                f"${worst.cost_usd:.6f} ({share:.1f}% of total)"
            )

        # Cost components.
        comp = self.component_costs()
        total = self.total_cost() or 1e-12
        print("Cost components:")
        print(f"  Uncached input {comp['uncached_input_tokens']:>8} tok  "
              f"${comp['uncached_input_cost']:.6f}  ({comp['uncached_input_cost']/total*100:.1f}%)")
        print(f"  Cached input   {comp['cached_input_tokens']:>8} tok  "
              f"${comp['cached_input_cost']:.6f}  ({comp['cached_input_cost']/total*100:.1f}%)")
        print(f"  Output         {comp['output_tokens']:>8} tok  "
              f"${comp['output_cost']:.6f}  ({comp['output_cost']/total*100:.1f}%)")
        if comp["tool_ctx_tokens"] > 0:
            print(f"  Injected tool results total {comp['tool_ctx_tokens']} input tokens")

        # Per-step cost distribution.
        dist = self.cost_distribution()
        print(f"Per-step cost distribution (n={dist['n']}): mean ${dist['mean']:.6f}  "
              f"p50 ${dist['p50']:.6f}  p95 ${dist['p95']:.6f}  p99 ${dist['p99']:.6f}")
