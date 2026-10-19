"""Render current run artifacts; captions and scales come from the run itself."""

from html import escape
import json
from pathlib import Path


def render_report(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metric = summary["score_metric"]
    mode = summary["mode"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="630" viewBox="0 0 1100 630">',
        '<rect width="1100" height="630" fill="#f3f7f6"/>',
    ]

    def text(x, y, value, size=16, color="#172b36"):
        parts.append(
            f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}">{escape(str(value))}</text>'
        )

    text(35, 45, f"{summary['dataset']} — four-arm memory experiment", 25)
    text(35, 77, f"Mode: {mode} | Metric: {metric} | Implementation: standalone", 15)
    text(
        35,
        105,
        "Offline results measure execution and lexical overlap; they are not LLM quality measurements."
        if mode == "offline"
        else "Scores exclude failed calls; inspect coverage and errors before comparing arms.",
        14,
    )
    headers = [
        "Arm",
        "Score",
        "Valid / total",
        "Prompt tokens / answer",
        "Answer latency (s)",
    ]
    xs = [40, 300, 440, 620, 875]
    for x, label in zip(xs, headers):
        text(x, 158, label, 15)
    labels = {
        "A": "A Sliding window",
        "B": "B Full context",
        "C": "C Naive RAG",
        "D": "D Memory graph",
    }
    for i, (arm, values) in enumerate(summary["arms"].items()):
        y = 215 + i * 80
        parts.append(
            f'<rect x="30" y="{y - 32}" width="1040" height="64" rx="8" fill="white"/>'
        )
        score = values["mean_score"]
        prompt = values["mean_prompt_tokens"]
        vals = [
            labels[arm],
            f"{score:.3f}" if score is not None else "unavailable",
            f"{values['scored']}/{values['questions']}",
            f"{prompt:.0f}" if prompt is not None else "unavailable",
            f"{values['mean_answer_latency_seconds']:.3f}",
        ]
        for x, v in zip(xs, vals):
            text(x, y, v)
    text(
        40,
        555,
        f"Memory build: {summary['build_seconds']:.3f}s | {summary['nodes']} nodes | {summary['edges']} edges",
        16,
    )
    text(
        40,
        585,
        "Prompt tokens are provider-reported input usage. Missing usage stays unavailable.",
        14,
    )
    parts.append("</svg>")
    (directory / "report.svg").write_text("\n".join(parts), encoding="utf-8")
    return directory / "report.svg"
