"""SFT rows with messages shorter than 2 must not IndexError in analyze_data."""

import json
import sys
from pathlib import Path

import analyze_data as ad


def test_short_messages_skipped_without_index_error(tmp_path, monkeypatch, capsys):
    sft = tmp_path / "sft.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "only user"}]},
        {"messages": []},
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "<think>\nLet me verify that once more\n</think>\nFinal Answer: 1",
                },
            ]
        },
    ]
    sft.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_data.py", "--sft", str(sft), "--raw", str(tmp_path / "missing.jsonl")],
    )
    ad.main()
    out = capsys.readouterr().out
    assert "SFT samples: 3" in out
    assert "Skipped samples with fewer than 2 messages: 2" in out
    assert "Samples with reflection/verification behavior: 1/1" in out


def test_normal_two_message_row_still_scored(tmp_path, monkeypatch, capsys):
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "<think>\nok\n</think>\n1"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_data.py", "--sft", str(sft), "--raw", str(tmp_path / "missing.jsonl")],
    )
    ad.main()
    out = capsys.readouterr().out
    assert "Skipped samples" not in out
    assert "Samples with reflection/verification behavior: 0/1" in out
