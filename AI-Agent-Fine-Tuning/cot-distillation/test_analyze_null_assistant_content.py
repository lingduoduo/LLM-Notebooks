"""SFT rows with null assistant content must not TypeError in analyze_data."""

import json
import sys

import analyze_data as ad


def test_null_assistant_content_skipped(tmp_path, monkeypatch, capsys):
    sft = tmp_path / "sft.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None},
            ]
        },
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
    assert "SFT samples: 2" in out
    assert "Skipped samples with fewer than 2 messages: 1" in out
    assert "Samples with reflection/verification behavior: 1/1" in out


def test_missing_content_key_skipped(tmp_path, monkeypatch, capsys):
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant"},
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
    assert "Skipped samples with fewer than 2 messages: 1" in out
