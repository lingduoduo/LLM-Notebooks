"""prepare_jevbench_snapshot + render_jevbench_snapshot on a synthetic archive.

Everything is written under tmp_path; the archive below is test data only.
"""
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_jevbench_snapshot")
render = load("render_jevbench_snapshot")


def fake_run(model_id, n=6):
    rows = [
        {"task_id": f"easy-intent-{i:02d}", "family": "intent", "status": "ok",
         "valid": True, "correct_value": i % 2 == 0}
        for i in range(n)
    ]
    summary = {"n_scorable": n, "accuracy": 0.5, "brier": 0.2, "ece": 0.05,
               "p50_s": 0.7, "p95_s": 1.0, "cost_usd_total": 0.01,
               "jevbench_score": {"score": 60.0, "axes": {}}}
    return summary, rows


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "runs.zip"
    with zipfile.ZipFile(path, "w") as z:
        for model_id in prepare.MODEL_IDS:
            summary, rows = fake_run(model_id)
            base = f"runs/notebook-demo/real/{model_id}/"
            z.writestr(base + "summary.json", json.dumps(summary))
            z.writestr(base + "results.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    return path


def run_prepare(monkeypatch, tmp_path, zip_path, date="2030-01-02"):
    monkeypatch.setattr(prepare, "CHAPTER", tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare", "--zip", str(zip_path), "--date", date])
    prepare.main()
    return tmp_path / "figures" / f"jevbench-{date}" / "snapshot.json"


def test_lineup_is_jev_plus_openai_models():
    from llm_eval.adapters import OPENAI_MODELS

    assert prepare.MODEL_IDS == ["jev", *OPENAI_MODELS]
    assert not any(
        old in model_id
        for model_id in prepare.MODEL_IDS
        for old in ("deepseek", "doubao", "glm", "kimi", "qwen", "step", "mimo")
    )


def test_prepare_writes_dated_snapshot_for_every_model(monkeypatch, tmp_path, archive):
    out = run_prepare(monkeypatch, tmp_path, archive)
    data = json.loads(out.read_text())
    assert data["metadata"]["snapshot_date"] == "2030-01-02"
    assert data["metadata"]["models_order"] == prepare.MODEL_IDS
    assert [m["id"] for m in data["models"]] == prepare.MODEL_IDS


def test_prepare_skips_models_missing_from_archive(monkeypatch, tmp_path, archive, capsys):
    partial = tmp_path / "partial.zip"
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(partial, "w") as dst:
        for name in src.namelist():
            if "/real/jev/" not in name:
                dst.writestr(name, src.read(name))
    data = json.loads(run_prepare(monkeypatch, tmp_path, partial).read_text())
    assert "jev" not in data["metadata"]["models_order"]
    assert "skip jev" in capsys.readouterr().out


def test_render_fits_six_models_on_every_canvas(monkeypatch, tmp_path, archive):
    snapshot = run_prepare(monkeypatch, tmp_path, archive)
    monkeypatch.setattr(sys, "argv", ["render", "--snapshot", str(snapshot)])
    render.main()
    svgs = sorted(snapshot.parent.glob("*.svg"))
    assert len(svgs) == 5
    for svg in svgs:
        s = svg.read_text()
        width, height = map(float, re.search(r'<svg[^>]*width="([\d.]+)" height="([\d.]+)"', s).groups())
        for x, y, w, h in re.findall(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)"', s):
            assert float(x) + float(w) <= width and float(y) + float(h) <= height, svg.name
        for x, y in re.findall(r'<text x="([-\d.]+)" y="([-\d.]+)"', s):
            assert float(x) <= width and float(y) <= height, svg.name
    dashboard = (snapshot.parent / "dashboard.svg").read_text()
    assert f"{len(prepare.MODEL_IDS)}-provider snapshot" in dashboard
    assert "2 Jan 2030" in dashboard


def test_layout_helpers_keep_five_model_geometry():
    # Five rows/columns must reproduce the original fixed layout exactly.
    assert render.row_positions(249, 5, 156, 39) == [249 + 39 * i for i in range(5)]
    assert render.column_width(5, 1540 - 340 - 70, 16, 207) == 207
    assert render.column_width(6, 1540 - 340 - 70, 16, 207) < 207


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
