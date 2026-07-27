"""Regression test: num_frames=0 must not cause ZeroDivisionError.

The LLM-supplied num_frames parameter was used directly as a divisor in
`frame_count // num_frames`; num_frames=0 crashed with ZeroDivisionError
(surfacing as a confusing tool error). It is now clamped to >= 1 up front.
"""
import asyncio
import json
from types import SimpleNamespace

import cv2
import numpy as np

from perception_tools import media_processing_tools
from perception_tools.media_processing_tools import extract_video_keyframes


def _make_clip(path, frames=20):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for _ in range(frames):
        out.write(np.zeros((48, 64, 3), dtype=np.uint8))
    out.release()


def test_extract_keyframes_zero_num_frames_is_clamped(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    result = asyncio.run(extract_video_keyframes(str(clip), num_frames=0))
    payload = json.loads(result.text)
    assert payload["success"] is True
    assert "division" not in str(payload["message"]).lower()


def test_analyze_video_ai_zero_num_frames_is_clamped(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)

    message = SimpleNamespace(content="a frame")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kwargs: response)))
    monkeypatch.setattr(media_processing_tools, "_make_vision_client",
                        lambda: (client, "fake-model"))

    result = asyncio.run(media_processing_tools.analyze_video_ai(str(clip), num_frames=0))
    payload = json.loads(result.text)
    assert payload["success"] is True
    assert "division" not in str(payload["message"]).lower()
