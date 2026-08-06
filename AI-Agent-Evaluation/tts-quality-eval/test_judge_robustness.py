"""
Regression tests for judge-response robustness (Experiment 6-5 TTS quality evaluation).

Covers two failure classes on LLM/Gemini judge responses:
  - judge_rubric: judge returns "score": null (or a bare null dimension) -> int(None) TypeError
  - judge_gemini_audio: safety-blocked Gemini responses have no
    candidates/content/parts -> KeyError/IndexError instead of a clear error

Network is stubbed: the OpenAI-compatible judge client is replaced with a fake,
and urllib.request.urlopen is monkeypatched for the Gemini REST call.
"""
import io
import json
from types import SimpleNamespace

import pytest

import demo
import pipeline


class _FakeMessage:
    content = "{}"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    choices = [_FakeChoice()]


class _FakeCompletions:
    @staticmethod
    def create(**kwargs):
        return _FakeResp()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


def _stub_judge(monkeypatch, payload: dict):
    _FakeMessage.content = json.dumps(payload, ensure_ascii=False)
    monkeypatch.setattr(
        pipeline, "get_judge_client_and_model", lambda model=None: (_FakeClient(), "fake-judge"))


def test_get_judge_client_reuses_matching_backend(monkeypatch):
    """Matching direct-OpenAI judge calls reuse their configured client."""
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(pipeline, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(pipeline, "_judge_client", None)
    monkeypatch.setattr(pipeline, "_judge_client_kind", "")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    first, first_model = pipeline.get_judge_client_and_model("gpt-4.1")
    second, second_model = pipeline.get_judge_client_and_model("gpt-4.1-mini")

    assert first is second
    assert (first_model, second_model) == ("gpt-4.1", "gpt-4.1-mini")
    assert created == [{"api_key": "openai-key", "max_retries": 5, "timeout": 60.0}]


def test_get_judge_client_switches_to_openrouter_and_preserves_native_model(monkeypatch):
    """A gpt-5 model selects OpenRouter and retains an already-native model ID."""
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(pipeline, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(pipeline, "_judge_client", None)
    monkeypatch.setattr(pipeline, "_judge_client_kind", "")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    direct, direct_model = pipeline.get_judge_client_and_model("gpt-4.1")
    monkeypatch.delenv("OPENAI_API_KEY")
    routed, routed_model = pipeline.get_judge_client_and_model("openai/gpt-5.6-luna")

    assert direct is not routed
    assert (direct_model, routed_model) == ("gpt-4.1", "openai/gpt-5.6-luna")
    assert created == [
        {"api_key": "openai-key", "max_retries": 5, "timeout": 60.0},
        {"api_key": "openrouter-key", "max_retries": 5, "timeout": 60.0,
         "base_url": pipeline.OPENROUTER_BASE_URL},
    ]


def test_rubric_from_json_rejects_non_object_root():
    """A JSON array cannot be normalized as a rubric response."""
    with pytest.raises(RuntimeError, match="JSON object"):
        pipeline._rubric_from_json("[]")


def test_gemini_response_text_returns_first_text_part():
    """Gemini responses use the first available text part as their rubric payload."""
    data = {"candidates": [{"content": {"parts": [{"text": "rubric-json"}]}}]}
    assert pipeline._gemini_response_text(data) == "rubric-json"


def test_judge_paths_delegate_raw_responses_to_shared_rubric_parser(monkeypatch, tmp_path):
    """Both judge transports send their raw JSON text through one rubric parser."""
    received = []
    expected = pipeline.RubricResult(scores={}, reasons={})

    def record_rubric(raw):
        received.append(raw)
        return expected

    _stub_judge(monkeypatch, {"clarity": 4})
    _stub_gemini(monkeypatch, {
        "candidates": [{"content": {"parts": [{"text": "gemini-json"}]}}],
    })
    monkeypatch.setattr(pipeline, "_rubric_from_json", record_rubric)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 256)

    openai_result = pipeline.judge_rubric(
        "Reference text", "neutral", "Transcript text", 3.0, 0.05,
    )
    gemini_result = pipeline.judge_gemini_audio("Reference text", "neutral", str(audio))

    assert (openai_result, gemini_result) == (expected, expected)
    assert received == [json.dumps({"clarity": 4}), "gemini-json"]


def test_normalize_words_handles_case_punctuation_and_contractions():
    assert pipeline.normalize_words("Hello, WORLD! Don't stop.") == [
        "hello", "world", "don't", "stop"
    ]


def test_word_error_rate_counts_word_edits():
    result = pipeline.word_error_rate(
        "The quick brown fox", "The fast brown fox jumps"
    )
    assert result.edits == 2
    assert result.ref_len == 4
    assert result.wer == pytest.approx(0.5)
    assert result.accuracy == pytest.approx(0.5)


def test_word_error_rate_empty_reference_is_perfect():
    assert pipeline.word_error_rate("", "").wer == 0.0
    assert pipeline.word_error_rate("", "").accuracy == 1.0


def test_word_error_rate_empty_reference_counts_nonempty_hypothesis_as_errors():
    """Unrequested transcript words are errors even when reference normalization is empty."""
    result = pipeline.word_error_rate("...", "one two three")
    assert result.edits == 3
    assert result.ref_len == 0
    assert result.wer == 1.0
    assert result.accuracy == 0.0


def test_audio_path_includes_a_stable_reference_text_hash(monkeypatch, tmp_path):
    """Cache paths must not reuse audio when a custom or corpus text changes."""
    monkeypatch.setattr(demo, "OUT_DIR", str(tmp_path))
    original = demo.audio_path("test-config", "custom", "alpha")
    changed = demo.audio_path("test-config", "custom", "beta")

    assert original.endswith("test-config__custom__8ed3f6ad685b.mp3")
    assert demo.audio_path("test-config", "custom", "alpha") == original
    assert changed != original


def test_evaluate_one_reports_word_metrics_without_legacy_character_fields(monkeypatch, tmp_path):
    """The CLI record exposes WER-derived word metrics to downstream JSON consumers."""
    cfg = SimpleNamespace(name="test-config", provider="openai")
    sample = SimpleNamespace(
        id="test-sample",
        text="One two three four",
        challenge="word metric fixture",
        emotion="neutral",
    )
    captured = {}

    monkeypatch.setattr(demo, "audio_path", lambda *_: str(tmp_path / "clip.mp3"))
    monkeypatch.setattr(pipeline, "synthesize", lambda *_: None)
    monkeypatch.setattr(pipeline, "probe_duration", lambda _: 4.0)
    monkeypatch.setattr(pipeline, "transcribe", lambda _: "One two three")
    monkeypatch.setattr(
        pipeline,
        "word_error_rate",
        lambda *_: pipeline.ErrorRate(wer=0.25, accuracy=0.75, edits=1, ref_len=4),
    )

    def judge(reference, emotion, hypothesis, duration, wer, model=None):
        captured["wer"] = wer
        return pipeline.RubricResult(
            scores={"clarity": 4, "naturalness": 4, "pacing": 4, "overall": 4},
            reasons={"clarity": "Clear.", "naturalness": "Natural.",
                     "pacing": "Even.", "overall": "Strong."},
        )

    monkeypatch.setattr(pipeline, "judge_rubric", judge)

    record = demo.evaluate_one(cfg, sample, use_gemini=False, fresh=True)

    assert captured["wer"] == pytest.approx(0.25)
    assert record["wer"] == pytest.approx(0.25)
    assert record["word_accuracy"] == pytest.approx(0.75)
    assert record["words_per_second"] == pytest.approx(1.0)
    assert "cer" not in record
    assert "accuracy" not in record
    assert "speed" not in record


def test_summarize_sorts_descending_overall_then_ascending_wer():
    """Configuration summaries prefer higher overall scores, then lower WER."""
    records = [
        {"config": "tie-higher-wer", "ok": True, "wer": 0.30,
         "word_accuracy": 0.70, "words_per_second": 2.0,
         "scores": {"clarity": 4, "naturalness": 4, "pacing": 4, "overall": 5}},
        {"config": "lower-overall", "ok": True, "wer": 0.00,
         "word_accuracy": 1.00, "words_per_second": 2.0,
         "scores": {"clarity": 5, "naturalness": 5, "pacing": 5, "overall": 4}},
        {"config": "tie-lower-wer", "ok": True, "wer": 0.10,
         "word_accuracy": 0.90, "words_per_second": 2.0,
         "scores": {"clarity": 4, "naturalness": 4, "pacing": 4, "overall": 5}},
    ]

    rows = demo.summarize(records)

    assert [row["config"] for row in rows] == [
        "tie-lower-wer", "tie-higher-wer", "lower-overall"
    ]


def test_judge_rubric_tolerates_null_score(monkeypatch):
    """'score': null in a dimension dict is scored 0, not int(None) TypeError."""
    _stub_judge(monkeypatch, {
        "clarity": {"score": None, "reason": "Unable to determine"},
        "naturalness": {"score": 4, "reason": "The speaking rate is natural"},
        "pacing": {"score": 3},
        "overall": {"score": 5, "reason": "Usable overall"},
    })
    rub = pipeline.judge_rubric("Reference text", "neutral", "Transcript text", 3.0, 0.05)
    assert rub.scores["clarity"] == 0
    assert rub.scores["naturalness"] == 4
    assert rub.scores["overall"] == 5


def test_judge_rubric_tolerates_null_dimension(monkeypatch):
    """Null or omitted dimensions become the zero sentinel without raising."""
    _stub_judge(monkeypatch, {"clarity": None, "naturalness": 4, "overall": 5})
    rub = pipeline.judge_rubric("Reference text", "neutral", "Transcript text", 3.0, 0.05)
    assert rub.scores["clarity"] == 0
    assert rub.scores["naturalness"] == 4
    assert rub.scores["pacing"] == 0


def test_judge_rubric_rejects_malformed_and_out_of_range_scores(monkeypatch):
    """Invalid rubric scores must be sentinels instead of affecting configuration means."""
    _stub_judge(monkeypatch, {
        "clarity": {"score": 6, "reason": "too high"},
        "naturalness": {"score": "5", "reason": "wrong type"},
        "pacing": {"score": 1, "reason": "valid lower bound"},
        "overall": {"score": 5, "reason": "valid upper bound"},
    })

    rub = pipeline.judge_rubric("Reference text", "neutral", "Transcript text", 3.0, 0.05)

    assert rub.scores == {"clarity": 0, "naturalness": 0, "pacing": 1, "overall": 5}


class _FakeHTTPResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _stub_gemini(monkeypatch, payload: dict):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(pipeline, "_resolve_gemini_model", lambda key: "gemini-fake")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeHTTPResp(json.dumps(payload).encode()))


@pytest.mark.parametrize("payload", [
    {"promptFeedback": {"blockReason": "SAFETY"}},      # Prompt blocked: no candidates.
    {"candidates": []},                                    # Generation blocked: empty candidates.
    {"candidates": [{"finishReason": "SAFETY", "index": 0}]},  # Candidate has no content.
])
def test_judge_gemini_audio_blocked_raises_clear_error(monkeypatch, tmp_path, payload):
    """Blocked/empty Gemini responses raise a clear RuntimeError, not KeyError/IndexError."""
    _stub_gemini(monkeypatch, payload)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 256)
    with pytest.raises(RuntimeError, match="Gemini did not return evaluation text"):
        pipeline.judge_gemini_audio("Reference text", "neutral", str(audio))


def test_judge_gemini_audio_parses_valid_response(monkeypatch, tmp_path):
    """A normal Gemini response still parses (defensive navigation keeps working)."""
    inner = json.dumps({"clarity": {"score": 4, "reason": "ok"}, "naturalness": 4,
                        "pacing": None, "overall": {"score": 5}})
    _stub_gemini(monkeypatch, {
        "candidates": [{"content": {"parts": [{"text": inner}]}}],
    })
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 256)
    rub = pipeline.judge_gemini_audio("Reference text", "neutral", str(audio))
    assert rub.scores["clarity"] == 4
    assert rub.scores["pacing"] == 0   # Null score becomes 0.
    assert rub.scores["overall"] == 5


def test_judge_gemini_audio_rejects_malformed_and_out_of_range_scores(monkeypatch, tmp_path):
    """Gemini must apply the same 1-to-5 validation as the LLM rubric path."""
    inner = json.dumps({"clarity": {"score": -1}, "naturalness": "4",
                        "pacing": {"score": 1}, "overall": {"score": 5}})
    _stub_gemini(monkeypatch, {
        "candidates": [{"content": {"parts": [{"text": inner}]}}],
    })
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 256)

    rub = pipeline.judge_gemini_audio("Reference text", "neutral", str(audio))

    assert rub.scores == {"clarity": 0, "naturalness": 0, "pacing": 1, "overall": 5}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
