"""Core steps of the TTS quality evaluation pipeline.

An evaluation run follows this path:
  synthesis (OpenAI TTS) -> duration probe (ffprobe) -> transcription
  (Whisper) -> word error rate and word accuracy -> LLM rubric scoring
  (gpt-5.6-luna) [optional: Gemini audio evaluation with gemini-3.5-flash].

TTS synthesis and Whisper transcription require direct OpenAI access because
OpenRouter does not provide audio synthesis or transcription. Only the LLM
rubric chat evaluation can fall back to OpenRouter. Since direct gpt-5.x access
requires organization verification, the evaluator prefers OpenRouter whenever
OPENROUTER_API_KEY is available (see get_judge_client_and_model).

Public functions handle errors robustly: individual failures raise contextual
exceptions for demo.py to record in the summary table without stopping the run.
"""

import base64
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

import config

# ---------------------------------------------------------------------------
# Client with automatic retries to smooth intermittent network failures.
# ---------------------------------------------------------------------------
_client: Optional[OpenAI] = None


def _new_openai_client(api_key: str, base_url: str | None = None) -> OpenAI:
    kwargs = {"api_key": api_key, "max_retries": 5, "timeout": 60.0}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_client() -> OpenAI:
    """Return the direct OpenAI client for TTS synthesis and Whisper transcription."""
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for direct OpenAI TTS synthesis and Whisper transcription. "
                "Run `export OPENAI_API_KEY=sk-...` or add it to .env."
            )
        _client = _new_openai_client(key)
    return _client


# ---------------------------------------------------------------------------
# LLM rubric client with OpenRouter fallback. Direct gpt-5.x access requires
# organization verification, so prefer OpenRouter when its API key is present.
# Only chat evaluation can fall back; TTS and Whisper still use get_client().
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_judge_client: Optional[OpenAI] = None
_judge_client_kind: str = ""


def _to_openrouter_model(model: str) -> str:
    """Map a model name to an OpenRouter ID.

    IDs containing a slash are already native. gpt-* maps to openai/*,
    claude-* maps to anthropic/claude-opus-4.8, and all other values fall back
    to openai/gpt-5.6-luna.
    """
    if "/" in model:
        return model
    if model.startswith("gpt-"):
        return "openai/" + model
    if model.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    return "openai/gpt-5.6-luna"


def get_judge_client_and_model(model: str):
    """Build the LLM evaluation client and return ``(client, resolved_model)``.

    A gpt-5.x model with OPENROUTER_API_KEY uses OpenRouter first. Otherwise,
    OpenAI is used when OPENAI_API_KEY is available, then OpenRouter with a
    mapped model name. A clear error is raised if neither key is configured.
    """
    global _judge_client, _judge_client_kind
    primary = os.environ.get("OPENAI_API_KEY", "").strip()
    orkey = os.environ.get("OPENROUTER_API_KEY", "").strip()
    prefer_or = bool(orkey) and model.startswith("gpt-5")

    if not prefer_or and primary:
        if _judge_client_kind != "openai":
            _judge_client = _new_openai_client(primary)
            _judge_client_kind = "openai"
        return _judge_client, model
    if orkey:
        if _judge_client_kind != "openrouter":
            _judge_client = _new_openai_client(orkey, OPENROUTER_BASE_URL)
            _judge_client_kind = "openrouter"
        return _judge_client, _to_openrouter_model(model)
    if primary:
        if _judge_client_kind != "openai":
            _judge_client = _new_openai_client(primary)
            _judge_client_kind = "openai"
        return _judge_client, model
    raise RuntimeError(
        "OPENAI_API_KEY or OPENROUTER_API_KEY is required for LLM rubric evaluation."
    )


# ---------------------------------------------------------------------------
# 1) TTS synthesis with multi-provider dispatch.
# ---------------------------------------------------------------------------
def synthesize(cfg: config.TTSConfig, text: str, out_path: str) -> None:
    """Synthesize speech through ``cfg.provider`` and write an MP3 to ``out_path``.

    OpenAI uses its official SDK. Other providers use their public REST APIs
    through built-in urllib without adding dependencies. Missing provider keys
    produce contextual exceptions for the caller to record as a failed row.
    """
    fn = _SYNTH_DISPATCH.get(cfg.provider)
    if fn is None:
        raise RuntimeError(
            f"Unknown provider: {cfg.provider!r} (available: {', '.join(_SYNTH_DISPATCH)})"
        )
    audio = fn(cfg, text)
    if not audio:
        raise RuntimeError(f"{cfg.provider} TTS returned empty audio")
    with open(out_path, "wb") as f:
        f.write(audio)


def _require_env(name: str) -> str:
    # config.env_get supports aliases such as FISH_API_KEY and FISHAUDIO_API_KEY.
    val = config.env_get(name)
    if not val:
        raise RuntimeError(f"Environment variable {name} is required for this provider.")
    return val


def _http_post(url: str, body: dict, headers: dict, timeout: float = 90.0) -> bytes:
    """POST JSON and return raw response bytes; include response excerpts on non-2xx errors."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None


def _synth_openai(cfg: config.TTSConfig, text: str) -> bytes:
    kwargs = dict(model=cfg.model, voice=cfg.voice, input=text)
    if cfg.supports_speed() and abs(cfg.speed - 1.0) > 1e-6:
        kwargs["speed"] = cfg.speed
    return get_client().audio.speech.create(**kwargs).content


def _synth_elevenlabs(cfg: config.TTSConfig, text: str) -> bytes:
    key = _require_env("ELEVENLABS_API_KEY")
    voice = cfg.voice or "21m00Tcm4TlvDq8ikWAM"
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
           f"?output_format=mp3_44100_128")
    body = {"text": text, "model_id": cfg.model or "eleven_multilingual_v2"}
    # ElevenLabs returns raw MP3 bytes.
    return _http_post(url, body, {"xi-api-key": key, "Accept": "audio/mpeg"})


def _synth_fishaudio(cfg: config.TTSConfig, text: str) -> bytes:
    key = _require_env("FISH_API_KEY")
    body = {"text": text, "format": "mp3"}
    if cfg.voice:
        body["reference_id"] = cfg.voice
    # Fish Audio /v1/tts accepts JSON and returns audio bytes directly.
    return _http_post("https://api.fish.audio/v1/tts", body,
                      {"Authorization": f"Bearer {key}"})


def _synth_minimax(cfg: config.TTSConfig, text: str) -> bytes:
    key = _require_env("MINIMAX_API_KEY")
    group = _require_env("MINIMAX_GROUP_ID")
    url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={group}"
    body = {
        "model": cfg.model or "speech-01-turbo",
        "text": text,
        "stream": False,
        "voice_setting": {"voice_id": cfg.voice, "speed": cfg.speed},
        "audio_setting": {"format": "mp3", "sample_rate": 32000},
    }
    raw = _http_post(url, body, {"Authorization": f"Bearer {key}"})
    data = json.loads(raw)
    # The response is JSON with hex-encoded audio at data.audio.
    hexstr = (data.get("data") or {}).get("audio")
    if not hexstr:
        err = data.get("base_resp", {})
        raise RuntimeError(f"Minimax returned no audio: {err or data}")
    return bytes.fromhex(hexstr)


def _synth_doubao(cfg: config.TTSConfig, text: str) -> bytes:
    import uuid
    appid = _require_env("DOUBAO_APP_ID")
    token = _require_env("DOUBAO_ACCESS_TOKEN")
    body = {
        "app": {"appid": appid, "token": token,
                "cluster": cfg.model or "volcano_tts"},
        "user": {"uid": "tts-quality-eval"},
        "audio": {"voice_type": cfg.voice, "encoding": "mp3",
                  "speed_ratio": cfg.speed},
        "request": {"reqid": str(uuid.uuid4()), "text": text, "operation": "query"},
    }
    # Volcengine requires the special Bearer;{token} header and returns base64 audio in data.
    raw = _http_post("https://openspeech.bytedance.com/api/v1/tts", body,
                     {"Authorization": f"Bearer;{token}"})
    data = json.loads(raw)
    b64 = data.get("data")
    if not b64:
        raise RuntimeError(f"Doubao returned no audio: code={data.get('code')} "
                           f"message={data.get('message')}")
    return base64.b64decode(b64)


_SYNTH_DISPATCH = {
    "openai": _synth_openai,
    "elevenlabs": _synth_elevenlabs,
    "fishaudio": _synth_fishaudio,
    "minimax": _synth_minimax,
    "doubao": _synth_doubao,
}


# ---------------------------------------------------------------------------
# 2) Duration probing with ffprobe.
# ---------------------------------------------------------------------------
def probe_duration(path: str) -> float:
    """Return audio duration in seconds, raising if ffprobe is missing or fails."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe was not found; install ffmpeg (macOS: brew install ffmpeg).")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    out = proc.stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise RuntimeError(f"Unable to parse ffprobe duration output: {out!r}")


# ---------------------------------------------------------------------------
# 3) Back-transcription with Whisper.
# ---------------------------------------------------------------------------
# An English prompt guides Whisper toward English output.
_EN_PROMPT = "This is an English sentence."


def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        tr = get_client().audio.transcriptions.create(
            model=config.WHISPER_MODEL, file=f, language="en", prompt=_EN_PROMPT,
        )
    return tr.text or ""


# ---------------------------------------------------------------------------
# 4) Text normalization and English word-level WER.
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def normalize_words(text: str) -> list[str]:
    """Normalize English text to lowercase tokens, preserving inner apostrophes."""
    return _WORD_RE.findall(text.lower())


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Compute word-level Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


@dataclass
class ErrorRate:
    wer: float          # Word error rate: edits divided by reference word count.
    accuracy: float     # Word accuracy: 1 - WER, floored at zero.
    edits: int
    ref_len: int


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        if not hyp:
            return ErrorRate(0.0, 1.0, 0, 0)
        return ErrorRate(1.0, 0.0, len(hyp), 0)
    dist = _edit_distance(ref, hyp)
    wer = dist / len(ref)
    return ErrorRate(wer=wer, accuracy=max(0.0, 1.0 - wer), edits=dist, ref_len=len(ref))


# ---------------------------------------------------------------------------
# 5) LLM rubric evaluation (the default closed loop through OpenAI).
# ---------------------------------------------------------------------------
RUBRIC_DIMENSIONS = ["clarity", "naturalness", "pacing", "overall"]

# Dimension descriptions support --dump-rubric and the evaluation prompt.
# The default back-transcription judge cannot hear the audio, so it cannot
# directly assess emotional expression or voice consistency. Gemini can listen
# to audio with --gemini; voice consistency would additionally need a reference
# voice, which this demo does not provide.
RUBRIC_DESCRIPTIONS = {
    "clarity": "How closely the transcription matches the reference text; omissions, substitutions, and insertions reduce the score.",
    "naturalness": "Whether speaking rate is near natural English delivery (roughly 2–3 words per second); much faster or slower speech is less natural.",
    "pacing": "Whether pauses and rhythm suit the text length and speaking rate; excessive speed often suggests swallowed words and weak pacing.",
    "overall": "Overall quality impression based on the other dimensions.",
}

_JUDGE_SYSTEM = """You are a strict text-to-speech quality evaluation expert.
You receive the original reference text, expected emotion, Whisper transcript of
the synthesized speech, measured audio duration, speaking rate in words per
second, and word error rate (WER). Score synthesized-speech quality from 1 to 5
(5 is best) for each rubric dimension:

- clarity: how closely the transcript matches the reference; more omissions,
  substitutions, or insertions, and higher WER, reduce the score.
- naturalness: whether the rate is close to natural English reading, roughly
  2–3 words per second; rates much faster or slower are less natural.
- pacing: whether pauses and rhythm are appropriate for the text length and
  rate; excessive speed often indicates swallowed words and weak pacing.
- overall: the overall quality impression based on the preceding dimensions.

You cannot hear the audio itself. Make conservative, explainable judgments
using only the supplied measurements. Output JSON only, in exactly this form:
{"clarity": {"score": int, "reason": str},
 "naturalness": {"score": int, "reason": str},
 "pacing": {"score": int, "reason": str},
 "overall": {"score": int, "reason": str}}
Each reason must be a short English sentence."""


@dataclass
class RubricResult:
    scores: dict[str, int]  # Dimension name -> integer score.
    reasons: dict[str, str]  # Dimension name -> reason string.
    raw: str = ""


def _valid_rubric_score(value: object) -> int:
    """Return a valid 1--5 integer rubric score, or zero as the missing-data sentinel."""
    return value if type(value) is int and 1 <= value <= 5 else 0


def parse_rubric_response(data: object, raw: str) -> RubricResult:
    """Normalize LLM and Gemini rubric JSON into bounded scores and safe reasons."""
    response = data if isinstance(data, dict) else {}
    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for dim in RUBRIC_DIMENSIONS:
        item = response.get(dim)
        if isinstance(item, dict):
            scores[dim] = _valid_rubric_score(item.get("score"))
            reason = item.get("reason", "")
            reasons[dim] = reason.strip() if isinstance(reason, str) else ""
        else:
            scores[dim] = _valid_rubric_score(item)
            reasons[dim] = ""
    return RubricResult(scores=scores, reasons=reasons, raw=raw)


def _rubric_from_json(raw: str) -> RubricResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Judge returned invalid JSON: {raw[:300]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Judge returned JSON object expected: {raw[:300]}")
    return parse_rubric_response(data, raw)


def judge_rubric(reference: str, emotion: str, hypothesis: str,
                 duration: float, wer: float, model: Optional[str] = None) -> RubricResult:
    """Score a transcript against the English rubric using the evaluation model.

    The chat request can fall back to OpenRouter (see get_judge_client_and_model).
    """
    words = len(normalize_words(reference))
    speed = words / duration if duration > 0 else 0.0
    user = (
        f"Reference text: {reference}\n"
        f"Expected emotion: {emotion}\n"
        f"Whisper transcript: {hypothesis}\n"
        f"Audio duration: {duration:.2f} seconds\n"
        f"Speaking rate: {speed:.2f} words/second ({words} reference words)\n"
        f"Word error rate (WER): {wer:.3f}\n"
    )
    judge_client, judge_model = get_judge_client_and_model(model or config.JUDGE_MODEL)
    resp = judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    return _rubric_from_json(raw)


# ---------------------------------------------------------------------------
# 6) Optional Gemini multimodal audio evaluation. REST avoids another SDK.
# ---------------------------------------------------------------------------
def _resolve_gemini_model(api_key: str) -> str:
    """Discover a currently available Gemini model so an expired default can be avoided."""
    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
        names = [m["name"].split("/")[-1] for m in data.get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        # Prefer the verified audio-capable default, then Pro or older Flash models.
        for want in (config.GEMINI_MODEL_DEFAULT, "gemini-3.5-flash",
                     "gemini-2.5-pro", "gemini-2.5-flash", "gemini-flash-latest"):
            if want in names:
                return want
        # Last resort: any available model that is not TTS, image, or embedding-only.
        for n in names:
            if "tts" not in n and "image" not in n and "embedding" not in n:
                return n
    except Exception:
        pass
    return config.GEMINI_MODEL_DEFAULT


def _gemini_response_text(data: object) -> str:
    if isinstance(data, dict):
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            content = candidate.get("content") if isinstance(candidate, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part["text"]
    raise RuntimeError(f"Gemini did not return evaluation text: {str(data)[:300]}")


def judge_gemini_audio(reference: str, emotion: str, audio_path: str) -> RubricResult:
    """Have Gemini listen directly to synthesized audio and score the English rubric.

    Requires GEMINI_API_KEY and is disabled unless --gemini is selected.
    Failures raise contextual exceptions for the caller to record as failed rows.
    """
    import urllib.request
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini audio evaluation.")
    model = _resolve_gemini_model(key)
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    rubric = "\n".join(
        f"- {dim}: {RUBRIC_DESCRIPTIONS[dim]}" for dim in RUBRIC_DIMENSIONS
    )
    prompt = (
        "You are a TTS quality evaluation expert. Listen directly to the synthesized audio "
        "below, compare it with the reference text and expected emotion, and score four "
        "dimensions from 1 to 5 with short English reasons. Use this rubric:\n"
        f"{rubric}\n"
        "Use roughly 2–3 words per second as the natural-English reading-rate guide. "
        "Output JSON only: "
        '{"clarity":{"score":int,"reason":str},"naturalness":{"score":int,"reason":str},'
        '"pacing":{"score":int,"reason":str},"overall":{"score":int,"reason":str}}\n'
        f"Reference text: {reference}\nExpected emotion: {emotion}"
    )
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "audio/mp3", "data": audio_b64}},
        ]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read())
    text = _gemini_response_text(data)
    return _rubric_from_json(text)
