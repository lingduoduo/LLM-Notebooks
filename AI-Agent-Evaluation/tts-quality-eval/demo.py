"""Experiment 6-5: a fully automated TTS quality evaluation pipeline.

    python demo.py                         # Four default OpenAI configurations x four samples
    python demo.py --providers openai,minimax  # Compare providers side by side
    python demo.py --text 'A custom sentence'  # Evaluate custom text
    python demo.py --gemini                # Have Gemini listen to audio directly (GEMINI_API_KEY required)
    python demo.py --quick                 # Use the first two samples for a quick smoke test
    python demo.py --list-providers        # Offline: show provider configuration state
    python demo.py --dump-rubric           # Offline: show rubric dimensions

Flow: multi-provider TTS synthesis -> ffprobe duration -> Whisper transcript
      -> WER/word accuracy -> LLM/Gemini rubric scoring -> detailed records and summary table.
Audio is written to output/ and reused unless --fresh is supplied. Run
`python demo.py --help` for all options.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from statistics import mean
from typing import Optional

import config
import pipeline

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def load_env() -> None:
    """Load .env values without adding a dependency."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def audio_path(cfg_name: str, sample_id: str, reference_text: str) -> str:
    """Return a cache path that changes whenever its reference text changes."""
    text_hash = hashlib.sha256(reference_text.encode("utf-8")).hexdigest()[:12]
    return os.path.join(OUT_DIR, f"{cfg_name}__{sample_id}__{text_hash}.mp3")


def evaluate_one(cfg: config.TTSConfig, sample: config.Sample, use_gemini: bool, fresh: bool,
                 judge_model: Optional[str] = None) -> dict[str, object]:
    """Evaluate one configuration/sample pair and return errors as records."""
    rec = {"config": cfg.name, "sample": sample.id, "challenge": sample.challenge,
           "provider": getattr(cfg, "provider", "openai"), "ok": False, "error": None}
    path = audio_path(cfg.name, sample.id, sample.text)
    try:
        # 1) Synthesize, reusing a nonempty file unless --fresh was supplied.
        if fresh or not os.path.exists(path) or os.path.getsize(path) == 0:
            pipeline.synthesize(cfg, sample.text, path)
        # 2) Measure duration.
        dur = pipeline.probe_duration(path)
        # 3) Back-transcribe with Whisper.
        hyp = pipeline.transcribe(path)
        # 4) Compute WER and word accuracy.
        er = pipeline.word_error_rate(sample.text, hyp)
        # 5) Score the rubric.
        if use_gemini:
            rub = pipeline.judge_gemini_audio(sample.text, sample.emotion, path)
        else:
            rub = pipeline.judge_rubric(sample.text, sample.emotion, hyp, dur, er.wer,
                                        model=judge_model)
        rec.update({
            "ok": True, "duration": dur, "hypothesis": hyp,
            "wer": er.wer, "word_accuracy": er.accuracy,
            "words_per_second": (er.ref_len / dur) if dur else 0.0,
            "scores": rub.scores, "reasons": rub.reasons,
        })
    except Exception as e:  # A failed record does not stop the full evaluation.
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def print_detail(rec, sample_text):
    head = f"[{rec['config']} | {rec['sample']}] {rec['challenge']}"
    if not rec["ok"]:
        print(f"  {head}\n    !! Failed: {rec['error']}")
        return
    print(f"  {head}")
    print(f"    Reference : {sample_text}")
    print(f"    Transcript: {rec['hypothesis']}")
    print(f"    Duration  : {fmt(rec['duration'])}s   Words/s: {fmt(rec['words_per_second'])}"
          f"   WER: {fmt(rec['wer'], 3)}   Word accuracy: {fmt(rec['word_accuracy'] * 100, 1)}%")
    s, r = rec["scores"], rec["reasons"]
    for dim in pipeline.RUBRIC_DIMENSIONS:
        print(f"    {dim:<12}: {s.get(dim, '-')}/5  {r.get(dim, '')}")


def summarize(records):
    """Aggregate each configuration's word metrics, rubric means, and successes."""
    by_cfg = {}
    for rec in records:
        by_cfg.setdefault(rec["config"], []).append(rec)
    rows = []
    for cfg_name, recs in by_cfg.items():
        ok = [r for r in recs if r["ok"]]
        row = {"config": cfg_name, "n_ok": len(ok), "n": len(recs)}
        if ok:
            row["wer"] = mean(r["wer"] for r in ok)
            row["word_accuracy"] = mean(r["word_accuracy"] for r in ok)
            row["words_per_second"] = mean(r["words_per_second"] for r in ok)
            for dim in pipeline.RUBRIC_DIMENSIONS:
                row[dim] = mean(r["scores"].get(dim, 0) for r in ok)
        rows.append(row)
    # Sort by descending overall score, then ascending WER.
    rows.sort(key=lambda row: (-row.get("overall", 0), row.get("wer", 1)))
    return rows


def print_table(rows):
    cols = ["overall", "clarity", "naturalness", "pacing"]
    header = (f"{'Configuration':<18}{'Success':>8}{'Word accuracy':>15}{'WER':>8}{'Words/s':>10}"
              + "".join(f"{c:>9}" for c in cols))
    print(header)
    print("-" * len(header))
    for r in rows:
        ok_str = f"{r['n_ok']}/{r['n']}"
        if not r.get("n_ok"):
            print(f"{r['config']:<18}{ok_str:>8}   (all failed)")
            continue
        acc = f"{r['word_accuracy'] * 100:.1f}%"
        line = (f"{r['config']:<18}{ok_str:>8}{acc:>15}{r['wer']:>8.3f}"
                f"{r['words_per_second']:>10.2f}")
        line += "".join(f"{r.get(c,0):>9.2f}" for c in cols)
        print(line)


def print_providers() -> None:
    """Print all TTS providers and their configuration state without network access."""
    print("Available TTS providers (OpenAI / ElevenLabs / Fish Audio / Minimax / Doubao):\n")
    for key, p in config.PROVIDERS.items():
        state = "configured" if p.configured() else "not configured"
        env = " + ".join(p.env)
        print(f"  [{key}]  {p.label}   ({state}; requires {env})")
        print(f"      {p.note}")
    print("\nUse --providers openai,minimax to compare providers (the default is OpenAI only).")
    print("Non-OpenAI providers require their own keys (see env.example); a missing key fails only that row.")


def print_rubric() -> None:
    """Print rubric dimension definitions without network access."""
    print("TTS quality evaluation rubric (1–5, where 5 is best):\n")
    for dim in pipeline.RUBRIC_DIMENSIONS:
        print(f"  {dim}: {pipeline.RUBRIC_DESCRIPTIONS.get(dim, '')}")
    print("\nThe default Whisper transcript + LLM judge conservatively scores the transcript, duration,")
    print("speaking rate, and WER. --gemini lets a multimodal model listen to the audio directly,")
    print("which can assess emotional delivery and vocal consistency.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the TTS evaluation."""
    ap = argparse.ArgumentParser(
        description="Fully automated TTS quality evaluation (Experiment 6-5): multi-provider synthesis and LLM rubric scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python demo.py                              Four OpenAI configurations × four samples\n"
               "  python demo.py --providers openai,minimax   Compare providers side by side\n"
               "  python demo.py --text 'The weather is pleasant today.' --gemini   Custom text + Gemini audio judge\n"
               "  python demo.py --list-providers             View offline provider configuration state\n"
               "  python demo.py --dump-rubric                View offline rubric dimensions",
    )
    ap.add_argument("--text", metavar="TEXT",
                    help="Replace the evaluation corpus with one custom sentence")
    ap.add_argument("--providers", metavar="LIST",
                    help="Comma-separated providers (openai,elevenlabs,fishaudio,minimax,doubao); "
                         "each uses one representative configuration; default: multiple OpenAI configurations")
    ap.add_argument("--judge-model", metavar="MODEL", dest="judge_model",
                    help=f"Override the LLM judge model (default: {config.JUDGE_MODEL}); ignored with --gemini")
    ap.add_argument("--output", metavar="DIRECTORY",
                    help=f"Output directory for audio and results.json (default: {OUT_DIR})")
    ap.add_argument("--extra", action="store_true", help="Include the gpt-4o-mini-tts configuration")
    ap.add_argument("--gemini", action="store_true", help="Use Gemini to evaluate audio directly (requires GEMINI_API_KEY)")
    ap.add_argument("--quick", action="store_true", help="Use only the first two samples for a quick smoke test")
    ap.add_argument("--fresh", action="store_true", help="Ignore existing audio and synthesize every sample again")
    ap.add_argument("--list-providers", action="store_true", dest="list_providers",
                    help="Print all TTS providers and their configuration state, then exit (no key required)")
    ap.add_argument("--dump-rubric", action="store_true", dest="dump_rubric",
                    help="Print rubric dimension definitions, then exit (no key required)")
    return ap.parse_args(argv)


def select_configs(provider_names: str | None, include_extra: bool) -> list[config.TTSConfig]:
    """Return the requested provider configurations in command-line order."""
    if not provider_names:
        return list(config.TTS_CONFIGS) + (list(config.EXTRA_CONFIGS) if include_extra else [])

    selected = []
    for name in (item.strip() for item in provider_names.split(",")):
        if not name:
            continue
        try:
            selected.append(config.PROVIDER_CONFIGS[name])
        except KeyError:
            available = ", ".join(config.PROVIDER_CONFIGS)
            raise ValueError(f"unknown provider {name!r}. Available: {available}") from None
    return selected


def select_corpus(text: str | None, quick: bool) -> list[config.Sample]:
    """Return either one custom sample or the selected built-in corpus."""
    if text:
        return [config.Sample("custom", text, "custom text", "neutral")]
    return list(config.CORPUS[:2] if quick else config.CORPUS)


def run_evaluations(
    configs: list[config.TTSConfig],
    corpus: list[config.Sample],
    use_gemini: bool,
    fresh: bool,
    judge_model: str | None,
) -> list[dict]:
    """Evaluate each configuration/sample pair in grid order."""
    records = []
    for cfg in configs:
        print(f"\n### Configuration {cfg.name}  (provider={getattr(cfg, 'provider', 'openai')}, "
              f"model={cfg.model}, voice={cfg.voice}, speed={cfg.speed})")
        for sample in corpus:
            record = evaluate_one(cfg, sample, use_gemini, fresh, judge_model)
            print_detail(record, sample.text)
            records.append(record)
    return records


def main() -> None:
    global OUT_DIR
    args = parse_args()

    load_env()

    # Offline paths do not need network access or API keys.
    if args.list_providers:
        print_providers()
        return
    if args.dump_rubric:
        print_rubric()
        return

    if args.output:
        OUT_DIR = os.path.abspath(args.output)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("Error: OPENAI_API_KEY is required for transcription and the default judge. "
              "Set it with export or add it to .env, then try again.",
              file=sys.stderr)
        sys.exit(1)

    try:
        configs = select_configs(args.providers, args.extra)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    corpus = select_corpus(args.text, args.quick)

    judge_model = args.judge_model or config.JUDGE_MODEL
    mode = ("Gemini multimodal audio evaluation" if args.gemini
            else f"Whisper transcript + LLM rubric ({judge_model})")
    providers_used = sorted({getattr(c, "provider", "openai") for c in configs})
    print("=" * 72)
    print("Experiment 6-5: Fully Automated TTS Quality Evaluation")
    print(f"Evaluation mode: {mode}")
    print(f"Providers: {', '.join(providers_used)}")
    print(f"Configurations: {len(configs)}   Samples: {len(corpus)}   "
          f"Total evaluations: {len(configs) * len(corpus)}")
    print("=" * 72)

    t0 = time.time()
    records = run_evaluations(
        configs, corpus, args.gemini, args.fresh,
        judge_model=None if args.gemini else args.judge_model,
    )

    rows = summarize(records)
    print("\n" + "=" * 72)
    print("Configuration Summary (sorted by overall score, then WER)")
    print("=" * 72)
    print_table(rows)

    ok = sum(1 for r in records if r["ok"])
    print(f"\nCompleted: {ok}/{len(records)} evaluations succeeded in {time.time() - t0:.1f}s.")

    # Persist structured results for later analysis.
    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"Detailed results written to {out_json}")


if __name__ == "__main__":
    main()
