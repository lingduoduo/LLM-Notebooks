"""Build, align and monitor a recommendation explanation judge."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from benchmark import load_jsonl
from judge import DemoJudge, MetaJudge, Reflector, exact_score_agreement, pass_fail_metrics
from generation import DemoGenerator, generate_with_guardrail
from llm_backend import BackendError, ChatClient, LLMGenerator, LLMJudge, LLMReflector
from monitoring import check_drift, random_sample

ROOT = Path(__file__).resolve().parent


def fingerprint(judge):
    config = {'rubric': judge.rubric, 'backend': type(judge).__name__,
              'model': getattr(getattr(judge, 'client', None), 'model', None),
              'base_url': getattr(getattr(judge, 'client', None), 'base_url', None),
              'decouple_relevance': getattr(judge, 'decouple_relevance', None)}
    return {**config, 'version': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]}


def evaluate(judge, examples):
    results = [judge.evaluate(ex) for ex in examples]
    return results, {'agreement': exact_score_agreement(examples, results),
                     'pass_fail': pass_fail_metrics(examples, results)}


def assert_disjoint(*datasets):
    ids, contents = set(), set()
    for examples in datasets:
        for ex in examples:
            content = json.dumps([asdict(ex.user), asdict(ex.item), ex.explanation], sort_keys=True)
            if ex.id in ids or content in contents:
                raise ValueError('Datasets must have distinct IDs and no duplicate examples')
            ids.add(ex.id)
            contents.add(content)


def run(args):
    calibration = load_jsonl(args.benchmark)
    validation = load_jsonl(args.validation)
    production = load_jsonl(args.production)
    assert_disjoint(calibration, validation, production)
    if args.backend == 'demo':
        judge, generator, reflector = DemoJudge(), DemoGenerator(), Reflector()
    else:
        client = ChatClient(args.model, args.base_url, os.getenv('OPENAI_API_KEY', ''), args.timeout)
        judge, generator, reflector = LLMJudge(client), LLMGenerator(client), LLMReflector()

    calibration_results, calibration_metrics = evaluate(judge, calibration)
    feedback = []
    for ex, result in zip(calibration, calibration_results):
        human_scores = {name: getattr(ex.human, name) for name in result.scores()}
        if human_scores != result.scores():
            feedback.append({'id': ex.id, 'human': asdict(ex.human), 'judge': asdict(result),
                             'example': {'user': asdict(ex.user), 'item': asdict(ex.item), 'explanation': ex.explanation},
                             'analysis': MetaJudge().analyze_disagreement(ex, result)})
    reflection_input = [f['analysis'] for f in feedback] if args.backend == 'demo' else feedback
    candidate = reflector.update(judge, reflection_input)
    before_results, before = evaluate(judge, validation)
    after_results, after = evaluate(candidate, validation)
    # A single held-out comparison, with no iterative tuning on validation labels.
    promoted = (after['agreement']['macro'] > before['agreement']['macro']
                and all(after['agreement'][k] >= before['agreement'][k] for k in before['agreement'])
                and after['pass_fail']['fp'] <= before['pass_fail']['fp']
                and after['pass_fail']['fn'] <= before['pass_fail']['fn'])
    active = candidate if promoted else judge
    outcome = generate_with_guardrail(calibration[0], generator, active, args.max_retries)
    sample = random_sample(production, args.sample_size, args.seed)
    production_results, _ = evaluate(active, sample)
    drift = check_drift(sample, production_results, args.agreement_threshold, args.max_false_accept_rate)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'backend': args.backend,
        'calibration_ids': [ex.id for ex in calibration],
        'validation_ids': [ex.id for ex in validation],
        'production_ids': [ex.id for ex in sample],
        'calibration': calibration_metrics, 'disagreements': feedback,
        'validation_before': before, 'validation_after': after,
        'candidate_promoted': promoted, 'active_version': fingerprint(active)['version'],
        'generation': asdict(outcome), 'monitoring': asdict(drift),
        'thresholds': {'agreement': args.agreement_threshold, 'false_accept_rate': args.max_false_accept_rate},
        'action': 'Queue disagreements for human review and recalibration; no automatic retraining.'
                  if drift.drifted else 'No configured quality threshold breached in this labeled sample.',
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, obj in [('report', report), ('rubric_v1', fingerprint(judge)),
                      ('rubric_candidate', fingerprint(candidate)), ('rubric_active', fingerprint(active))]:
        (output / f'{name}.json').write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    with (output / 'evaluations.jsonl').open('w', encoding='utf-8') as f:
        for stage, examples, results, evaluator in [
            ('calibration', calibration, calibration_results, judge),
            ('validation_before', validation, before_results, judge),
            ('validation_after', validation, after_results, candidate),
            ('production', sample, production_results, active)]:
            for ex, result in zip(examples, results):
                f.write(json.dumps({'stage': stage, 'id': ex.id,
                                    'version': fingerprint(evaluator)['version'], 'result': asdict(result)}) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Artifacts: {output.resolve()}')
    return 2 if args.fail_on_drift and drift.drifted else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['demo', 'llm'], default='demo')
    parser.add_argument('--model', default=os.getenv('LLM_MODEL'))
    parser.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'data/benchmark.jsonl')
    parser.add_argument('--validation', type=Path, default=ROOT / 'data/validation.jsonl')
    parser.add_argument('--production', type=Path, default=ROOT / 'data/production_sample.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--sample-size', type=int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--agreement-threshold', type=float, default=0.85)
    parser.add_argument('--max-false-accept-rate', type=float, default=0.10)
    parser.add_argument('--fail-on-drift', action='store_true')
    args = parser.parse_args()
    if args.max_retries < 0 or args.sample_size < 1 or args.timeout <= 0:
        parser.error('retry budget must be nonnegative; sample size and timeout must be positive')
    if not 0 <= args.agreement_threshold <= 1 or not 0 <= args.max_false_accept_rate <= 1:
        parser.error('thresholds must be between 0 and 1')
    try:
        return run(args)
    except (ValueError, OSError, BackendError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
