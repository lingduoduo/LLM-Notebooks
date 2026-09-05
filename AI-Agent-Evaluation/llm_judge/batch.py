"""Evaluate production JSONL using a reusable task and a dedicated judge model."""
import argparse
import json
import os
from pathlib import Path

from llm_backend import BackendError, ChatClient
from quality import TaskSpec, TaskJudge, align, digest, evaluate_stream, read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--judge-model', default=os.getenv('JUDGE_MODEL'))
    parser.add_argument('--base-url', default=os.getenv('JUDGE_BASE_URL', 'https://api.openai.com/v1'))
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--review-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--agreement-threshold', type=float, default=0.85)
    parser.add_argument('--max-false-accept-rate', type=float, default=0.10)
    parser.add_argument('--fail-on-alert', action='store_true')
    args = parser.parse_args()
    if bool(args.calibration) != bool(args.validation):
        parser.error('--calibration and --validation must be supplied together')
    if not 1 <= args.workers <= 64 or args.review_size < 1:
        parser.error('--workers must be 1..64 and --review-size must be positive')
    if not 0 <= args.agreement_threshold <= 1 or not 0 <= args.max_false_accept_rate <= 1:
        parser.error('thresholds must be between 0 and 1')
    try:
        # Refuse to overwrite prior results before making any billable calls.
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError('Use a new or empty output directory')
        task = TaskSpec(**json.loads(args.task.read_text(encoding='utf-8')))
        client = ChatClient(args.judge_model, args.base_url,
                            os.getenv('JUDGE_API_KEY', os.getenv('OPENAI_API_KEY', '')), args.timeout)
        judge = TaskJudge(client, task)
        excluded_ids, excluded_content = set(), set()
        if args.calibration:
            calibration, validation = list(read_records(args.calibration)), list(read_records(args.validation))
            judge, alignment = align(judge, calibration, validation)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / 'alignment.json').write_text(json.dumps(alignment, indent=2) + '\n')
            for record in calibration + validation:
                excluded_ids.add(record['id'])
                excluded_content.add(digest({'context': record['context'], 'output': record['output']}))
        def production():
            for record in read_records(args.input):
                if isinstance(record, dict) and (record.get('id') in excluded_ids or
                        digest({'context': record.get('context'), 'output': record.get('output')}) in excluded_content):
                    raise ValueError('Production data overlaps an alignment split')
                yield record
        report = evaluate_stream(production(), judge, args.output_dir, args.workers, args.review_size,
                                 args.seed, args.agreement_threshold, args.max_false_accept_rate)
        print(json.dumps(report, indent=2))
        return 2 if args.fail_on_alert and report['alerts'] else 0
    except (OSError, ValueError, TypeError, BackendError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
