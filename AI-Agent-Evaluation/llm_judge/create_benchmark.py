"""Prepare expert/LLM candidates and assemble human-reviewed benchmarks."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from benchmark_authoring import augment, build_benchmark, import_experts, rating_queue, split_groups, split_training_groups, task_hash
from llm_backend import BackendError, ChatClient
from quality import TaskSpec, digest, read_records


def write_jsonl(path, records):
    with path.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Import expert cases and optionally generate gray-area candidates')
    build = commands.add_parser('build', help='Resolve ratings and export reviewed examples')
    for command in (prepare, build):
        command.add_argument('--task', type=Path, required=True)
        command.add_argument('--output-dir', type=Path, required=True)
    prepare.add_argument('--experts', type=Path, required=True)
    prepare.add_argument('--augment-count', type=int, default=0, help='LLM variants per expert seed; 0 makes no model calls')
    prepare.add_argument('--model', default=os.getenv('GENERATOR_MODEL'))
    prepare.add_argument('--base-url', default=os.getenv('GENERATOR_BASE_URL', 'https://api.openai.com/v1'))
    prepare.add_argument('--timeout', type=float, default=60)
    build.add_argument('--candidates', type=Path, required=True)
    build.add_argument('--ratings', type=Path, nargs='+', required=True)
    build.add_argument('--adjudications', type=Path)
    build.add_argument('--min-raters', type=int, default=2)
    build.add_argument('--validation-fraction', type=float, default=0, help='0 disables splitting; otherwise split by family')
    build.add_argument('--holdout-fraction', type=float, default=0, help='Reserve a third split for iterative training')
    build.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError('Use a new or empty output directory')
        task = TaskSpec(**json.loads(args.task.read_text(encoding='utf-8')))
        if args.command == 'prepare':
            if not 0 <= args.augment_count <= 20:
                raise ValueError('augment-count must be 0..20')
            candidates = import_experts(list(read_records(args.experts)), task)
            if args.augment_count:
                client = ChatClient(args.model, args.base_url,
                                    os.getenv('GENERATOR_API_KEY', os.getenv('OPENAI_API_KEY', '')), args.timeout)
                candidates += augment(candidates, task, client, args.augment_count)
            queue = rating_queue(candidates, task)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(args.output_dir / 'candidates.jsonl', candidates)
            write_jsonl(args.output_dir / 'rating_queue.jsonl', queue)
            (args.output_dir / 'task.json').write_text(json.dumps(asdict(task), indent=2) + '\n')
            print(f'Prepared {len(candidates)} unlabeled candidates. Human ratings are required.')
            return 0
        candidates = list(read_records(args.candidates))
        ratings = [r for path in args.ratings for r in read_records(path)]
        adjudications = list(read_records(args.adjudications)) if args.adjudications else []
        accepted, pending = build_benchmark(candidates, ratings, task, args.min_raters, adjudications)
        splits = None
        if args.holdout_fraction:
            splits = split_training_groups(accepted, args.validation_fraction, args.holdout_fraction, args.seed)
        elif args.validation_fraction:
            splits = split_groups(accepted, args.validation_fraction, args.seed)
        report = {'version': digest({'task': asdict(task), 'records': sorted(accepted, key=lambda r: r['id'])}),
                  'task_version': task_hash(task), 'candidate_count': len(candidates), 'accepted_count': len(accepted),
                  'pending_count': len(pending), 'min_raters': args.min_raters,
                  'human_pass_count': sum(r['human_decision'] for r in accepted),
                  'human_fail_count': sum(not r['human_decision'] for r in accepted),
                  'sources': {source: sum(r['metadata']['provenance']['source'] == source for r in accepted)
                              for source in ('expert', 'llm')},
                  'validation_fraction': args.validation_fraction, 'holdout_fraction': args.holdout_fraction, 'seed': args.seed}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / 'benchmark.jsonl', accepted)
        write_jsonl(args.output_dir / 'pending.jsonl', pending)
        if splits:
            write_jsonl(args.output_dir / 'calibration.jsonl', splits[0])
            if len(splits) == 3:
                write_jsonl(args.output_dir / 'development.jsonl', splits[1])
                write_jsonl(args.output_dir / 'holdout.jsonl', splits[2])
            else:
                write_jsonl(args.output_dir / 'validation.jsonl', splits[1])
        (args.output_dir / 'task.json').write_text(json.dumps(asdict(task), indent=2) + '\n')
        (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 2 if pending else 0
    except (ValueError, TypeError, KeyError, OSError, BackendError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
