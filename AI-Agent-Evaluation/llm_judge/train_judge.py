"""Train a judge rubric through self-reflection, anchors and measured stopping gates."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from llm_backend import BackendError, ChatClient
from quality import TaskJudge, TaskSpec, read_records
from training import TrainingConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('task', 'calibration', 'development', 'holdout', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--judge-model', default=os.getenv('JUDGE_MODEL'))
    parser.add_argument('--base-url', default=os.getenv('JUDGE_BASE_URL', 'https://api.openai.com/v1'))
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--anchor-id', action='append', dest='anchor_ids')
    defaults = TrainingConfig()
    for name in ('max_rounds', 'patience', 'min_examples', 'min_per_class', 'anchor_count'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(defaults, name))
    for name in ('target_agreement', 'target_score_agreement', 'max_false_accept_rate', 'min_recall'):
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=getattr(defaults, name))
    args = parser.parse_args()
    started = False
    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError('Use a new or empty output directory')
        config = TrainingConfig(**{name: getattr(args, name) for name in asdict(defaults)})
        task = TaskSpec(**json.loads(args.task.read_text(encoding='utf-8')))
        calibration = list(read_records(args.calibration))
        development = list(read_records(args.development))
        holdout = list(read_records(args.holdout))
        client = ChatClient(args.judge_model, args.base_url,
                            os.getenv('JUDGE_API_KEY', os.getenv('OPENAI_API_KEY', '')), args.timeout)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        started = True
        with (args.output_dir / 'iterations.jsonl').open('x', encoding='utf-8') as stream:
            def checkpoint(event):
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                stream.flush()
                selected = event.get('judge') if event['stage'] == 'baseline' else (
                    event.get('candidate') if event.get('accepted') else None)
                if selected:
                    (args.output_dir / 'checkpoint_judge.json').write_text(json.dumps(selected, indent=2) + '\n')
            judge, report = train(TaskJudge(client, task), calibration, development, holdout,
                                  config, args.anchor_ids, checkpoint)
        (args.output_dir / 'active_task.json').write_text(json.dumps(asdict(judge.task), indent=2) + '\n')
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('aligned', 'stop_reason', 'rounds_completed', 'metrics', 'unmet_targets')}, indent=2))
        return 0 if report['aligned'] else 2
    except (ValueError, TypeError, OSError, BackendError) as exc:
        if started:
            (args.output_dir / 'failure.json').write_text(json.dumps({'status': 'error', 'error': str(exc)}, indent=2) + '\n')
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
