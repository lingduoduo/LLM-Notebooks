#!/usr/bin/env python3
"""Run offline knowledge-example suites in separate Python processes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# These are manual service exercises, not offline unit tests. One launches ML models.
LIVE_SCRIPTS = {"test_client.py", "test_pipeline.py", "test_improvements.py"}
FOLDERS = {
    "shared": ".",
    "dense-embedding": "dense-embedding",
    "sparse-embedding": "sparse-embedding",
    "retrieval-pipeline": "retrieval-pipeline",
    "contextual-retrieval": "contextual-retrieval",
    "structured-index": "structured-index",
    "user-memory": "user-memory",
    "ontology-use-case": "ontology-use-case/adult_ontology_v1",
}


def suite_commands(python):
    commands = {}
    for name, folder in FOLDERS.items():
        directory = ROOT / folder
        files = (
            ["tests"]
            if name == "shared"
            else [
                path.name
                for path in sorted(directory.glob("test_*.py"))
                if path.name not in LIVE_SCRIPTS
            ]
        )
        commands[name] = (
            directory,
            [python, "-m", "pytest", "-q", "--rootdir", str(directory), *files],
        )
    return commands


def check_environment(source):
    environment = dict(source)
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        RUN_LIVE_SMOKE_TESTS="0",
        STRUCTURED_INDEX_E2E="",
    )
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=FOLDERS,
        help="Run selected suites (repeatable); defaults to all",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter with pytest and the selected examples' dependencies",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List suites and commands without running them",
    )
    args = parser.parse_args(argv)
    commands = suite_commands(args.python)
    selected = args.suite or list(commands)
    failed = []
    for name in selected:
        directory, command = commands[name]
        print(f'\n[{name}] cwd={directory}\n  {" ".join(command)}', flush=True)
        if args.list:
            continue
        try:
            code = subprocess.run(
                command, cwd=directory, env=check_environment(os.environ), check=False
            ).returncode
        except OSError as error:
            print(error, file=sys.stderr)
            code = 1
        if code:
            failed.append(name)
    if failed:
        print("\nFailed suites: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
