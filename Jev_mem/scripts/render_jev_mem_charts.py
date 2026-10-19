#!/usr/bin/env python3
"""Render a report from a standalone run's actual summary (no historical snapshot)."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Jev_mem.pipeline.report import render_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="Run directory containing summary.json"
    )
    args = parser.parse_args(argv)
    try:
        print(render_report(args.run))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Cannot render report: {exc}\n")


if __name__ == "__main__":
    main()
