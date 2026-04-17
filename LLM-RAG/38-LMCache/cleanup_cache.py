#!/usr/bin/env python3
"""
Low-impact maintenance script for cleaning expired LMCache L2 files.
"""

import argparse
import json
import sys
from pathlib import Path

from test_lmcache import OptimizedDiskL2Cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean expired LMCache L2 files")
    parser.add_argument("--root", default="./l2_cache", help="L2 cache root directory")
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=24.0,
        help="Delete cache files older than this many hours",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.8,
        help="Warn when filesystem usage exceeds this ratio",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    cache = OptimizedDiskL2Cache(root_dir=str(root))
    try:
        before_files = cache.get_file_count()
        before_size_mb = cache.get_size_mb()
        removed = cache.cleanup_expired(int(args.older_than_hours * 3600))
        after_files = cache.get_file_count()
        after_size_mb = cache.get_size_mb()
        disk = cache.get_disk_usage()
    finally:
        cache.close()

    summary = {
        "root": str(root),
        "removed_files": removed,
        "before_files": before_files,
        "after_files": after_files,
        "before_size_mb": round(before_size_mb, 2),
        "after_size_mb": round(after_size_mb, 2),
        "disk_used_ratio": round(disk["disk_used_ratio"], 4),
        "disk_used_gb": round(disk["disk_used_gb"], 2),
        "disk_free_gb": round(disk["disk_free_gb"], 2),
        "warning": disk["disk_used_ratio"] >= args.warn_threshold,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            "LMCache cleanup: removed={removed_files}, files={after_files}, "
            "size={after_size_mb}MB, disk_used={disk_used_ratio:.1%}".format(**summary)
        )
        if summary["warning"]:
            print(
                "WARNING: disk usage is above threshold "
                f"({summary['disk_used_ratio']:.1%} >= {args.warn_threshold:.0%})"
            )

    return 2 if summary["warning"] else 0


if __name__ == "__main__":
    sys.exit(main())
