#!/usr/bin/env python3
"""Build a single-file offline interactive view from the current taxonomy."""
import argparse
import json
from pathlib import Path

from taxonomy import validate_taxonomy

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "taxonomy_v1.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "visuals/taxonomy_explorer.html"
    )
    args = parser.parse_args(argv)
    data = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    validate_taxonomy(data)
    template = (ROOT / "visuals/explorer.template.html").read_text(encoding="utf-8")
    # Embedded JSON cannot terminate its script element; labels render via textContent.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        template.replace("__TAXONOMY_JSON__", payload), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
