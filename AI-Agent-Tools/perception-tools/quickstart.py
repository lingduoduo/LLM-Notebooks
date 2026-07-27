"""Compatibility launcher for the packaged offline quick-start demo."""

from perception_tools.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["demo", "--offline"]))
