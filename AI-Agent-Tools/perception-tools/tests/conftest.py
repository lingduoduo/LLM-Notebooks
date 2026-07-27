"""Shared pytest configuration."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip explicitly live tests unless the user opts in."""
    if os.getenv("RUN_LIVE_PERCEPTION_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_LIVE_PERCEPTION_TESTS=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
