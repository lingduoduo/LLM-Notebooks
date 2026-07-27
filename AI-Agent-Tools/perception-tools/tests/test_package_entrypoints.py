from importlib import import_module
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_package_imports_without_sys_path_mutation():
    package = import_module("perception_tools")
    assert package.__name__ == "perception_tools"
    import_module("perception_tools.cli")
    server = import_module("perception_tools.server")
    assert callable(server.main)


def test_console_entry_points_are_declared():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts["perception-tools"] == "perception_tools.cli:main"
    assert scripts["perception-tools-mcp"] == "perception_tools.server:main"
