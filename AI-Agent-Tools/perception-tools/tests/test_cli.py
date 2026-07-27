import pytest

from perception_tools.cli import main


def test_cli_help_is_english(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Perception Tools MCP Server" in output
    assert "Search" in output


def test_cli_list_is_english(capsys):
    assert main(["list", "--category", "filesystem"]) == 0

    output = capsys.readouterr().out
    assert "File System" in output
    assert "tools" in output
