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


def test_demo_offline_reports_grep_matches(capsys):
    """The demo notes mention "Protocol", so grep must report a real match."""
    assert main(["demo", "--offline"]) == 0

    output = capsys.readouterr().out
    assert "grep found 0 matches" not in output
    assert "grep found 1 matches" in output


def test_demo_offline_previews_file_content(capsys):
    """The preview must show the note's first line, not the raw response dict."""
    assert main(["demo", "--offline"]) == 0

    output = capsys.readouterr().out
    assert "first line: # MCP research notes" in output
    assert "file_path" not in output
