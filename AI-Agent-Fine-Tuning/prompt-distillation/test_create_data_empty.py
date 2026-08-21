import asyncio

from create_data import generate_distillation_data


def test_generate_empty_input_file(tmp_path):
    # Empty input file (0 sentences): should return cleanly without loading a model
    # and without raising IndexError/ZeroDivisionError
    input_file = tmp_path / "empty.txt"
    input_file.write_text("", encoding="utf-8")
    output_file = tmp_path / "out.jsonl"

    result = asyncio.run(
        generate_distillation_data(
            input_file=str(input_file),
            output_file=str(output_file),
            model_name="stub",
        )
    )
    assert result is None
    assert not output_file.exists()


def test_generate_blank_lines_only(tmp_path):
    # A file containing only blank lines counts as empty too
    input_file = tmp_path / "blank.txt"
    input_file.write_text("\n   \n\t\n", encoding="utf-8")
    output_file = tmp_path / "out.jsonl"

    result = asyncio.run(
        generate_distillation_data(
            input_file=str(input_file),
            output_file=str(output_file),
            model_name="stub",
        )
    )
    assert result is None
    assert not output_file.exists()
