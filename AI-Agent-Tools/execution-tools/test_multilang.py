"""Multi-language code execution tests.

Previously a manual `python test_multilang.py` script whose helper
`test_language(executor, language, code, description)` was collected by pytest
as a test and errored on its unresolvable arguments. It is now a parametrized
suite that skips any language whose toolchain is not installed.
"""

import shutil

import pytest

from multilang_executor import LanguageExecutor, ExecutionStatus


# (language, required executable, code, expected substring in stdout)
LANGUAGE_CASES = [
    (
        "python",
        "python3",
        "print('Sum:', sum(range(1, 11)))",
        "Sum: 55",
    ),
    (
        "javascript",
        "node",
        "console.log('Sum:', [1,2,3,4,5].reduce((a, b) => a + b, 0));",
        "Sum: 15",
    ),
    (
        "typescript",
        "tsx",
        "const point: { x: number } = { x: 10 };\nconsole.log(`X: ${point.x}`);",
        "X: 10",
    ),
    (
        "go",
        "go",
        'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("Sum:", 55)\n}\n',
        "Sum: 55",
    ),
    (
        "java",
        "javac",
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        '        System.out.println("Sum: " + 55);\n'
        "    }\n"
        "}\n",
        "Sum: 55",
    ),
    (
        "cpp",
        "g++",
        "#include <iostream>\n"
        "int main() {\n"
        '    std::cout << "Sum: " << 55 << std::endl;\n'
        "    return 0;\n"
        "}\n",
        "Sum: 55",
    ),
    (
        "rust",
        "rustc",
        'fn main() {\n    println!("Sum: {}", 55);\n}\n',
        "Sum: 55",
    ),
    (
        "php",
        "php",
        '<?php\necho "Sum: " . array_sum([1,2,3,4,5]) . "\\n";\n',
        "Sum: 15",
    ),
    (
        "bash",
        "bash",
        'sum=0\nfor i in {1..10}; do sum=$((sum + i)); done\necho "Sum: $sum"\n',
        "Sum: 55",
    ),
]


@pytest.mark.parametrize(
    "language,executable,code,expected",
    LANGUAGE_CASES,
    ids=[case[0] for case in LANGUAGE_CASES],
)
async def test_language_executes_and_reports_stdout(language, executable, code, expected):
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} is not installed")

    executor = LanguageExecutor()
    result = await executor.execute_code(code, language, timeout=60.0, compile_timeout=60.0)

    assert result["status"] == ExecutionStatus.SUCCESS, (
        f"{language} failed: {result.get('stderr') or result.get('error')}"
    )
    assert expected in result["stdout"]
    assert result["language"] == language


async def test_unsupported_language_is_rejected():
    executor = LanguageExecutor()
    result = await executor.execute_code("noop", "brainfuck")

    assert result["status"] == ExecutionStatus.ERROR
    assert "Unsupported language" in result["error"]
