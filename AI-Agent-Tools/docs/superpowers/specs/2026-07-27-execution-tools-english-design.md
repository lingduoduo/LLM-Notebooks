# Execution Tools English Conversion Design

## Goal

Convert all maintained user-facing and developer-facing Chinese text in
`execution-tools` to clear English while preserving the package's commands,
interfaces, and runtime behavior.

## Scope

The conversion covers:

- CLI descriptions, option help, errors, headings, and demo output
- Runtime messages returned by execution tools
- Python module, class, and function docstrings
- Inline and block comments
- Embedded demonstration content
- References to Chinese prose or chapter headings
- Tests or fixtures whose maintained text is Chinese

Already-English text may receive small consistency edits only when needed to
make the translated result read naturally. The already-English
`perception-tools` package is outside this change.

## Compatibility

The conversion will preserve:

- Python identifiers and import paths
- CLI command and option names
- Environment-variable names
- JSON keys and result structures
- External APIs and integration behavior
- File paths and configuration semantics
- Existing safety checks and execution behavior

English runtime strings will intentionally replace Chinese runtime strings.
Tests that assert those messages will be updated to assert the English text.

## Implementation

Translate the Chinese text in `execution-tools/cli.py`,
`execution-tools/execution_tools.py`, and any other maintained source,
documentation, configuration, or test file found by a repository-wide Unicode
scan. Use concise technical English and consistent terminology:

- "execution tools" for 执行工具
- "workspace" for 工作区 or 工作目录
- "validation" for 校验
- "approval" for 审批
- "truncation and persistence" for 截断与持久化
- "offline demo" for 离线演示

No unrelated refactoring or packaging changes are included.

## Validation

Add an automated English-only test for maintained `execution-tools` text files.
The test will scan relevant source, test, documentation, and configuration
files for Han characters while excluding generated artifacts, caches, virtual
environments, and Git metadata.

Validation will include:

1. Running the English-only scan.
2. Running the full `execution-tools` test suite.
3. Exercising CLI help, tool listing, and the offline demo to confirm that
   visible output is English and behavior remains intact.

## Success Criteria

- No Han characters remain in maintained `execution-tools` files.
- All user-visible CLI and runtime text is English.
- All comments and docstrings are English.
- Commands, APIs, structured result fields, and behavior remain compatible.
- The full non-live test suite passes.
