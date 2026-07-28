# Experiment 4.3: Execution Tools MCP Server

## Objective

Implement a comprehensive MCP server that provides execution tools with built-in safety mechanisms, demonstrating real-world best practices for AI agent tool execution.

## Experiment Overview

This experiment explores three critical aspects of execution tools:

1. **Safety Mechanisms**: LLM-based approval for dangerous operations
2. **Result Processing**: Automatic summarization of complex outputs
3. **Verification**: Automatic validation of tool execution results

## Architecture

### Safety Layer

The safety layer implements a multi-level protection system:

**LLM-Based Approval**: Before executing irreversible operations (file overwrite, file deletion, system commands, external API calls), the system consults a secondary LLM to evaluate the risk. The approval process analyzes the operation for potential data loss, security risks, and resource consumption concerns. This mirrors real-world approval workflows where critical operations require managerial sign-off or risk control review.

Two details matter for the gate to mean anything. First, detection is structural rather than textual: shell commands are tokenized so the command word and its flags are examined (`rm -fr` and `rm -r -f` are caught, `git rm --cached` is not), and code is matched with word-boundary patterns keyed on a canonical language name so that an alias like `python3` cannot land on an empty rule table. Second, the reviewed operation is passed to the reviewer inside a delimited untrusted-content block with an explicit instruction to ignore directives found inside it; interpolating the reviewed code straight into the prompt lets the code address its own reviewer.

**Result Summarization**: When a tool produces output exceeding 10,000 characters, the system invokes an LLM to distill the essential information from the *complete* text, and writes that complete text to a file whose path is returned. Ordering matters here: summarizing after truncation means the summarizer only ever sees the already-trimmed head and tail, which is both useless and, since the trimmed text is under the threshold, effectively unreachable. Outputs under the threshold are returned as-is.

**Automatic Verification**: Python is validated locally with `compile()`, which is authoritative. Other languages are validated by their own compiler or interpreter when the code runs, which is also authoritative and needs no LLM. Verification reports `passed`, `failed`, `unverified` or `skipped`. The `unverified` state exists deliberately: a verifier that cannot run must not report success, and answering "valid" whenever the check raised turned an unreachable verifier into a silent clean bill of health.

### Tool Implementation

#### File System Tools

The file system tools provide safe, verified file operations. The write operation supports automatic syntax checking for code files in Python, JavaScript, and TypeScript, preventing the creation of invalid source files. It refuses to replace an existing file unless `overwrite=true` is passed, and asks for approval when it does — a flag that only triggers a review, with no branch that actually refuses, is not a safety control. The edit operation generates diff previews before applying changes. Both enforce workspace boundaries, preventing file access outside designated directories.

The enhanced filesystem tools extend this to reading, searching, inspecting, moving, copying, deleting and creating, with approval required for the destructive operations.

#### Generic Execution Tools

The code interpreter executes code in nine languages, each in a fresh temporary directory as a subprocess. It captures both standard output and error streams, detects dangerous constructs before running, and provides error analysis when execution fails. Auxiliary files are written as UTF-8 text unless they carry an explicit `base64:` prefix; inferring binary content from the character set silently corrupts ordinary words that happen to be valid base64. The virtual terminal executes shell commands with configurable timeouts and the same approval gate. A separate stateful terminal session keeps a working directory and command history across calls.

**Isolation caveat**: "temporary directory plus subprocess" is containment, not a security sandbox. There are no resource limits, no network restrictions and no privilege separation, so untrusted code should be run through the provided `Dockerfile` rather than directly on a workstation.

#### External Integration Tools

The Google Calendar integration adds events with validation of datetime formats and logical consistency checks. The GitHub integration creates pull requests with branch verification and approval workflows. Both tools demonstrate patterns for safely interacting with external systems while maintaining visibility and control.

## Setup

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy environment template:
```bash
cp env.example .env
```

2. Configure the OpenAI API key:

```bash
OPENAI_API_KEY=your-key
```

The LLM steps call the OpenAI API directly. `MODEL` is optional and defaults to
`gpt-5.6`; set it to any OpenAI model id to override.

3. (Optional) Configure external services:
```bash
# Google Calendar
GOOGLE_CALENDAR_CREDENTIALS_FILE=credentials.json

# GitHub
GITHUB_TOKEN=ghp_...
```

### Safety Settings

```bash
# Enable/disable safety features
REQUIRE_APPROVAL_FOR_DANGEROUS_OPS=true
AUTO_SUMMARIZE_COMPLEX_OUTPUT=true
AUTO_VERIFY_CODE=true
AUTO_ANALYZE_ERRORS=true
MAX_OUTPUT_LENGTH=1000
```

## Running the Experiment

### Quick Start

```bash
python quickstart.py
```

This demonstrates all major features with minimal setup.

### Test Suite

```bash
# Everything (offline, isolated temporary workspace)
python -m pytest

# One area at a time
python -m pytest test_file_tools.py
python -m pytest test_execution_tools.py
python -m pytest test_safety_layer.py
python -m pytest test_tool_registry.py
```

Language cases whose toolchain is not installed are skipped rather than failed.

### Comprehensive Examples

```bash
python examples.py
```

### Running as MCP Server

```bash
python server.py
```

The server will start in stdio mode, ready to accept MCP protocol connections.

## Experiment Results

### Safety Mechanism Evaluation

Test the approval system by attempting dangerous operations:

1. File overwrite of important files
2. Terminal commands with destructive patterns
3. Code execution with system calls

Observe how the LLM evaluates risk and makes approval decisions.

### Summarization Effectiveness

Generate complex outputs and measure summarization quality:

1. Execute commands that produce verbose output (>10,000 characters)
2. Run code that generates extensive logs
3. Verify that outputs under 10,000 characters are returned unchanged
4. Compare original vs. summarized information density for large outputs

### Verification Accuracy

Test automatic verification across different scenarios:

1. Valid code with correct syntax
2. Code with syntax errors
3. Code with runtime errors
4. Terminal commands that succeed/fail

## Key Observations

### Safety Trade-offs

The approval mechanism introduces latency as each dangerous operation requires an additional LLM call. However, this overhead prevents catastrophic failures and provides audit trails for critical actions. The system can be tuned by adjusting `REQUIRE_APPROVAL_FOR_DANGEROUS_OPS` based on trust level and use case requirements.

### Summarization Benefits

Automatic summarization significantly reduces token consumption when dealing with verbose tool outputs exceeding 10,000 characters. The LLM effectively extracts actionable information while preserving critical details. For terminal errors spanning hundreds of lines, summarization typically captures the root cause in a concise format. Outputs under the threshold are returned as-is, ensuring no information loss for moderately-sized results.

### Verification Limitations

While syntax verification catches many issues before execution, it cannot predict runtime failures or logical errors. The system works best when combined with error analysis that provides suggestions for fixing failed operations. For Python, compile-time syntax checking is highly accurate. For other languages the real compiler or interpreter is the authoritative check, so no pre-flight approximation is attempted at execution time; a written source file that is never executed has no such fallback and is reported as `unverified`.

## Discussion Questions

1. How does LLM-based approval compare to rule-based safety checks?
2. What are the trade-offs between automation and human oversight?
3. How can verification be extended to more complex validation scenarios?
4. What metrics should be used to evaluate summarization quality?
5. How should the system handle edge cases where approval is needed but the LLM is unavailable?

## Extensions

### Suggested Improvements

1. **Caching**: Cache approval decisions for identical operations
2. **Rollback**: Implement undo functionality for file operations
3. **Sandboxing**: Use containers for true code isolation
4. **Multi-step Planning**: Break complex operations into verified steps
5. **Learning**: Train models on historical approval patterns

### Additional Tools

Consider implementing:
- Database query tools with schema validation
- API calling tools with rate limiting
- File backup/restore functionality
- Distributed execution across multiple machines

## Conclusion

This experiment demonstrates that production-ready execution tools require multiple layers of safety, verification, and result processing. The combination of LLM-based approval, automatic summarization, and verification creates a robust system suitable for real-world autonomous agent deployments. The architecture patterns shown here can be adapted to virtually any tool category where safety and reliability are paramount.
