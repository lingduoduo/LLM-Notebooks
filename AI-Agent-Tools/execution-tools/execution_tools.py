"""Generic execution tools: code interpreter and virtual terminal."""

import subprocess
import traceback
from typing import Dict, Any, Optional, Tuple

import output_store
from llm_helper import LLMHelper
from config import Config
from multilang_executor import LanguageExecutor, ExecutionStatus, normalize_language
from safety import detect_dangerous_code, detect_dangerous_command

# Long-output handling thresholds (see "Long-output truncation and
# persistence" in chapter 4).
# When output exceeds either threshold, keep the head and tail few lines in the
# context and persist the full output to a temp file for later retrieval.
MAX_OUTPUT_LINES = 200
MAX_OUTPUT_CHARS = 10000
HEAD_LINES = 50
TAIL_LINES = 50


def truncate_and_persist(
    text: str,
    tool_name: str = "execution",
    max_lines: int = MAX_OUTPUT_LINES,
    max_chars: int = MAX_OUTPUT_CHARS,
    head_lines: int = HEAD_LINES,
    tail_lines: int = TAIL_LINES,
) -> Tuple[str, Optional[str]]:
    """Truncate over-long output and persist the full text to a temp file.

    Returns a tuple of (processed_text, saved_path). When the output is within
    both thresholds, it is returned unchanged with ``saved_path`` set to None.
    Otherwise only the first ``head_lines`` and last ``tail_lines`` lines are
    kept in context, with a middle marker pointing to the saved file. This
    keeps the agent's context bounded without discarding any information and
    requires no LLM call.
    """
    if text is None:
        return text, None

    lines = text.split("\n")
    if len(text) <= max_chars and len(lines) <= max_lines:
        return text, None

    # Persist the complete output for later retrieval via read_file. The store
    # owns the file's lifetime so a long-running server cannot leak them.
    path = output_store.save(text, tool_name)

    # lines[-0:] is the whole list in Python; treat 0 as "keep no tail".
    head_n = max(0, head_lines)
    tail_n = max(0, tail_lines)
    head_part = lines[:head_n] if head_n else []
    tail_part = lines[-tail_n:] if tail_n else []
    omitted = max(len(lines) - head_n - tail_n, 0)

    guide = f"[To read the complete output, use the fs_read_file tool on {path}]"
    if omitted == 0:
        # Head+tail cover the file; do not concatenate overlapping slices.
        truncated = "\n".join(lines + [guide])
    else:
        middle = f"... [{omitted} lines omitted; complete output saved to {path}] ..."
        truncated = "\n".join(head_part + [middle] + tail_part + [guide])
    return truncated, path


def prepare_output(
    llm_helper: LLMHelper,
    tool_name: str,
    text: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Reduce one output stream to something context-sized.

    Returns ``(context_text, saved_path)``. The complete text is always
    persisted when it exceeds a threshold, so nothing is lost either way.

    Summarization is evaluated against the *full* text. Previously truncation
    ran first and the summarizer was then asked whether the already-trimmed
    ~100 lines were too long, which they almost never were -- so the summary
    path was effectively dead code.
    """
    if not text:
        return text, None

    truncated, saved_path = truncate_and_persist(text, tool_name)
    if saved_path is None:
        return truncated, None

    if Config.AUTO_SUMMARIZE_COMPLEX_OUTPUT and len(text) > MAX_OUTPUT_CHARS:
        summary = llm_helper.summarize_output(tool_name, text)
        return f"{summary}\n\n[Complete output saved to {saved_path}]", saved_path

    return truncated, saved_path


class ExecutionTools:
    """Generic execution tools with safety and result analysis."""
    
    def __init__(self, llm_helper: LLMHelper):
        """Initialize execution tools with LLM helper."""
        self.llm_helper = llm_helper
        self.lang_executor = LanguageExecutor(workspace_dir=Config.WORKSPACE_DIR)

    def _analyze_failure(
        self,
        tool_name: str,
        command: str,
        success: bool,
        stderr: Optional[str],
        error: Optional[str],
    ) -> Optional[str]:
        """Return an LLM explanation of a failure, or None when not applicable.

        Only runs for actual failures that produced diagnostic text, so a
        successful call never pays for an LLM round-trip.
        """
        if success or not Config.AUTO_ANALYZE_ERRORS:
            return None

        diagnostics = "\n".join(part for part in (stderr, error) if part).strip()
        if not diagnostics:
            return None

        return self.llm_helper.analyze_error(tool_name, command, diagnostics)
    
    async def code_interpreter(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
        stdin: Optional[str] = None,
        files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment with multi-language support.
        
        Args:
            code: Code to execute
            language: Programming language (python, javascript, typescript, go, java, cpp, rust, php, bash)
            timeout: Execution timeout in seconds
            stdin: Optional stdin input
            files: Optional additional files
            
        Returns:
            Result dictionary with output and analysis
        """
        # Resolve aliases before any safety check. Keying the checks on the
        # raw name let `python3` reach an empty pattern table and skip both
        # the approval gate and local syntax verification.
        language = normalize_language(language)

        verification = "skipped"
        if Config.AUTO_VERIFY_CODE:
            status, error_msg = self.llm_helper.verify_code_syntax(code, language)
            if status == "invalid":
                return {
                    "success": False,
                    "error": f"Syntax error: {error_msg}",
                    "verification": "failed",
                    "language": language
                }
            # "unverified" still runs: the interpreter or compiler invoked
            # below is the authoritative syntax check for every language other
            # than Python, and blocking here would make offline multi-language
            # execution impossible.
            verification = "passed" if status == "valid" else "unverified"

        # Check for dangerous operations
        if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
            detected = detect_dangerous_code(code, language)

            if detected:
                approved, reason = self.llm_helper.request_approval(
                    "code_execution",
                    {
                        "code": code,
                        "language": language,
                        "detected_patterns": detected
                    }
                )

                if not approved:
                    return {
                        "success": False,
                        "error": f"Execution not approved: {reason}",
                        "language": language
                    }
        
        # Execute code using multi-language executor
        try:
            result = await self.lang_executor.execute_code(
                code=code,
                language=language,
                timeout=timeout,
                stdin=stdin,
                files=files
            )
            
            # Convert status to success flag
            success = result.get('status') == ExecutionStatus.SUCCESS
            
            # Long outputs: truncate head/tail and persist the full text to a
            # temp file (offline-safe), then optionally LLM-summarize whatever
            # still exceeds the char threshold.
            raw_stderr = result.get('stderr', '')
            stdout, stdout_file = prepare_output(
                self.llm_helper, "code_interpreter", result.get('stdout', '')
            )
            stderr, stderr_file = prepare_output(
                self.llm_helper, "code_interpreter", raw_stderr
            )

            response = {
                "success": success,
                "status": result.get('status'),
                "language": result.get('language', language),
                "stdout": stdout,
                "stderr": stderr,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "returncode": result.get('returncode'),
                "error": result.get('error'),
                "compile_output": result.get('compile_output'),
                "phase": result.get('phase'),
                "verification": verification
            }

            analysis = self._analyze_failure(
                "code_interpreter", code, success, raw_stderr, result.get('error')
            )
            if analysis is not None:
                response["error_analysis"] = analysis

            return response
            
        except Exception as e:
            error_output = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return {
                "success": False,
                "error": error_output,
                "language": language
            }
    
    async def virtual_terminal(
        self,
        command: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute shell command in a virtual terminal.
        
        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            
        Returns:
            Result dictionary with output and analysis
        """
        # Check for dangerous commands
        if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
            detected = detect_dangerous_command(command)

            if detected:
                approved, reason = self.llm_helper.request_approval(
                    "terminal_command",
                    {
                        "command": command,
                        "detected_patterns": detected
                    }
                )

                if not approved:
                    return {
                        "success": False,
                        "error": f"Command execution not approved: {reason}"
                    }
        
        # Execute command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Config.WORKSPACE_DIR
            )
            
            # Long output: summarize or trim for context, persist in full.
            raw_stderr = result.stderr
            stdout, stdout_file = prepare_output(
                self.llm_helper, "virtual_terminal", result.stdout
            )
            stderr, stderr_file = prepare_output(
                self.llm_helper, "virtual_terminal", raw_stderr
            )

            response = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file
            }

            analysis = self._analyze_failure(
                "virtual_terminal", command, response["success"], raw_stderr, None
            )
            if analysis is not None:
                response["error_analysis"] = analysis

            return response
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}"
            }
