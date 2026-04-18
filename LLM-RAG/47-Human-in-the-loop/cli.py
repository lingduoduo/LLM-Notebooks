# cli.py
"""
Command Line Interface for Human-in-the-Loop Agent System.
Provides interactive CLI for testing and running the HITL agent.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from config import (
    SHOW_DETAILED_LOGS,
    TEST_MODE, MOCK_APPROVAL_RESPONSE
)
from logger import get_logger

logger = get_logger(__name__)


class ApprovalDecision(dict):
    """Dict resume payload that also compares equal to its decision type."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.get("type") == other
        return super().__eq__(other)


class CLIInterface:
    """Command Line Interface for HITL Agent interactions."""

    def __init__(self):
        self.colors = {
            'reset': '\033[0m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m'
        }

    def print_header(self, title: str) -> None:
        """Print a formatted header."""
        width = 60
        print(f"\n{self.colors['cyan']}{'=' * width}{self.colors['reset']}")
        print(f"{self.colors['bold']}{title.center(width)}{self.colors['reset']}")
        print(f"{self.colors['cyan']}{'=' * width}{self.colors['reset']}\n")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        print(f"{self.colors['green']}✓ {message}{self.colors['reset']}")

    def print_error(self, message: str) -> None:
        """Print an error message."""
        print(f"{self.colors['red']}✗ {message}{self.colors['reset']}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        print(f"{self.colors['yellow']}⚠ {message}{self.colors['reset']}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        print(f"{self.colors['blue']}ℹ {message}{self.colors['reset']}")

    def print_step(self, step_num: int, message: str) -> None:
        """Print a numbered step."""
        print(f"{self.colors['magenta']}[Step {step_num}]{self.colors['reset']} {message}")

    def get_user_input(
        self,
        prompt: str,
        default: Optional[str] = None,
        timeout: Optional[int] = None,
        validator: Optional[callable] = None
    ) -> str:
        """Get validated user input with timeout support."""
        full_prompt = prompt
        if default:
            full_prompt += f" (default: {default})"
        full_prompt += ": "

        if TEST_MODE and MOCK_APPROVAL_RESPONSE:
            print(f"{full_prompt}{MOCK_APPROVAL_RESPONSE}")
            return MOCK_APPROVAL_RESPONSE

        try:
            if timeout:
                print(f"{full_prompt}(timeout: {timeout}s) ", end="", flush=True)
                # Simple timeout implementation (not perfect but works for CLI)
                import select
                import sys

                ready, _, _ = select.select([sys.stdin], [], [], timeout)
                if ready:
                    response = sys.stdin.readline().strip()
                else:
                    print(f"\nTimeout after {timeout} seconds")
                    return default or ""
            else:
                if prompt.endswith(":"):
                    print(prompt)
                    response = input().strip()
                else:
                    response = input(full_prompt).strip()

            # Use default if empty
            if not response and default:
                response = default

            # Validate if validator provided
            if validator and response:
                try:
                    validator(response)
                except ValueError as e:
                    self.print_error(f"Invalid input: {e}")
                    return self.get_user_input(prompt, default, timeout, validator)

            return response

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except EOFError:
            print("\nEnd of input reached")
            sys.exit(1)

    def display_approval_request(self, approval_data: Dict[str, Any]) -> None:
        """Display purchase approval request details."""
        self.print_header("HUMAN APPROVAL REQUIRED")

        print(f"{self.colors['bold']}Purchase Request Details:{self.colors['reset']}")
        print(f"  Item: {approval_data.get('item', 'Unknown')}")
        print(f"  Price: {approval_data.get('price', 'Unknown')} {approval_data.get('currency', 'CNY')}")
        print(f"  Vendor: {approval_data.get('vendor', 'Unknown')}")
        print(f"  Message: {approval_data.get('message', 'No message provided')}")

        print(f"\n{self.colors['bold']}Available Actions:{self.colors['reset']}")
        print("  1. Accept - Approve the purchase as requested")
        print("  2. Reject - Deny the purchase request")
        print("  3. Edit   - Modify parameters before approving")

    def get_approval_decision(self) -> Dict[str, Any]:
        """Get human approval decision from user input."""
        while True:
            print(f"\n{self.colors['bold']}Choose your action:{self.colors['reset']}")
            choice = self.get_user_input(
                "Enter 1 (accept), 2 (reject), or 3 (edit)",
                validator=self._validate_choice
            )

            if choice in {"1", "y", "yes", "accept"}:
                self.print_success("Purchase approved!")
                return ApprovalDecision({"type": "accept"})

            elif choice in {"2", "n", "no", "reject"}:
                self.print_warning("Purchase rejected!")
                return ApprovalDecision({"type": "reject"})

            elif choice in {"3", "e", "edit"}:
                return self._get_edit_decision()

    def _validate_choice(self, choice: str) -> None:
        """Validate menu choice input."""
        if choice.lower() not in ["1", "2", "3", "y", "yes", "n", "no", "accept", "reject", "e", "edit"]:
            raise ValueError("Please enter 1, 2, 3, y, or n")

    def _get_edit_decision(self) -> Dict[str, Any]:
        """Get edit decision with new parameters."""
        self.print_info("Edit mode - enter new values (press Enter to keep original)")

        # Get new item name
        new_item = self.get_user_input("New item name")

        # Get new price
        while True:
            new_price_str = self.get_user_input("New price")
            if not new_price_str:
                new_price = None
                break
            try:
                new_price = float(new_price_str)
                if new_price <= 0:
                    raise ValueError()
                break
            except ValueError:
                self.print_error("Please enter a valid positive number for price")

        # Get new vendor
        new_vendor = self.get_user_input("New vendor")

        # Build edit response
        edit_args = {}
        if new_item:
            edit_args["item_name"] = new_item
        if new_price is not None:
            edit_args["price"] = new_price
        if new_vendor:
            edit_args["vendor"] = new_vendor

        if not edit_args:
            self.print_warning("No changes made, proceeding with original parameters")
            return ApprovalDecision({"type": "accept"})

        self.print_success(f"Parameters updated: {edit_args}")
        return ApprovalDecision({
            "type": "edit",
            "args": edit_args
        })

    def display_agent_output(self, output: Dict[str, Any], step_num: int) -> None:
        """Display agent execution output."""
        if SHOW_DETAILED_LOGS:
            self.print_step(step_num, "Agent Output:")
            for key, value in output.items():
                print(f"  {key}: {value}")
        else:
            self.print_step(step_num, f"Agent processed: {list(output.keys())}")

    def display_completion(self, success: bool = True) -> None:
        """Display completion message."""
        if success:
            self.print_success("HITL Agent execution completed successfully!")
        else:
            self.print_error("HITL Agent execution failed!")

    def show_help(self) -> None:
        """Display help information."""
        self.print_header("HITL Agent CLI Help")

        print(f"{self.colors['bold']}Usage:{self.colors['reset']}")
        print("  python hitl.py                    # Run interactive demo")
        print("  python hitl.py --test            # Run in test mode")
        print("  python hitl.py --help            # Show this help")

        print(f"\n{self.colors['bold']}Environment Variables:{self.colors['reset']}")
        print("  HITL_ENV=production             # Set environment")
        print("  HITL_LLM_MODEL=gpt-4           # Change LLM model")
        print("  HITL_VERBOSE=true              # Show detailed logs")
        print("  HITL_TEST_MODE=true            # Enable test mode")

        print(f"\n{self.colors['bold']}Test Mode:{self.colors['reset']}")
        print("  Set HITL_TEST_MODE=true and HITL_MOCK_APPROVAL=accept")
        print("  to automatically approve purchases for testing")

    def confirm_exit(self) -> bool:
        """Ask user to confirm exit."""
        response = self.get_user_input("Do you want to exit? (y/N)", "n")
        return response.lower() in ["y", "yes"]


# Global CLI instance
cli = CLIInterface()


def validate_numeric(value: str) -> float:
    """Validate numeric input."""
    try:
        num = float(value)
        if num <= 0:
            raise ValueError()
        return num
    except ValueError:
        raise ValueError("Must be a positive number")


def validate_choice(options: list, value: str) -> str:
    """Validate choice from options."""
    if value not in options:
        raise ValueError(f"Must be one of: {', '.join(options)}")
    return value
