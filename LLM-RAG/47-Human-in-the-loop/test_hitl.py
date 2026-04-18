# test_hitl.py
"""
Comprehensive test suite for Human-in-the-Loop Agent System.
Tests all components: config, logging, tools, agent, CLI, and web interface.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from config import (
    PRODUCT_DATABASE, get_config,
    LLM_MODEL,
)
from logger import AuditLogger, AuditEvent, get_logger, setup_logging
from tools import purchase_item, search_product, validate_tool_args, TOOLS
from agent import HITLAgent, AgentState
from cli import CLIInterface


class TestConfig(unittest.TestCase):
    """Test configuration management."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_config_initialization(self):
        """Test config initialization with defaults."""
        config = get_config()
        self.assertEqual(config["llm_model"], "gpt-4-mini")
        self.assertEqual(config["approval_timeout_hours"], 24)
        self.assertTrue(len(PRODUCT_DATABASE) > 0)

    def test_environment_variables(self):
        """Test environment variable overrides."""
        os.environ["LLM_MODEL"] = "gpt-4"
        os.environ["APPROVAL_TIMEOUT_HOURS"] = "48"

        config = get_config()
        self.assertEqual(config.llm_model, "gpt-4")
        self.assertEqual(config.approval_timeout_hours, 48)

    def test_product_database(self):
        """Test product database structure."""
        self.assertIn("MacBook Pro", PRODUCT_DATABASE)
        product = PRODUCT_DATABASE["MacBook Pro"]
        self.assertIn("price", product)
        self.assertIn("vendor", product)
        self.assertIn("currency", product)


class TestLogger(unittest.TestCase):
    """Test logging and audit functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
        self.audit_file = Path(self.temp_dir) / "audit.jsonl"

    def tearDown(self):
        """Clean up test files."""
        for file in [self.log_file, self.audit_file]:
            if file.exists():
                file.unlink()
        os.rmdir(self.temp_dir)

    def test_audit_logger_initialization(self):
        """Test audit logger setup."""
        AuditLogger(self.audit_file)
        self.assertTrue(self.audit_file.exists())

    def test_audit_event_creation(self):
        """Test audit event creation and logging."""
        audit_logger = AuditLogger(self.audit_file)

        event = AuditEvent(
            thread_id="test-thread",
            user_id="test-user",
            action="purchase_approved",
            details={"item": "MacBook Pro", "price": 15999.0}
        )

        audit_logger.log_event(event)

        # Verify event was written
        with open(self.audit_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)

            logged_event = json.loads(lines[0])
            self.assertEqual(logged_event["thread_id"], "test-thread")
            self.assertEqual(logged_event["action"], "purchase_approved")

    def test_purchase_approval_logging(self):
        """Test purchase approval logging."""
        audit_logger = AuditLogger(self.audit_file)

        audit_logger.log_purchase_approval(
            "thread-123",
            "user-456",
            "MacBook Pro",
            15999.0,
            "Apple Store",
            "approved"
        )

        with open(self.audit_file, 'r') as f:
            logged_event = json.loads(f.readline())
            self.assertEqual(logged_event["action"], "purchase_approved")
            self.assertEqual(logged_event["details"]["item"], "MacBook Pro")

    def test_setup_logging(self):
        """Test logging setup."""
        setup_logging(self.log_file)
        logger = get_logger("test")
        logger.info("Test message")

        # Verify log file was created and contains message
        self.assertTrue(self.log_file.exists())
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test message", content)


class TestTools(unittest.TestCase):
    """Test tool functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_file = Path(self.temp_dir) / "audit.jsonl"
        self.audit_logger = AuditLogger(self.audit_file)
        self.audit_logger_patch = patch("tools.audit_logger", self.audit_logger)
        self.audit_logger_patch.start()

    def tearDown(self):
        """Clean up test files."""
        self.audit_logger_patch.stop()
        if self.audit_file.exists():
            self.audit_file.unlink()
        os.rmdir(self.temp_dir)

    def test_validate_tool_args_valid(self):
        """Test tool argument validation with valid inputs."""
        # Valid purchase_item args
        valid_purchase = {
            "item": "MacBook Pro",
            "price": 15999.0,
            "vendor": "Apple Store"
        }
        self.assertTrue(validate_tool_args("purchase_item", valid_purchase))

        # Valid search_product args
        valid_search = {"query": "laptop"}
        self.assertTrue(validate_tool_args("search_product", valid_search))

    def test_validate_tool_args_invalid(self):
        """Test tool argument validation with invalid inputs."""
        # Invalid purchase_item args (missing required)
        invalid_purchase = {"item": "MacBook Pro"}  # missing price and vendor
        self.assertFalse(validate_tool_args("purchase_item", invalid_purchase))

        # Invalid search_product args (empty query)
        invalid_search = {"query": ""}
        self.assertFalse(validate_tool_args("search_product", invalid_search))

        # Unknown tool
        self.assertFalse(validate_tool_args("unknown_tool", {}))

    @patch('tools.interrupt')
    def test_purchase_item_execution(self, mock_interrupt):
        """Test purchase item tool execution."""
        # Mock interrupt to simulate HITL
        mock_interrupt.return_value = "approved"

        result = purchase_item.invoke({
            "item": "MacBook Pro",
            "price": 15999.0,
            "vendor": "Apple Store",
            "thread_id": "test-thread"
        })

        # Verify interrupt was called
        mock_interrupt.assert_called_once()
        self.assertIn("Successfully purchased MacBook Pro", result)

        # Verify audit log was created
        with open(self.audit_file, 'r') as f:
            logged_event = json.loads(f.readline())
            self.assertEqual(logged_event["action"], "purchase_requested")

    def test_search_product_execution(self):
        """Test search product tool execution."""
        result = search_product.invoke({"query": "MacBook"})

        # Should return matching products
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

        # Check result structure
        product = result[0]
        self.assertIn("name", product)
        self.assertIn("price", product)
        self.assertIn("vendor", product)

    def test_tools_registration(self):
        """Test that tools are properly registered."""
        self.assertEqual(len(TOOLS), 2)
        tool_names = [tool.name for tool in TOOLS]
        self.assertIn("purchase_item", tool_names)
        self.assertIn("search_product", tool_names)


class TestAgent(unittest.TestCase):
    """Test agent functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "agent.log"

    def tearDown(self):
        """Clean up test files."""
        if self.log_file.exists():
            self.log_file.unlink()
        os.rmdir(self.temp_dir)

    @patch('agent.ChatOpenAI')
    def test_agent_initialization(self, mock_chat_openai):
        """Test agent initialization."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        agent = HITLAgent()

        # Verify LLM was initialized with correct parameters
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        self.assertEqual(call_args[1]["model"], LLM_MODEL)

        # Verify graph was built
        self.assertIsNotNone(agent.graph)

    @patch('agent.ChatOpenAI')
    def test_agent_batch_execution(self, mock_chat_openai):
        """Test batch execution mode."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_llm.invoke.return_value = mock_response

        agent = HITLAgent()
        messages = ["Hello", "How are you?"]

        results = agent.run_batch(messages)

        # Verify results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result["success"])
            self.assertIn("result", result)

    def test_agent_state_structure(self):
        """Test agent state structure."""
        state = AgentState(messages=[])
        self.assertIsInstance(state["messages"], list)


class TestCLI(unittest.TestCase):
    """Test CLI interface functionality."""

    def setUp(self):
        """Set up test environment."""
        self.cli = CLIInterface()

    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_user_input(self, mock_print, mock_input):
        """Test user input handling."""
        mock_input.return_value = "test input"

        result = self.cli.get_user_input("Enter something:")

        mock_print.assert_called_with("Enter something:")
        self.assertEqual(result, "test input")

    @patch('builtins.input')
    def test_get_approval_decision_accept(self, mock_input):
        """Test approval decision input - accept."""
        mock_input.return_value = "y"

        decision = self.cli.get_approval_decision()

        self.assertEqual(decision, "accept")

    @patch('builtins.input')
    def test_get_approval_decision_reject(self, mock_input):
        """Test approval decision input - reject."""
        mock_input.return_value = "n"

        decision = self.cli.get_approval_decision()

        self.assertEqual(decision, "reject")

    @patch('builtins.input')
    @patch('cli.print')
    def test_display_agent_output(self, mock_print_func, mock_input):
        """Test agent output display."""
        chunk = {
            "chatbot": {
                "messages": [{"content": "Hello from agent"}]
            }
        }

        self.cli.display_agent_output(chunk, 1)

        # Verify print was called
        mock_print_func.assert_called()


class TestWebInterface(unittest.TestCase):
    """Test web interface functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files."""
        os.rmdir(self.temp_dir)

    def test_web_app_initialization(self):
        """Test web app setup."""
        from web import app

        # Verify FastAPI app is initialized at import time for ASGI servers.
        self.assertEqual(app.title, "HITL Agent API")
        self.assertEqual(app.version, "1.0.0")

    def test_chat_request_model(self):
        """Test chat request model validation."""
        from web import ChatRequest

        # Valid request
        request = ChatRequest(message="Hello", thread_id="123", user_id="user1")
        self.assertEqual(request.message, "Hello")

        # Invalid request (empty message)
        with self.assertRaises(ValueError):
            ChatRequest(message="")

    def test_approval_request_model(self):
        """Test approval request model validation."""
        from web import ApprovalRequest

        # Valid request
        request = ApprovalRequest(
            approval_id="123",
            decision="accept",
            edit_args={"price": 15000}
        )
        self.assertEqual(request.decision, "accept")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "integration.log"
        self.audit_file = Path(self.temp_dir) / "audit.jsonl"

        # Setup logging
        setup_logging(self.log_file)

    def tearDown(self):
        """Clean up integration test files."""
        for file in [self.log_file, self.audit_file]:
            if file.exists():
                file.unlink()
        os.rmdir(self.temp_dir)

    @patch('agent.ChatOpenAI')
    def test_full_agent_workflow(self, mock_chat_openai):
        """Test complete agent workflow from config to execution."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        # Mock tool call response
        mock_response = MagicMock()
        mock_response.tool_calls = [{
            "name": "search_product",
            "args": {"query": "laptop"},
            "id": "call_123"
        }]
        mock_llm.invoke.return_value = mock_response

        # Mock tool binding
        mock_llm.bind_tools.return_value = mock_llm

        # Create agent
        agent = HITLAgent()

        # Test search workflow
        messages = ["I want to buy a laptop"]
        results = agent.run_batch(messages)

        # Verify workflow completed
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])

    def test_config_to_tools_integration(self):
        """Test integration between config and tools."""
        # Get product from config
        product = PRODUCT_DATABASE["MacBook Pro"]

        # Test tool validation with config data
        args = {
            "item": "MacBook Pro",
            "price": product["price"],
            "vendor": product["vendor"]
        }

        self.assertTrue(validate_tool_args("purchase_item", args))

    def test_logging_integration(self):
        """Test logging integration across components."""
        audit_logger = AuditLogger(self.audit_file)

        # Log an event
        audit_logger.log_agent_action(
            "test-thread",
            "test-user",
            "test_action",
            {"key": "value"}
        )

        # Verify it was written
        self.assertTrue(self.audit_file.exists())
        with open(self.audit_file, 'r') as f:
            event = json.loads(f.readline())
            self.assertEqual(event["action"], "test_action")
            self.assertEqual(event["details"]["key"], "value")


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestConfig,
        TestLogger,
        TestTools,
        TestAgent,
        TestCLI,
        TestWebInterface,
        TestIntegration
    ]

    for test_class in test_classes:
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\nTest Results: {result.testsRun} tests run")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
