"""Controller tests use disposable servers, never the model services."""

import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from services import Controller, Service, ServiceError, service_plan


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def free_port(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return listener.getsockname()[1]

    def test_plan_uses_canonical_ports_and_sparse_positional_argument(self):
        plan = service_plan(Path("/repo/AI-Agent-KnowledgeBase"), sys.executable, {})
        self.assertEqual([service.port for service in plan], [4240, 4241, 4242])
        self.assertEqual(plan[1].command[-1], "4241")
        self.assertNotIn("--port", plan[1].command)
        self.assertEqual(plan[0].command[-2:], ("--port", "4240"))
        override = service_plan(
            self.root, sys.executable, {"DENSE_PYTHON": "/dense/python"}
        )
        self.assertEqual(override[0].command[0], "/dense/python")
        self.assertEqual(override[1].command[0], sys.executable)

    def test_start_status_stop_disposable_server(self):
        port = self.free_port()
        service = Service(
            "dummy",
            self.root,
            port,
            (sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"),
        )
        controller = Controller([service], self.root / "state", timeout=5)
        self.addCleanup(controller.stop)
        controller.start()
        self.assertEqual(controller.status()["dummy"], "running")
        with self.assertRaises(ServiceError):
            controller.start()
        # A later invocation has no Popen handle and must rely on persisted identity.
        fresh_controller = Controller([service], controller.state_dir, timeout=5)
        fresh_controller.stop()
        self.assertEqual(fresh_controller.status()["dummy"], "stopped")
        controller.stop()  # Reap the disposable child from its original parent.

    def test_occupied_port_is_left_untouched(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            service = Service(
                "occupied",
                self.root,
                port,
                (sys.executable, "-c", "raise SystemExit(99)"),
            )
            controller = Controller([service], self.root / "state", timeout=1)
            with self.assertRaisesRegex(ServiceError, "occupied"):
                controller.start()
            controller.stop()
            self.assertEqual(listener.getsockname()[1], port)

    def test_stale_pid_identity_does_not_signal_foreign_process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        self.addCleanup(lambda: process.wait(timeout=5))
        self.addCleanup(process.terminate)
        service = Service("foreign", self.root, self.free_port(), (sys.executable,))
        controller = Controller([service], self.root / "state", timeout=1)
        controller.state_dir.mkdir()
        controller._save(
            service.name, {"pid": process.pid, "identity": "a different process"}
        )
        controller.stop()
        self.assertIsNone(process.poll())

    def test_failed_start_rolls_back_processes(self):
        service = Service(
            "failed",
            self.root,
            self.free_port(),
            (sys.executable, "-c", "raise SystemExit(2)"),
        )
        controller = Controller([service], self.root / "state", timeout=1)
        with self.assertRaises(ServiceError):
            controller.start()
        self.assertEqual(controller.status()["failed"], "stopped")

    def test_identity_lookup_failure_reaps_the_direct_child(self):
        service = Service(
            "unidentified",
            self.root,
            self.free_port(),
            (sys.executable, "-c", "import time; time.sleep(60)"),
        )
        controller = Controller([service], self.root / "state", timeout=1)
        try:
            with patch("services.process_identity", return_value=None):
                with self.assertRaises(ServiceError):
                    controller.start()
            self.assertFalse(
                any(child.poll() is None for child in controller.children.values())
            )
        finally:
            for child in controller.children.values():
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)

    def test_one_shutdown_failure_does_not_skip_other_services(self):
        services = [
            Service(name, self.root, self.free_port(), (sys.executable,))
            for name in ("dense", "sparse", "pipeline")
        ]
        controller = Controller(services, self.root / "state", timeout=1)
        controller.state_dir.mkdir()
        for number, service in enumerate(services, 100):
            controller._save(service.name, {"pid": number, "identity": "example"})
        ticks = iter(range(100))
        alive = {100, 101, 102}

        def identity(pid):
            return "example" if pid in alive else None

        def terminate(pid, _signal):
            if pid != 102:
                alive.discard(pid)

        with patch("services.process_identity", side_effect=identity), patch(
            "services.os.kill", side_effect=terminate
        ) as kill, patch(
            "services.time.monotonic", side_effect=lambda: next(ticks) * 20
        ), patch(
            "services.time.sleep"
        ):
            with self.assertRaises(ServiceError):
                controller.stop()
        self.assertEqual(
            [call.args[0] for call in kill.call_args_list], [102, 101, 100]
        )
        self.assertFalse((controller.state_dir / "dense.json").exists())
        self.assertFalse((controller.state_dir / "sparse.json").exists())
        self.assertTrue((controller.state_dir / "pipeline.json").exists())

    def test_malformed_state_fails_without_signalling(self):
        service = Service("bad", self.root, self.free_port(), (sys.executable,))
        controller = Controller([service], self.root / "state", timeout=1)
        controller.state_dir.mkdir()
        (controller.state_dir / "bad.json").write_text('{"pid": -1, "identity": "bad"}')
        with self.assertRaises(ServiceError):
            controller.stop()


if __name__ == "__main__":
    unittest.main()
