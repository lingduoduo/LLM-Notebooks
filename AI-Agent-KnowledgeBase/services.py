#!/usr/bin/env python3
"""Control the three retrieval services on macOS/Linux using recorded PID identities."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


class ServiceError(RuntimeError):
    """An actionable service startup, state, or shutdown failure."""


@dataclass(frozen=True)
class Service:
    name: str
    directory: Path
    port: int
    command: tuple


def service_plan(root, python, environment):
    """Keep service ports, launch arguments and per-service interpreters in one place."""
    specs = (
        ("dense", "dense-embedding", 4240, "main.py", ("--port", "4240")),
        ("sparse", "sparse-embedding", 4241, "server.py", ("4241",)),
        ("pipeline", "retrieval-pipeline", 4242, "main.py", ("--port", "4242")),
    )
    return [
        Service(
            name,
            root / directory,
            port,
            (
                environment.get(name.upper() + "_PYTHON") or python,
                str(root / directory / script),
                *arguments,
            ),
        )
        for name, directory, port, script, arguments in specs
    ]


def process_identity(pid):
    """PID alone is unsafe after reuse; compare start time and full command as well."""
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "stat=", "-o", "lstart=", "-o", "args="],
        capture_output=True,
        text=True,
        check=False,
    )
    identity = result.stdout.strip()
    if result.returncode or not identity or identity.split()[0].startswith("Z"):
        return None
    # Process state (R/S/etc.) changes normally; it is not part of the identity.
    return identity.split(None, 1)[1]


def port_is_open(port):
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return True
        except OSError:
            pass
    return False


class Controller:
    def __init__(self, services, state_dir, timeout=120):
        self.services = services
        self.state_dir = Path(state_dir)
        self.timeout = timeout
        self.children = {}

    @contextmanager
    def locked(self):
        """Serialize mutations made by separate launcher invocations."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.state_dir / ".lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ServiceError("Another service command is in progress") from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _load(self, name):
        path = self.state_dir / (name + ".json")
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(record, dict)
                or type(record.get("pid")) is not int
                or record["pid"] <= 1
                or not isinstance(record.get("identity"), str)
                or not record["identity"]
            ):
                raise ValueError("invalid PID or identity")
            return record
        except (ValueError, OSError) as error:
            raise ServiceError(f"Invalid service state {path}: {error}") from None

    def _save(self, name, record):
        path = self.state_dir / (name + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(path)

    def _owns(self, record):
        return (
            record is not None and process_identity(record["pid"]) == record["identity"]
        )

    def status(self):
        statuses = {}
        for service in self.services:
            record = self._load(service.name)
            if self._owns(record):
                statuses[service.name] = (
                    "running" if port_is_open(service.port) else "starting"
                )
            else:
                statuses[service.name] = (
                    "unmanaged" if port_is_open(service.port) else "stopped"
                )
        return statuses

    def start(self):
        with self.locked():
            # Preflight the whole plan before launching any process.
            for service in self.services:
                if self._owns(self._load(service.name)):
                    raise ServiceError(
                        f"{service.name} is already managed; use status or restart"
                    )
                if port_is_open(service.port):
                    raise ServiceError(
                        f"Port {service.port} is occupied; leave its process running or stop it manually"
                    )
            started = []
            try:
                for service in self.services:
                    log_path = self.state_dir / (service.name + ".log")
                    with log_path.open("ab") as log:
                        environment = dict(os.environ)
                        # Keep the pipeline consistent even if shell URL overrides are present.
                        if service.name == "pipeline":
                            environment.update(
                                DENSE_SERVICE_URL="http://localhost:4240",
                                SPARSE_SERVICE_URL="http://localhost:4241",
                            )
                        process = subprocess.Popen(
                            service.command,
                            cwd=service.directory,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    self.children[service.name] = process
                    started.append(service)
                    identity = process_identity(process.pid)
                    if not identity:
                        raise ServiceError(
                            f"{service.name} could not be identified during startup; see {log_path}"
                        )
                    self._save(service.name, {"pid": process.pid, "identity": identity})
                    deadline = time.monotonic() + self.timeout
                    while True:
                        if process.poll() is not None:
                            raise ServiceError(
                                f"{service.name} exited during startup; see {log_path}"
                            )
                        if port_is_open(service.port):
                            # macOS framework Python may re-exec during startup, changing
                            # argv while retaining its PID. Persist the ready process.
                            identity = process_identity(process.pid)
                            if not identity:
                                raise ServiceError(
                                    f"{service.name} exited before readiness"
                                )
                            self._save(
                                service.name, {"pid": process.pid, "identity": identity}
                            )
                            print(
                                f"{service.name}: http://localhost:{service.port} (PID {process.pid})",
                                flush=True,
                            )
                            break
                        if time.monotonic() >= deadline:
                            raise ServiceError(
                                f"{service.name} startup timed out; see {log_path}"
                            )
                        time.sleep(0.1)
            except BaseException:
                self._stop(started)
                raise

    def _stop(self, services):
        errors = []
        for service in reversed(services):
            try:
                record = self._load(service.name)
                child = self.children.get(service.name)
                if child is not None:
                    # A direct, unreaped child is ours even if ps/state writing failed.
                    if child.poll() is None:
                        child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if record is not None:
                            raise ServiceError("did not stop; retained state for retry")
                        # No usable identity was saved: do not leave an unmanageable child.
                        child.kill()
                        child.wait(timeout=5)
                    self.children.pop(service.name, None)
                elif self._owns(record):
                    try:
                        os.kill(record["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 10
                    while self._owns(record):
                        if time.monotonic() >= deadline:
                            raise ServiceError("did not stop; retained state for retry")
                        time.sleep(0.1)
                (self.state_dir / (service.name + ".json")).unlink(missing_ok=True)
            except (ServiceError, OSError, subprocess.SubprocessError) as error:
                errors.append(f"{service.name}: {error}")
        if errors:
            raise ServiceError("; ".join(errors))

    def stop(self):
        with self.locked():
            self._stop(self.services)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "restart", "status"])
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Default service Python; DENSE_PYTHON/SPARSE_PYTHON/PIPELINE_PYTHON override it",
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Startup seconds per service"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command plan without changing files or processes",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    services = service_plan(ROOT, args.python, os.environ)
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "name": s.name,
                        "port": s.port,
                        "cwd": str(s.directory),
                        "command": s.command,
                    }
                    for s in services
                ],
                indent=2,
            )
        )
        return 0
    controller = Controller(services, ROOT / ".runtime" / "services", args.timeout)
    try:
        if args.action in ("stop", "restart"):
            controller.stop()
        if args.action in ("start", "restart"):
            controller.start()
        for name, status in controller.status().items():
            print(f"{name}: {status}")
        return 0
    except (ServiceError, OSError, subprocess.SubprocessError) as error:
        print(f"Service error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
