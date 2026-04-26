import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional

from worker import worker_loop


@dataclass
class WorkerState:
    worker_id: int
    process: mp.Process
    task_queue: mp.Queue
    last_crash_time: Optional[float] = None
    restart_count: int = 0


class ReliabilityScheduler:
    """
    A tiny cluster scheduler simulation.

    Responsibilities:
    - maintain worker processes
    - route requests to workers
    - detect crashes
    - restart crashed workers
    - measure recovery time
    """

    def __init__(self, num_workers: int, failure_rate: float):
        self.num_workers = num_workers
        self.failure_rate = failure_rate
        self.result_queue = mp.Queue()
        self.workers: Dict[int, WorkerState] = {}
        self.next_worker_idx = 0

        self.crash_count = 0
        self.restart_count = 0
        self.recovery_times = []

    def start(self):
        for worker_id in range(self.num_workers):
            self.workers[worker_id] = self._start_worker(worker_id)

    def stop(self):
        for state in self.workers.values():
            try:
                state.task_queue.put(None)
            except Exception:
                pass

        for state in self.workers.values():
            if state.process.is_alive():
                state.process.terminate()
                state.process.join(timeout=1)

    def _start_worker(self, worker_id: int) -> WorkerState:
        task_queue = mp.Queue()
        process = mp.Process(
            target=worker_loop,
            args=(worker_id, task_queue, self.result_queue, self.failure_rate),
        )
        process.start()

        return WorkerState(
            worker_id=worker_id,
            process=process,
            task_queue=task_queue,
        )

    def _restart_worker(self, worker_id: int):
        old = self.workers[worker_id]
        recovery_start = old.last_crash_time if old.last_crash_time is not None else time.time()

        # Rescue tasks that were queued but not yet picked up by the dead worker.
        rescued = []
        while not old.task_queue.empty():
            try:
                task = old.task_queue.get_nowait()
                if task is not None:
                    rescued.append(task)
            except Exception:
                break

        try:
            if old.process.is_alive():
                old.process.terminate()
            old.process.join(timeout=1)
        except Exception:
            pass

        new_state = self._start_worker(worker_id)
        new_state.restart_count = old.restart_count + 1

        for task in rescued:
            new_state.task_queue.put(task)

        recovery_time = time.time() - recovery_start
        self.recovery_times.append(recovery_time)

        self.workers[worker_id] = new_state
        self.restart_count += 1

        print(
            f"[monitor] restarted worker={worker_id}, "
            f"recovery_time={recovery_time:.3f}s, rescued_tasks={len(rescued)}"
        )

    def monitor_once(self):
        for worker_id, state in list(self.workers.items()):
            if not state.process.is_alive():
                if state.last_crash_time is None:
                    state.last_crash_time = time.time()
                    self.crash_count += 1
                    print(f"[monitor] detected crash: worker={worker_id}")
                self._restart_worker(worker_id)

    def healthy_workers(self) -> List[int]:
        return [
            worker_id
            for worker_id, state in self.workers.items()
            if state.process.is_alive()
        ]

    def route_request(self, request_id: int, prompt: str):
        healthy = self.healthy_workers()

        if not healthy:
            raise RuntimeError("No healthy workers available")

        worker_id = healthy[self.next_worker_idx % len(healthy)]
        self.next_worker_idx = (self.next_worker_idx + 1) % self.num_workers

        self.workers[worker_id].task_queue.put({
            "request_id": request_id,
            "prompt": prompt,
        })

    def collect_result(self, timeout_sec: float = 2.0):
        try:
            return self.result_queue.get(timeout=timeout_sec)
        except Exception:
            return None
