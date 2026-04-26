import os
import time
import random
from multiprocessing import Queue


def worker_loop(worker_id: int, task_queue: Queue, result_queue: Queue, failure_rate: float):
    """
    Simulates an LLM inference worker.

    Each worker:
    - receives requests from its own queue
    - simulates variable inference latency
    - randomly crashes based on failure_rate
    - returns response metadata to the scheduler
    """
    random.seed(worker_id * 31337 ^ os.getpid())

    while True:
        task = task_queue.get()

        if task is None:
            break

        request_id = task["request_id"]
        prompt = task["prompt"]
        start = time.time()

        if random.random() < failure_rate:
            result_queue.put({
                "request_id": request_id,
                "worker_id": worker_id,
                "status": "worker_crashed",
                "latency_sec": time.time() - start,
                "response": None,
            })
            raise RuntimeError(f"Worker {worker_id} crashed while processing request {request_id}")

        time.sleep(random.uniform(0.03, 0.15))

        response = f"[worker={worker_id}] generated response for: {prompt[:50]}"

        result_queue.put({
            "request_id": request_id,
            "worker_id": worker_id,
            "status": "ok",
            "latency_sec": time.time() - start,
            "response": response,
        })
