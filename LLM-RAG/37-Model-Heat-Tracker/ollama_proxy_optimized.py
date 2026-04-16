import json
import time
import threading
from collections import defaultdict, deque
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ModelHeatTracker:
    """Tracks request patterns and calculates model heat scores efficiently."""

    def __init__(self, window_minutes: int = 15):
        self.request_history = defaultdict(deque)
        self.model_stats = defaultdict(
            lambda: {
                "total_requests": 0,
                "heat_score": 0.0,
                "last_request_at": 0.0,
            }
        )
        self.window_seconds = window_minutes * 60
        self._lock = threading.Lock()

    def record_request(self, model_id: str) -> None:
        current_time = time.time()
        cutoff = current_time - self.window_seconds

        with self._lock:
            history = self.request_history[model_id]
            history.append(current_time)

            while history and history[0] < cutoff:
                history.popleft()

            stats = self.model_stats[model_id]
            stats["total_requests"] += 1
            stats["heat_score"] = len(history)
            stats["last_request_at"] = current_time

    def get_hot_models(self, top_n: int = 2) -> List[str]:
        # Copy under lock, sort outside to reduce lock hold time.
        with self._lock:
            items = list(self.model_stats.items())
        items.sort(key=lambda x: x[1]["heat_score"], reverse=True)
        return [mid for mid, _ in items[:top_n]]

    def get_stats(self) -> Dict:
        with self._lock:
            return {mid: dict(stats) for mid, stats in self.model_stats.items()}


class OllamaProxy:
    """Intelligent Ollama proxy with heat-based model scheduling and resource optimization."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        max_hot_models: int = 3,
        request_timeout: int = 30,
        loader_threads: int = 2,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.heat_tracker = ModelHeatTracker()
        self.loaded_models: Set[str] = set()
        # Covers both queued-but-not-started and actively loading models.
        # Event is created at queue time so _wait_for_model never misses it.
        self.loading_models: Dict[str, threading.Event] = {}
        self.failed_models: Dict[str, str] = {}
        self.model_queue: Queue = Queue()
        self.max_hot_models = max_hot_models
        self.request_timeout = request_timeout
        self._loader_threads = loader_threads
        self._lock = threading.Lock()
        self._shutdown = False

        self.session = self._create_session()

        for _ in range(loader_threads):
            threading.Thread(target=self._loader_worker, daemon=True).start()

        threading.Thread(target=self._optimization_loop, daemon=True).start()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=10, pool_maxsize=10
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _loader_worker(self) -> None:
        while not self._shutdown:
            model_id: Optional[str] = None
            try:
                model_id = self.model_queue.get(timeout=1)
                if model_id is None:  # shutdown sentinel
                    return
                with self._lock:
                    already_loaded = model_id in self.loaded_models
                if not already_loaded:
                    self._load_model(model_id)
            except Empty:
                continue
            finally:
                if model_id is not None:
                    self.model_queue.task_done()

    def _ensure_model_queued(self, model_id: str) -> None:
        """Queue a model load at most once; create the wait event immediately so
        _wait_for_model never returns False for a pending load."""
        with self._lock:
            if model_id in self.loaded_models or model_id in self.loading_models:
                return
            self.loading_models[model_id] = threading.Event()
        self.model_queue.put(model_id)

    def _load_model(self, model_id: str) -> bool:
        with self._lock:
            if model_id in self.loaded_models:
                # Loaded while waiting in queue; signal and clean up.
                event = self.loading_models.pop(model_id, None)
                if event:
                    event.set()
                return True
            event = self.loading_models.get(model_id)
            if event is None:
                event = threading.Event()
                self.loading_models[model_id] = event
            self.failed_models.pop(model_id, None)

        try:
            start = time.time()
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model_id, "prompt": "", "stream": False},
                timeout=self.request_timeout * 2,
            )

            if response.status_code == 200:
                with self._lock:
                    self.loaded_models.add(model_id)
                print(f"✓ Loaded {model_id} in {time.time() - start:.1f}s")
                return True
            else:
                with self._lock:
                    self.failed_models[model_id] = f"HTTP {response.status_code}"
                print(f"✗ Failed to load {model_id}: HTTP {response.status_code}")
                return False
        except Exception as e:
            with self._lock:
                self.failed_models[model_id] = str(e)
            print(f"✗ Failed to load {model_id}: {e}")
            return False
        finally:
            event.set()
            with self._lock:
                self.loading_models.pop(model_id, None)

    def _wait_for_model(self, model_id: str, timeout: int = 30) -> bool:
        with self._lock:
            if model_id in self.loaded_models:
                return True
            event = self.loading_models.get(model_id)

        if event:
            event.wait(timeout=timeout)
            with self._lock:
                return model_id in self.loaded_models

        return False

    def _get_load_error(self, model_id: str) -> Optional[str]:
        with self._lock:
            return self.failed_models.get(model_id)

    def generate(self, model_id: str, prompt: str, max_tokens: int = 100) -> str:
        if not prompt.strip():
            return "Error: Prompt cannot be empty"

        self.heat_tracker.record_request(model_id)
        self._ensure_model_queued(model_id)

        if not self._wait_for_model(model_id, timeout=30):
            error = self._get_load_error(model_id)
            if error:
                return f"Error: Model {model_id} failed to load: {error}"
            return f"Error: Model {model_id} failed to load within timeout"

        try:
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            return f"Error: Request timeout after {self.request_timeout}s"
        except requests.exceptions.JSONDecodeError:
            return "Error: Invalid response from Ollama server"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            loaded = list(self.loaded_models)
            loading = list(self.loading_models.keys())
            failed = dict(self.failed_models)

        return {
            "loaded_models": loaded,
            "loading_models": loading,
            "failed_models": failed,
            "hot_models": self.heat_tracker.get_hot_models(self.max_hot_models),
            "model_stats": self.heat_tracker.get_stats(),
        }

    def _optimization_loop(self) -> None:
        while not self._shutdown:
            try:
                time.sleep(300)

                hot_models = set(self.heat_tracker.get_hot_models(self.max_hot_models))

                with self._lock:
                    cold_models = [m for m in self.loaded_models if m not in hot_models]

                for model in cold_models:
                    self._unload_model(model)
            except Exception as e:
                print(f"Error in optimization loop: {e}")

    def _unload_model(self, model_id: str) -> None:
        """Remove model from tracking and signal Ollama to release its GPU memory."""
        with self._lock:
            self.loaded_models.discard(model_id)
        try:
            # keep_alive=0 tells Ollama to evict the model immediately.
            self.session.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model_id, "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass
        print(f"⊘ Unloaded {model_id}")

    def preheat_models(self, model_ids: List[str]) -> None:
        for model_id in model_ids:
            self._ensure_model_queued(model_id)

    def close(self) -> None:
        """Shutdown gracefully: unblock all worker threads, then release resources."""
        self._shutdown = True
        for _ in range(self._loader_threads):
            self.model_queue.put(None)  # wake each blocked worker
        self.session.close()


if __name__ == "__main__":
    def run_demo():
        proxy = OllamaProxy(max_hot_models=2)

        proxy.preheat_models(["llama2"])
        time.sleep(1)

        print("=" * 50)
        print("=== Cold Start Demo ===")
        print("=" * 50)
        start = time.time()
        response = proxy.generate("llama2", "Hello, introduce yourself briefly.")
        elapsed = time.time() - start
        print(f"Response: {response[:80]}...")
        print(f"Latency: {elapsed:.2f}s\n")

        print("=" * 50)
        print("=== Warm Request Demo (Same Model) ===")
        print("=" * 50)
        start = time.time()
        response = proxy.generate("llama2", "What is artificial intelligence?")
        elapsed = time.time() - start
        print(f"Response: {response[:80]}...")
        print(f"Latency: {elapsed:.2f}s (much faster!)\n")

        print("=" * 50)
        print("=== Model Switch Demo ===")
        print("=" * 50)
        start = time.time()
        response = proxy.generate("mistral", "Explain quantum computing briefly.")
        elapsed = time.time() - start
        print(f"Response: {response[:80]}...")
        print(f"Latency: {elapsed:.2f}s\n")

        print("=" * 50)
        print("=== System Status ===")
        print("=" * 50)
        print(json.dumps(proxy.get_status(), indent=2))

        proxy.close()

    run_demo()
