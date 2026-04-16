import os
import time
import json
import hashlib
import threading
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import zlib

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    import pickle

try:
    import redis
    from redis.connection import ConnectionPool
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers (shared by L2 and L3 to avoid code duplication)
# ---------------------------------------------------------------------------

def _serialize(value: Any, compression_level: int = 6) -> bytes:
    if HAS_MSGPACK:
        data = msgpack.packb(value, use_bin_type=True)
    else:
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if compression_level > 0:
        return zlib.compress(data, level=compression_level)
    return data


def _deserialize(data: bytes, compression_level: int = 6) -> Any:
    if compression_level > 0:
        data = zlib.decompress(data)
    if HAS_MSGPACK:
        return msgpack.unpackb(data, raw=False)
    return pickle.loads(data)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    l2_sets: int = 0
    l3_sets: int = 0
    l2_promotions: int = 0
    avg_l2_latency: float = 0.0
    avg_l3_latency: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# L2: local disk cache
# ---------------------------------------------------------------------------

class OptimizedDiskL2Cache:
    """
    Optimized L2 cache: local disk-based cache with compression and memory mapping.
    Uses msgpack for faster serialization and zlib for compression.
    """

    def __init__(self, root_dir: str = "./l2_cache", max_size_mb: int = 1024,
                 compression_level: int = 6, ttl_seconds: int = 86400) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression_level = compression_level
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="l2-cache")

        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()

    def _dir_for_key(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4]

    def _path_for_key(self, key: str) -> Path:
        return self._dir_for_key(key) / f"{key}.cache"

    def get(self, key: str) -> Optional[Any]:
        path = self._path_for_key(key)
        if not path.exists():
            return None

        try:
            stat = path.stat()
            if time.time() - stat.st_mtime > self.ttl_seconds:
                self._executor.submit(self._safe_unlink, path)
                return None

            with open(path, "rb") as f:
                data = f.read()
            return _deserialize(data, self.compression_level)
        except Exception as e:
            logger.warning(f"[L2] Failed to read key={key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        self._executor.submit(self._set_sync, key, value)

    def _set_sync(self, key: str, value: Any) -> None:
        path = self._path_for_key(key)
        tmp_path = path.with_suffix(".tmp")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = _serialize(value, self.compression_level)
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.rename(tmp_path, path)  # atomic rename
        except Exception as e:
            logger.error(f"[L2] Failed to write key={key}: {e}")
            with suppress(FileNotFoundError):
                tmp_path.unlink()

    def delete(self, key: str) -> None:
        self._executor.submit(self._safe_unlink, self._path_for_key(key))

    def _safe_unlink(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")

    def _periodic_cleanup(self) -> None:
        while not self._shutdown.wait(300):
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def _cleanup_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        removed = 0

        for cache_file in self.root.rglob("*.cache"):
            try:
                if cache_file.stat().st_mtime < cutoff:
                    cache_file.unlink()
                    removed += 1
            except Exception:
                continue

        if removed > 0:
            logger.info(f"[L2] Cleaned up {removed} expired entries")

    def get_size_mb(self) -> float:
        total_size = 0
        for cache_file in self.root.rglob("*.cache"):
            try:
                total_size += cache_file.stat().st_size
            except Exception:
                continue
        return total_size / (1024 * 1024)

    def clear(self) -> None:
        for cache_file in self.root.rglob("*.cache"):
            try:
                cache_file.unlink()
            except Exception:
                continue
        logger.info("[L2] Cache cleared")

    def close(self) -> None:
        self._shutdown.set()
        self._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# L3: Redis cache
# ---------------------------------------------------------------------------

class OptimizedRedisL3Cache:
    """
    Optimized L3 cache: Redis-backed with connection pooling and batch operations.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "lmcache:",
        ttl_seconds: int = 3600,
        max_connections: int = 10,
        compression_level: int = 6,
    ) -> None:
        if not HAS_REDIS:
            raise ImportError("redis package is required for L3 cache")

        self.pool = ConnectionPool(
            host=host, port=port, db=db, max_connections=max_connections,
            decode_responses=False, socket_timeout=5, socket_connect_timeout=5
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.compression_level = compression_level
        # Redis ConnectionPool is thread-safe; no extra lock needed.

    def _redis_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            raw = self.client.get(self._redis_key(key))
            if raw is None:
                return None
            return _deserialize(raw, self.compression_level)
        except Exception as e:
            logger.warning(f"[L3] Failed to read key={key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            raw = _serialize(value, self.compression_level)
            self.client.setex(self._redis_key(key), self.ttl_seconds, raw)
        except Exception as e:
            logger.error(f"[L3] Failed to write key={key}: {e}")

    def delete(self, key: str) -> None:
        try:
            self.client.delete(self._redis_key(key))
        except Exception as e:
            logger.warning(f"[L3] Failed to delete key={key}: {e}")

    def mget(self, keys: list) -> Dict[str, Any]:
        if not keys:
            return {}

        redis_keys = [self._redis_key(k) for k in keys]
        try:
            values = self.client.mget(redis_keys)
            result = {}
            for key, raw in zip(keys, values):
                if raw is not None:
                    try:
                        result[key] = _deserialize(raw, self.compression_level)
                    except Exception as e:
                        logger.warning(f"[L3] Failed to deserialize key={key}: {e}")
            return result
        except Exception as e:
            logger.error(f"[L3] Batch get failed: {e}")
            return {}

    def mset(self, key_value_pairs: Dict[str, Any]) -> None:
        if not key_value_pairs:
            return

        try:
            pipeline = self.client.pipeline()
            for key, value in key_value_pairs.items():
                raw = _serialize(value, self.compression_level)
                pipeline.setex(self._redis_key(key), self.ttl_seconds, raw)
            pipeline.execute()
        except Exception as e:
            logger.error(f"[L3] Batch set failed: {e}")

    def health_check(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    def close(self) -> None:
        with suppress(Exception):
            self.client.close()
        with suppress(Exception):
            self.pool.disconnect()


# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------

class OptimizedLMCacheManager:
    """
    Optimized two-level cache manager with advanced features:
    - L2 → L3 → miss lookup order
    - Automatic L3-to-L2 promotion
    - Cache statistics and monitoring
    - Batch operations support
    - Async L2 and L3 writes for better throughput
    """

    def __init__(self, l2: OptimizedDiskL2Cache, l3: Optional[OptimizedRedisL3Cache] = None,
                 enable_promotion: bool = True) -> None:
        self.l2 = l2
        self.l3 = l3
        self.enable_promotion = enable_promotion
        self.stats = CacheStats()
        self._lock = threading.Lock()
        # Offload L3 writes so they don't block callers (L2 already has its own executor).
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="manager")

    @staticmethod
    def make_key(prompt: str, model_name: str = "demo-model", **kwargs) -> str:
        payload = {"model": model_name, "prompt": prompt, **kwargs}
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value with L2 → L3 → miss lookup order."""
        start_time = time.perf_counter()

        # 1. Try L2 cache
        value = self.l2.get(key)
        if value is not None:
            latency = time.perf_counter() - start_time
            with self._lock:
                self.stats.hits += 1
                self.stats.l2_hits += 1
                # Welford cumulative moving average — unbiased unlike (old+new)/2
                self.stats.avg_l2_latency += (latency - self.stats.avg_l2_latency) / self.stats.l2_hits
            logger.debug(f"[Cache] L2 hit for key={key[:8]}...")
            return value

        # 2. Try L3 cache
        if self.l3 is None:
            with self._lock:
                self.stats.misses += 1
            logger.debug(f"[Cache] Miss for key={key[:8]}...")
            return None

        l3_start = time.perf_counter()
        value = self.l3.get(key)
        l3_latency = time.perf_counter() - l3_start

        if value is not None:
            with self._lock:
                self.stats.hits += 1
                self.stats.l3_hits += 1
                self.stats.avg_l3_latency += (l3_latency - self.stats.avg_l3_latency) / self.stats.l3_hits

            if self.enable_promotion:
                self.l2.set(key, value)
                with self._lock:
                    self.stats.l2_promotions += 1
                logger.debug(f"[Cache] L3 hit -> promoted to L2 for key={key[:8]}...")

            return value

        # 3. Cache miss
        with self._lock:
            self.stats.misses += 1
        logger.debug(f"[Cache] Miss for key={key[:8]}...")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in both L2 and L3 caches; L3 write is non-blocking."""
        self.l2.set(key, value)
        if self.l3 is not None:
            self._executor.submit(self.l3.set, key, value)
        with self._lock:
            self.stats.l2_sets += 1
            if self.l3 is not None:
                self.stats.l3_sets += 1
        logger.debug(f"[Cache] Set key={key[:8]}... in L2+L3")

    def delete(self, key: str) -> None:
        self.l2.delete(key)
        if self.l3 is not None:
            self.l3.delete(key)

    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get multiple keys efficiently."""
        if not keys:
            return {}

        result: Dict[str, Any] = {}
        l3_needed: List[str] = []

        for key in keys:
            value = self.l2.get(key)
            if value is not None:
                result[key] = value
            else:
                l3_needed.append(key)

        l2_hit_count = len(result)
        with self._lock:
            self.stats.hits += l2_hit_count
            self.stats.l2_hits += l2_hit_count

        if l3_needed and self.l3 is not None:
            l3_results = self.l3.mget(l3_needed)
            for key, value in l3_results.items():
                result[key] = value
                if self.enable_promotion:
                    self.l2.set(key, value)

            l3_hit_count = len(l3_results)
            with self._lock:
                self.stats.hits += l3_hit_count
                self.stats.l3_hits += l3_hit_count
                self.stats.l2_promotions += l3_hit_count if self.enable_promotion else 0

        with self._lock:
            self.stats.misses += len(keys) - len(result)

        return result

    def mset(self, key_value_pairs: Dict[str, Any]) -> None:
        """Batch set multiple key-value pairs; L3 write is non-blocking."""
        if not key_value_pairs:
            return

        for key, value in key_value_pairs.items():
            self.l2.set(key, value)

        if self.l3 is not None:
            self._executor.submit(self.l3.mset, key_value_pairs)

        count = len(key_value_pairs)
        with self._lock:
            self.stats.l2_sets += count
            if self.l3 is not None:
                self.stats.l3_sets += count

    def get_stats(self) -> Dict:
        """Get cache performance statistics."""
        # Snapshot stats under lock, then do I/O outside it.
        with self._lock:
            stats = self.stats.to_dict()
        stats["l2_size_mb"] = self.l2.get_size_mb()
        stats["l3_healthy"] = self.l3.health_check() if self.l3 is not None else False
        return stats

    def clear_l2(self) -> None:
        self.l2.clear()

    def warmup(self, keys: List[str]) -> None:
        """Warm up L2 by preloading specified keys from L3."""
        if not keys or self.l3 is None:
            return

        l3_results = self.l3.mget(keys)
        for key, value in l3_results.items():
            self.l2.set(key, value)
        with self._lock:
            self.stats.l2_promotions += len(l3_results)

        logger.info(f"[Cache] Warmed up {len(l3_results)} keys from L3 to L2")

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self.l2.close()
        if self.l3 is not None:
            self.l3.close()


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def fake_llm_prefill(prompt: str, model_name: str = "qwen-demo") -> dict:
    """Simulate LLM KV cache generation with realistic payload."""
    logger.info("[Model] Computing KV cache from scratch...")
    time.sleep(1.5)

    kv_cache = {
        "prompt": prompt,
        "model": model_name,
        "created_at": time.time(),
        "seq_len": len(prompt.split()),
        "layers": [
            {
                "layer_id": i,
                "k_cache": [[i * j + k for k in range(64)] for j in range(128)],
                "v_cache": [[i * j + k for k in range(64)] for j in range(128)],
                "attention_mask": [1.0] * 128,
            }
            for i in range(32)
        ],
        "metadata": {
            "compute_time": 1.5,
            "memory_usage_mb": 512,
            "model_params": "7B",
        }
    }
    return kv_cache


def benchmark_cache(cache: OptimizedLMCacheManager, iterations: int = 10) -> None:
    """Benchmark cache performance."""
    logger.info("Starting cache benchmark...")

    prompts = [
        "Hello, how are you?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What is the meaning of life?",
        "Describe the water cycle.",
    ]

    logger.info("Warming up cache...")
    for prompt in prompts:
        key = cache.make_key(prompt)
        if cache.get(key) is None:
            cache.set(key, fake_llm_prefill(prompt))

    logger.info(f"Running {iterations} iterations...")
    start_time = time.perf_counter()

    for i in range(iterations):
        prompt = prompts[i % len(prompts)]
        key = cache.make_key(prompt)
        value = cache.get(key)
        if value is None:
            cache.set(key, fake_llm_prefill(prompt))

    total_time = time.perf_counter() - start_time
    logger.info(
        "Benchmark completed in %.2fs (%.2f ops/s)",
        total_time,
        iterations / total_time if total_time > 0 else 0.0,
    )
    logger.info(f"Cache stats: {json.dumps(cache.get_stats(), indent=2)}")


def main() -> None:
    l2 = OptimizedDiskL2Cache("./l2_cache", max_size_mb=512, compression_level=6, ttl_seconds=3600)
    l3 = None
    try:
        candidate_l3 = OptimizedRedisL3Cache(host="localhost", port=6379, ttl_seconds=3600, compression_level=6)
        if candidate_l3.health_check():
            l3 = candidate_l3
        else:
            candidate_l3.close()
            logger.warning("Redis is unavailable; continuing with L2-only cache")
    except Exception as exc:
        logger.warning("Redis setup failed; continuing with L2-only cache: %s", exc)
    cache = OptimizedLMCacheManager(l2, l3, enable_promotion=True)

    logger.info("=== LMCache Demo ===")

    prompt = "You are a helpful assistant. Summarize LMCache in one paragraph."
    key = cache.make_key(prompt, model_name="qwen-demo")

    logger.info("--- First Request (Cold Start) ---")
    value = cache.get(key)
    if value is None:
        value = fake_llm_prefill(prompt)
        cache.set(key, value)
        logger.info("[Write] Stored into L2 and L3")
    logger.info(f"[Result 1] created_at={value['created_at']:.1f}, layers={len(value['layers'])}")

    logger.info("\n--- Second Request (L2 Hit) ---")
    value2 = cache.get(key)
    if value2 is None:
        value2 = fake_llm_prefill(prompt)
        cache.set(key, value2)
    logger.info(f"[Result 2] created_at={value2['created_at']:.1f}, layers={len(value2['layers'])}")

    logger.info("\n--- Third Request (L3 Hit + Promotion) ---")
    cache.clear_l2()
    value3 = cache.get(key)
    if value3 is None:
        logger.warning("[Result 3] Key not found — Redis may be unavailable")
    else:
        logger.info(f"[Result 3] created_at={value3['created_at']:.1f}, layers={len(value3['layers'])}")

    logger.info("\n--- Final Statistics ---")
    logger.info(json.dumps(cache.get_stats(), indent=2))

    logger.info("\n--- Benchmark ---")
    benchmark_cache(cache, iterations=5)
    cache.close()


if __name__ == "__main__":
    main()
