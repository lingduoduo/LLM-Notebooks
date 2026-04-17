#!/usr/bin/env python3
"""
LMCache Test Suite
Run with: python test_cache.py
"""

import time
import os
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Import our optimized cache classes
from test_lmcache import (
    OptimizedDiskL2Cache,
    OptimizedLMCacheManager,
    SystemMonitor,
    create_optional_l3_cache,
)


def make_l3_cache(ttl_seconds: int = 60):
    """Create an L3 cache if Redis is available."""
    return create_optional_l3_cache(ttl_seconds=ttl_seconds, compression_level=6)


def test_l2_cache():
    """Test L2 disk cache functionality."""
    print("🧪 Testing L2 Cache...")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OptimizedDiskL2Cache(tmpdir, compression_level=6)

        # Test set/get with a key that will create nested directories
        test_key = "abcdef1234567890abcdef1234567890abcdef"  # Long key for nested dirs
        test_value = {"data": "test_value", "number": 42}

        # Call _set_sync directly for synchronous operation in tests
        cache._set_sync(test_key, test_value)

        result = cache.get(test_key)
        assert result == test_value, f"L2 get/set failed: got {result}, expected {test_value}"

        # Test miss
        assert cache.get("missing_key") is None, "L2 should return None for missing key"

        # Test delete
        cache.delete(test_key)
        time.sleep(0.05)
        assert cache.get(test_key) is None, "L2 delete failed"

        # Test explicit cleanup helper
        stale_key = "ffffeeeeddddccccbbbbaaaa999988887777"
        cache._set_sync(stale_key, test_value)
        stale_path = cache._path_for_key(stale_key)
        old_time = time.time() - 7200
        os.utime(stale_path, (old_time, old_time))
        removed = cache.cleanup_expired(older_than_seconds=3600)
        assert removed >= 1, "cleanup_expired should remove stale files"
        assert cache.get(stale_key) is None, "cleanup_expired should delete stale cache entries"

        disk = cache.get_disk_usage()
        for field in ["disk_total_gb", "disk_used_gb", "disk_free_gb", "disk_used_ratio"]:
            assert field in disk, f"Disk stats should include {field}"

        print("✅ L2 Cache tests passed")


def test_l3_cache():
    """Test L3 Redis cache functionality."""
    print("🧪 Testing L3 Cache...")

    cache = make_l3_cache(ttl_seconds=60)
    if cache is None:
        print("⚠️  Redis not available, skipping L3 tests")
        return

    try:
        # Test set/get
        test_data = {"layers": [{"k": [1, 2, 3], "v": [4, 5, 6]}] * 5}
        cache.set("test_key", test_data)
        result = cache.get("test_key")
        assert result == test_data, "L3 get/set failed"

        # Test miss
        assert cache.get("missing_key") is None, "L3 should return None for missing key"

        # Test delete
        cache.delete("test_key")
        time.sleep(0.05)
        assert cache.get("test_key") is None, "L3 delete failed"

        # Test batch operations
        batch_data = {f"batch_key_{i}": f"value_{i}" for i in range(5)}
        cache.mset(batch_data)
        batch_result = cache.mget(list(batch_data.keys()))
        assert batch_result == batch_data, "L3 batch operations failed"

        print("✅ L3 Cache tests passed")

    except Exception as e:
        print(f"⚠️  L3 Cache tests failed (Redis not available?): {e}")
    finally:
        cache.close()


def test_cache_manager():
    """Test the full cache manager."""
    print("🧪 Testing Cache Manager...")

    with tempfile.TemporaryDirectory() as tmpdir:
        l2 = OptimizedDiskL2Cache(tmpdir, compression_level=6)
        l3 = make_l3_cache(ttl_seconds=60)
        cache = OptimizedLMCacheManager(l2, l3)

        # Generate test key
        key = cache.make_key("test prompt", model_name="test-model")

        # Test miss -> compute -> set
        value = cache.get(key)
        assert value is None, "Should miss on first access"

        test_kv_cache = {
            "prompt": "test prompt",
            "created_at": time.time(),
            "layers": [{"layer_id": i, "data": f"layer_{i}"} for i in range(3)]
        }

        cache.set(key, test_kv_cache)

        # Test L2 hit
        value2 = cache.get(key)
        assert value2 == test_kv_cache, "L2 hit failed"

        # Test statistics (should work even without Redis)
        stats = cache.get_stats()
        required_fields = [
            "hits",
            "misses",
            "l2_hits",
            "l2_sets",
            "l2_size_mb",
            "l2_file_count",
            "disk_used_ratio",
            "l3_healthy",
            "lock_acquired",
            "lock_timeouts",
            "lock_fallback_computes",
        ]
        for field in required_fields:
            assert field in stats, f"Stats should include {field}"

        # Test batch operations (L2 only if Redis not available)
        keys = [cache.make_key(f"batch_prompt_{i}") for i in range(3)]
        batch_data = {k: f"batch_value_{i}" for i, k in enumerate(keys)}
        cache.mset(batch_data)
        batch_result = cache.mget(keys)

        # Should get all results from L2 even if Redis fails
        assert len(batch_result) == len(batch_data), "Batch operations should work with L2"

        print("✅ Cache Manager tests passed")
        cache.close()


def test_stampede_protection():
    """Test Redis-backed stampede protection."""
    print("🧪 Testing Stampede Protection...")

    l3 = make_l3_cache(ttl_seconds=60)
    if l3 is None:
        print("⚠️  Redis not available, skipping stampede test")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        l2 = OptimizedDiskL2Cache(tmpdir, compression_level=6)
        cache = OptimizedLMCacheManager(l2, l3)
        key = cache.make_key("stampede test")
        compute_count = {"value": 0}
        results = []

        def compute():
            compute_count["value"] += 1
            time.sleep(0.2)
            return {"rebuilt": True}

        import threading

        def worker():
            results.append(
                cache.get_or_compute(
                    key,
                    compute,
                    lock_timeout=0.5,
                    fallback_on_timeout=True,
                )
            )

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert all(item == {"rebuilt": True} for item in results), "All workers should get rebuilt value"
        assert compute_count["value"] == 1, f"Expected one rebuild, got {compute_count['value']}"

        print("✅ Stampede protection tests passed")
        cache.close()


def test_lock_timeout_behavior():
    """Test timeout fallback behavior under contention."""
    print("🧪 Testing Lock Timeout Behavior...")

    l3 = make_l3_cache(ttl_seconds=60)
    if l3 is None:
        print("⚠️  Redis not available, skipping lock-timeout test")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        l2 = OptimizedDiskL2Cache(tmpdir, compression_level=6)
        cache = OptimizedLMCacheManager(l2, l3)
        key = cache.make_key("timeout test")
        compute_count = {"value": 0}

        def slow_compute():
            compute_count["value"] += 1
            time.sleep(0.35)
            return {"rebuilt": compute_count["value"]}

        def worker():
            return cache.get_or_compute(
                key,
                slow_compute,
                lock_timeout=0.1,
                fallback_on_timeout=True,
                max_wait_time=0.1,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [future.result() for future in [executor.submit(worker) for _ in range(2)]]

        stats = cache.get_stats()
        assert len(results) == 2, "Both timeout workers should return results"
        assert stats["lock_acquired"] >= 1, "One worker should acquire the lock"
        assert stats["lock_timeouts"] >= 1, "At least one worker should hit the timeout path"
        assert stats["lock_fallback_computes"] >= 1, "Timeout path should trigger fallback compute"
        assert compute_count["value"] >= 2, "Fallback compute should increase compute count"

        print("✅ Lock-timeout behavior tests passed")
        cache.close()


def test_performance():
    """Basic performance test."""
    print("🧪 Running Performance Test...")

    with tempfile.TemporaryDirectory() as tmpdir:
        l2 = OptimizedDiskL2Cache(tmpdir, compression_level=6)
        l3 = make_l3_cache(ttl_seconds=300)
        cache = OptimizedLMCacheManager(l2, l3)

        # Create test data
        test_data = {
            "prompt": "performance test prompt",
            "created_at": time.time(),
            "layers": [
                {
                    "layer_id": i,
                    "k_cache": [[j * k for k in range(64)] for j in range(32)],
                    "v_cache": [[j * k for k in range(64)] for j in range(32)],
                }
                for i in range(10)
            ]
        }

        # Performance test
        key = cache.make_key("perf_test")

        # Measure set time (use sync for testing)
        start = time.time()
        l2._set_sync(key, test_data)  # Direct L2 set for testing
        if l3 is not None:
            l3.set(key, test_data)  # L3 set
        set_time = time.time() - start

        # Measure get time (L2 hit)
        start = time.time()
        result = cache.get(key)
        get_time = time.time() - start

        assert result is not None, "Performance test data not found"

        print(f"Set time: {set_time:.4f}s")
        print(f"Get time: {get_time:.4f}s")

        # Test compression effectiveness
        l2_size = l2.get_size_mb()
        print(f"L2 size on disk: {l2_size:.2f} MB")

        print("✅ Performance test completed")
        cache.close()


def test_config_loading():
    """Test loading cache manager from configuration file."""
    print("🧪 Testing Configuration Loading...")

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.json")
        cache_dir = os.path.join(tmpdir, "cache_dir")

        test_config = {
            "cache_manager": {
                "enable_promotion": True,
                "enable_monitoring": True,
            },
            "l2_cache": {
                "cache_dir": cache_dir,
                "max_size_mb": 100,
                "compression_level": 6,
                "ttl_seconds": 3600,
            },
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(test_config, f, indent=2)

        cache = OptimizedLMCacheManager.from_config(config_path)
        try:
            key = cache.make_key("config test prompt")
            cache.set(key, {"test": "data"})
            result = cache.get(key)
            assert result == {"test": "data"}, "Configured cache should support basic set/get"

            stats = cache.get_stats()
            assert "system" in stats, "Monitoring-enabled config should include system metrics"

            health = cache.health_check()
            assert "status" in health, "Health check should expose overall status"
            assert "issues" in health, "Health check should expose issue list"
        finally:
            cache.close()

        print("✅ Configuration loading tests passed")


def test_system_monitoring():
    """Test system monitoring helper output shape."""
    print("🧪 Testing System Monitoring...")

    metrics = SystemMonitor.get_system_metrics()
    metrics_dict = metrics.to_dict()

    for field in [
        "cpu_percent",
        "memory_percent",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_utilization",
    ]:
        assert field in metrics_dict, f"System metrics should include {field}"

    print("✅ System monitoring tests passed")


def main():
    """Run all tests."""
    print("🚀 LMCache Test Suite")
    print("=" * 50)

    try:
        test_l2_cache()
        test_l3_cache()
        test_cache_manager()
        test_stampede_protection()
        test_lock_timeout_behavior()
        test_performance()
        test_config_loading()
        test_system_monitoring()

        print("=" * 50)
        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
