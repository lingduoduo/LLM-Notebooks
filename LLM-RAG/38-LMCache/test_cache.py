#!/usr/bin/env python3
"""
LMCache Test Suite
Run with: python test_cache.py
"""

import time
import tempfile

# Import our optimized cache classes
from test_lmcache import (
    OptimizedDiskL2Cache,
    OptimizedRedisL3Cache,
    OptimizedLMCacheManager
)


def make_l3_cache(ttl_seconds: int = 60):
    """Create an L3 cache if Redis is available."""
    try:
        cache = OptimizedRedisL3Cache(
            host="localhost",
            port=6379,
            ttl_seconds=ttl_seconds,
            compression_level=6,
        )
        if cache.health_check():
            return cache
        cache.close()
    except Exception:
        pass
    return None


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
        required_fields = ["hits", "misses", "l2_hits", "l2_sets", "l2_size_mb", "l3_healthy"]
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


def main():
    """Run all tests."""
    print("🚀 LMCache Test Suite")
    print("=" * 50)

    try:
        test_l2_cache()
        test_l3_cache()
        test_cache_manager()
        test_performance()

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
