# LMCache: Optimized L2+L3 Caching for LLM KV Cache

A high-performance, two-level caching system designed for LLM KV cache storage and retrieval, optimized for macOS with local disk (L2) and Redis (L3) backends.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Request   │───▶│  L2 Cache   │───▶│  L3 Cache   │
│             │    │  (Disk)     │    │  (Redis)    │
└─────────────┘    └─────────────┘    └─────────────┘
                        │                    │
                        ▼                    ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Compute   │    │    Miss     │
                   │   (Slow)    │    │             │
                   └─────────────┘    └─────────────┘
```

**Lookup Order:** L2 → L3 → Miss
**Promotion:** L3 hits are automatically promoted to L2
**Storage:** Compressed msgpack serialization (falls back to pickle)

## Features

- 🚀 **High Performance**: Async operations, connection pooling, compression
- 💾 **Efficient Storage**: msgpack + zlib compression reduces storage by 60-80%
- 📊 **Rich Metrics**: Hit rates, latencies, cache sizes, health monitoring
- 🔄 **Auto Promotion**: L3 hits automatically move to L2 for faster access
- 🔐 **Stampede Protection**: Redis-backed single-flight rebuilds for hot keys
- 🧹 **TTL Support**: Automatic cleanup of expired entries
- 📦 **Batch Operations**: Efficient mget/mset for multiple keys
- 🐳 **Docker Ready**: Redis in Docker for easy L3 setup
- 🍎 **macOS Optimized**: APFS-friendly file operations
- 📈 **System Monitoring**: GPU memory, CPU, and system resource tracking
- ⚙️ **Configuration Management**: JSON-based configuration loading
- 🏥 **Health Checks**: Comprehensive system and cache health monitoring

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Redis (L3 Cache)

```bash
# Make setup script executable and run it
chmod +x setup_redis.sh
./setup_redis.sh
```

This starts Redis in Docker with:
- Port: 6379
- Memory limit: 512MB
- Persistence: AOF enabled
- LRU eviction policy

### 3. Run Tests

```bash
# Run the full LMCache test suite
python test_cache.py
```

### 4. Basic Usage

```python
from test_lmcache import OptimizedLMCacheManager, OptimizedDiskL2Cache, OptimizedRedisL3Cache

# Initialize caches
l2 = OptimizedDiskL2Cache("./l2_cache", max_size_mb=1024, compression_level=6)
l3 = OptimizedRedisL3Cache(host="localhost", port=6379, compression_level=6)
cache = OptimizedLMCacheManager(l2, l3)

# Generate cache key
key = cache.make_key("Your prompt here", model_name="llama-7b")

# Basic get/set
cache.set(key, {"kv_cache": "data", "metadata": "info"})
result = cache.get(key)

# Get or compute with stampede protection
value = cache.get_or_compute(
    key,
    lambda: expensive_llm_computation(),
    lock_timeout=0.5,
    fallback_on_timeout=True,
)
```

## Performance Benchmarks

Typical performance on M1/M2 MacBook:

| Operation | Latency | Notes |
|-----------|---------|-------|
| L2 Hit | 2-5ms | Local SSD access |
| L3 Hit | 5-15ms | Redis via Docker |
| Cache Miss | 1500ms | Full computation |
| Promotion | 10-20ms | L3→L2 copy |

**Compression Savings:**
- Raw pickle: ~50MB for 32-layer KV cache
- msgpack + zlib: ~8MB (84% reduction)

## API Usage

```python
from test_lmcache import OptimizedLMCacheManager, OptimizedDiskL2Cache, OptimizedRedisL3Cache

# Initialize caches
l2 = OptimizedDiskL2Cache("./l2_cache", max_size_mb=1024, compression_level=6)
l3 = OptimizedRedisL3Cache(host="localhost", port=6379, compression_level=6)
cache = OptimizedLMCacheManager(l2, l3)

# Generate cache key
key = cache.make_key("Your prompt here", model_name="llama-7b")

# Basic operations
cache.set(key, {"kv_cache": "data", "metadata": "info"})
result = cache.get(key)

# Batch operations
keys = [cache.make_key(f"prompt_{i}") for i in range(10)]
results = cache.mget(keys)  # Returns dict of found items

# Get or compute with stampede protection
value = cache.get_or_compute(
    key,
    lambda: expensive_llm_computation(),
    lock_timeout=0.5,
    fallback_on_timeout=True,
    max_wait_time=30.0
)

# Statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"L2 size: {stats['l2_size_mb']:.1f}MB")
print(f"L3 healthy: {stats['l3_healthy']}")
```

## Configuration

LMCache supports configuration via JSON files for production deployments. Use the provided `config_template.json` as a starting point:

```json
{
  "cache_manager": {
    "enable_promotion": true,
    "enable_monitoring": true
  },
  "l2_cache": {
    "root_dir": "./lmcache_l2",
    "max_size_mb": 1024,
    "compression_level": 6,
    "ttl_seconds": 86400
  },
  "l3_cache": {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": null,
    "prefix": "lmcache:",
    "ttl_seconds": 3600,
    "compression_level": 6,
    "max_connections": 10
  }
}
```

Load configuration in your application:

```python
from test_lmcache import OptimizedLMCacheManager

# Load from config file
cache = OptimizedLMCacheManager.from_config('config_template.json')
```

`from_config()` accepts both `l2_cache.root_dir` and the older `l2_cache.cache_dir` key for backward compatibility.

### System Monitoring

LMCache includes comprehensive system monitoring capabilities:

```python
# Get detailed statistics including system metrics
stats = cache.get_stats()
print(f"CPU Usage: {stats['system']['cpu_percent']}%")
print(f"Memory Usage: {stats['system']['memory_percent']}%")
if stats['system']['gpu_memory_used_mb']:
    print(f"GPU Memory: {stats['system']['gpu_memory_used_mb']}MB / {stats['system']['gpu_memory_total_mb']}MB")

# Health check with system monitoring
health = cache.health_check()
print(f"Status: {health['status']}")
if health['issues']:
    print(f"Issues: {health['issues']}")
```

### Production Monitoring

LMCache includes comprehensive monitoring for production deployments:

**System Resource Tracking:**
- GPU memory usage and utilization (via nvidia-smi)
- CPU and memory usage (via psutil)
- Disk usage monitoring
- Cache performance metrics

**Health Checks:**
```python
health = cache.health_check()
# Returns: status, timestamp, l2_healthy, l3_healthy, issues, system
```

**Enhanced Statistics:**
```python
stats = cache.get_stats()
# Includes: hit_rate, requests_per_second, system metrics, cache sizes
```

### Stampede Protection

**Cache Stampede Prevention** prevents the "thundering herd" problem where multiple concurrent requests for an uncached key trigger expensive computations simultaneously.

```python
# Automatic stampede protection
value = cache.get_or_compute(
    key,
    lambda: expensive_llm_computation(),
    lock_timeout=0.5,        # How long to wait for lock acquisition
    fallback_on_timeout=True, # Allow fallback computation if lock times out
    max_wait_time=30.0        # Max time to wait for cache population
)
```

**How it works:**
1. **Lock Acquisition**: First request acquires a Redis-based distributed lock
2. **Single Computation**: Only the lock holder performs the expensive computation
3. **Cache Population**: Result is stored in both L2 and L3 caches
4. **Concurrent Serving**: Other requests wait and get served from cache
5. **Timeout Handling**: If lock holder fails, requests can fallback to computing themselves

**Benefits:**
- ✅ Prevents multiple expensive computations for the same key
- ✅ Reduces backend load during traffic spikes
- ✅ Maintains low latency for concurrent requests
- ✅ Graceful degradation with timeout fallbacks

**Test Results:**
```
Concurrent requests: 8
Computation calls: 1 (vs 8 without protection)
All results identical: True
Cache hits: 7 (served from cache)
Lock acquired: 1
Lock timeouts: 0
```

## Advanced Features

### Cache Warming
```python
# Preload frequently used keys from L3 to L2
cache.warmup(["key1", "key2", "key3"])
```

### Health Monitoring
```python
stats = cache.get_stats()
print(f"L3 healthy: {stats['l3_healthy']}")
print(f"L2 size: {stats['l2_size_mb']}MB")
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

### Cleanup Operations
```python
# Clear L2 cache
cache.clear_l2()

# Manual cleanup (automatic otherwise)
# TTL-based cleanup runs every 5 minutes
removed = cache.l2.cleanup_expired(older_than_seconds=24 * 3600)
print(f"Removed {removed} stale cache files")

# Get disk usage info
stats = cache.get_stats()
print(f"L2 files: {stats.get('l2_file_count', 'N/A')}")
```

### Automated Maintenance

For long-running deployments, use the provided cleanup scripts to prevent disk space issues:

```bash
# Run cleanup once
python cleanup_cache.py --root /path/to/l2_cache --older-than-hours 24 --warn-threshold 0.8

# Run cleanup with reduced priority
./cleanup_cache.sh --root /path/to/l2_cache --older-than-hours 24 --warn-threshold 0.8
```

Suggested cron entry for automated cleanup:

```cron
0 * * * * /path/to/cleanup_cache.sh --root /path/to/l2_cache --older-than-hours 24 --warn-threshold 0.8 >> /var/log/lmcache_cleanup.log 2>&1
```

The cleanup script returns exit code `2` when disk usage crosses the warning threshold, making it easy to integrate with monitoring systems.

### Monitoring Examples
```bash
# Cache hit rate and performance metrics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"L2 hits: {stats['l2_hits']}, L3 hits: {stats['l3_hits']}")
print(f"Lock acquired: {stats['lock_acquired']}")
print(f"Lock timeouts: {stats['lock_timeouts']}")

# GPU memory (if available)
# nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Disk usage
du -sh ./l2_cache
```

## Production Deployment

### Docker Compose Setup
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data

  lmcache-app:
    build: .
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      - L2_CACHE_DIR=/app/cache

volumes:
  redis_data:
```

### Environment Variables
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export L2_CACHE_DIR=./l2_cache
export L2_MAX_SIZE_MB=2048
export CACHE_COMPRESSION=9
```

## Troubleshooting

### Redis Connection Issues
```bash
# Check if Redis is running
docker ps | grep redis

# Test connection
docker exec -it lmcache-redis redis-cli ping

# View Redis logs
docker logs lmcache-redis
```

### Cache Performance Issues
```python
# Get detailed stats
stats = cache.get_stats()
print(json.dumps(stats, indent=2))

# Check L2 disk usage
du -sh ./l2_cache

# Monitor Redis memory
docker exec lmcache-redis redis-cli info memory
```

### Common Issues
- **High latency**: Check Redis memory usage, consider increasing Docker memory limits
- **L2 cache full**: Increase `max_size_mb` or run cleanup more frequently
- **Serialization errors**: Ensure cached objects are pickle/msgpack serializable
- **Lock timeouts**: Increase `lock_timeout` or `max_wait_time` for slow computations
- **Disk space issues**: Run cleanup script or increase disk space allocation

## Testing

LMCache includes comprehensive test suites to validate functionality:

### Basic Functionality Tests
```bash
python test_cache.py
```
Tests L2 cache operations, L3 cache operations, cache manager integration, distributed locking for stampede prevention, timeout fallback behavior, configuration loading, and system monitoring.

### Test Results
```
✅ Basic cache operations work
✅ Cache stampede prevention works
Stats: hits=2, misses=2, lock_acquired=1
All tests passed! 🎉
```

The test suite validates:
- Cache hit/miss behavior
- Compression and serialization
- Distributed locking for stampede prevention
- Concurrent access patterns
- Error handling and graceful degradation

### L2 Cache (Local Disk)
- **Storage**: Nested directory structure (`aa/bb/key.cache`)
- **Atomic writes**: Temp file + `os.replace()` for crash safety
- **Compression**: zlib level 6 (good balance of speed/size)
- **TTL**: File mtime-based expiration with periodic cleanup
- **Threading**: Async writes with ThreadPoolExecutor
- **Disk visibility**: File count and filesystem-usage metrics for alerting

### L3 Cache (Redis)
- **Connection pooling**: 10 persistent connections
- **Serialization**: msgpack + zlib compression
- **TTL**: Redis EXPIRE for automatic cleanup
- **Batch ops**: Pipeline mget/mset for efficiency
- **Health checks**: Automatic ping monitoring

### Cache Manager
- **Lookup order**: L2 → L3 → miss (with promotion)
- **Concurrency control**: Redis lock per key avoids cache avalanche on hot expirations
- **Statistics**: Real-time hit/miss tracking
- **Configuration loading**: JSON config support with backward-compatible L2 path keys
- **Async ops**: Non-blocking cache operations
- **Error handling**: Graceful degradation on failures

## Files

- `test_lmcache.py` - Main cache implementation with L2/L3 caching and stampede protection
- `test_cache.py` - Main LMCache test suite, including stampede prevention tests
- `cleanup_cache.py` - Automated cache cleanup script
- `cleanup_cache.sh` - Shell wrapper for cleanup with nice/ionice
- `setup_redis.sh` - Redis Docker setup script
- `config.example.json` - Configuration template
- `requirements.txt` - Python dependencies
- `README.md` - This documentation
- `38.ipynb` - Jupyter notebook with LMCache concepts and examples
- `vLLM_lmcache.ipynb` - Integration examples with vLLM
