#!/bin/bash

# LMCache Setup Script for macOS
# This script sets up Redis in Docker for L3 cache testing

set -e

echo "🚀 Setting up LMCache environment on macOS..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop for Mac:"
    echo "   https://docs.docker.com/desktop/install/mac-install/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"

if docker ps -a --format '{{.Names}}' | grep -qx 'lmcache-redis'; then
    echo "📦 Reusing existing Redis container..."
    docker start lmcache-redis >/dev/null
else
    # Pull and run Redis container
    echo "📦 Starting Redis container..."
    docker run -d \
        --name lmcache-redis \
        -p 6379:6379 \
        --restart unless-stopped \
        redis:7-alpine \
        redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru >/dev/null
fi

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to start..."
sleep 3

# Test Redis connection
if docker exec lmcache-redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis is running and responding"
else
    echo "❌ Redis failed to start properly"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Redis is now running on localhost:6379"
echo ""
echo "To run the cache demo:"
echo "  python test_lmcache.py"
echo ""
echo "To stop Redis:"
echo "  docker stop lmcache-redis"
echo ""
echo "To restart Redis:"
echo "  docker start lmcache-redis"
echo ""
echo "To completely remove Redis:"
echo "  docker rm -f lmcache-redis"
