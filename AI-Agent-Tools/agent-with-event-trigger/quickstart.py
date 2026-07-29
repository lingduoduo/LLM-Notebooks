"""
Quick Start Script for Event-Triggered Agent
Demonstrates the basic functionality in a simple way
"""

import os
import sys
import time
import subprocess
import signal
from event_types import EventType

# Check the API key up front (importing agent also loads the .env file).
from agent import resolve_api_key, DEFAULT_MODEL

api_key = resolve_api_key()

if not api_key:
    print("❌ Error: OPENAI_API_KEY is not set (checked the environment and the .env file)")
    print("\nSet it with either:")
    print("  echo \"OPENAI_API_KEY=sk-...\" >> .env")
    print("  export OPENAI_API_KEY='sk-...'")
    sys.exit(1)

print(f"✅ OpenAI API key found (model: {os.getenv('OPENAI_MODEL') or DEFAULT_MODEL})")

print("\n" + "="*80)
print("🚀 EVENT-TRIGGERED AGENT QUICK START")
print("="*80)
print()

# Check if server is already running
import requests
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    print("✅ Server is already running!")
    print("\n💡 You can now use the client to send events:")
    print("   python client.py --mode test")
    print("   python client.py --mode interactive")
    sys.exit(0)
except:
    pass

print("📦 Starting the event-triggered agent server...")
print("\n⏳ This may take a moment to initialize...\n")

# Start the server in a subprocess
try:
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for server to start
    print("⏰ Waiting for server to start...")
    max_wait = 30
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server is running!\n")
                break
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i}/{max_wait}s)")
    else:
        print("❌ Server failed to start in time")
        server_process.terminate()
        sys.exit(1)
    
    print("="*80)
    print("🎉 QUICK START READY!")
    print("="*80)
    print()
    print("The event-triggered agent server is now running on port 8000.")
    print()
    print("📋 What you can do now:")
    print()
    print("1. Send test events (in another terminal):")
    print("   python client.py --mode test")
    print()
    print("2. Use interactive mode:")
    print("   python client.py --mode interactive")
    print()
    print("3. Send individual events via API:")
    print("   curl -X POST http://localhost:8000/event \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"event_type\": \"web_message\", \"content\": \"Hello!\"}'")
    print()
    print("4. Check agent status:")
    print("   curl http://localhost:8000/agent/status")
    print()
    print("="*80)
    print("📺 Server output will appear below:")
    print("="*80)
    print()
    
    # Stream server output
    try:
        while True:
            line = server_process.stdout.readline()
            if not line:
                break
            print(line, end='')
    except KeyboardInterrupt:
        print("\n\n⚠️ Shutting down server...")
        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=5)
        print("✅ Server stopped")
        
except FileNotFoundError:
    print("❌ Error: Could not find server.py")
    print("Make sure you're in the agent-with-event-trigger directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
