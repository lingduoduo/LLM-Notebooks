# Agent Rollout Demo

A minimal runnable simulation of the veRL-style asynchronous agent rollout
architecture. It uses only the Python standard library and runs locally.

## Architecture

```text
PPOTrainer
  -> AgentLoopManager
      -> AgentLoopWorker
          -> ToolAgentLoop
              -> AsyncLLMServerManager
                  -> AsyncLLMServer
```

The demo includes:

- asynchronous agent trajectories
- Think -> Act -> Observe loop
- tool calling
- multiple AgentLoopWorkers
- multiple simulated LLM servers
- server routing by fewest cumulative trajectory assignments
- sticky trajectory-to-server assignment
- rollout wake/sleep lifecycle

## Run

Requires Python 3.9+ and no third-party packages.

```bash
python agent_rollout_demo.py
```

Run this command from this directory. The demo processes four prompts
concurrently; each trajectory calls the simulated search tool, then returns a
final answer on its second generation step. Log lines can interleave because
server latency is randomized. Results are grouped by worker, rather than input
order.

The LLM and search responses are fixed examples, so all four prompts receive
the same answer. Wake/sleep methods log lifecycle events without managing real
server resources. Use positive worker/server counts and unique server IDs when
constructing managers directly. Sticky assignments persist for the manager's
lifetime, and routing counts do not measure active server load.

## Relationship to veRL

This is an educational simulation rather than a copy of the veRL implementation.
Replace the simulated LLM servers with vLLM/SGLang and the local asyncio workers
with the corresponding Ray/veRL components to move toward the production architecture.
