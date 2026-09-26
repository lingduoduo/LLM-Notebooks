"""Simulate asynchronous, tool-using agent rollouts with sticky LLM routing."""

import asyncio
import heapq
import random
from collections.abc import Sequence
from dataclasses import dataclass

MAX_AGENT_STEPS = 5


@dataclass
class AgentLoopOutput:
    """The final answer and generation step count for one trajectory."""

    prompt: str
    answer: str
    steps: int


class AsyncLLMServer:
    """Simulated vLLM/SGLang rollout server."""

    def __init__(self, server_id: int) -> None:
        self.server_id = server_id

    async def generate(self, request_id: str, prompt: str) -> str:
        await asyncio.sleep(random.uniform(0.1, 0.4))
        print(f"    [LLM Server {self.server_id}] request={request_id}")

        if "Observation:" not in prompt:
            return "TOOL:search:agentic reinforcement learning"

        return (
            "FINAL: Agentic reinforcement learning trains an agent using "
            "rewards collected from its trajectories."
        )


class AsyncLLMServerManager:
    """Route new trajectories by cumulative assignments, with sticky sessions.

    Counts track assigned trajectories, not active requests. Each request ID
    stays on its original server for the lifetime of this manager.
    """

    def __init__(self, servers: Sequence[AsyncLLMServer]) -> None:
        self.servers = servers
        self.weighted_servers = [[0, server.server_id, server] for server in servers]
        heapq.heapify(self.weighted_servers)
        self.request_id_to_server: dict[str, AsyncLLMServer] = {}

    def _choose_server(self, request_id: str) -> AsyncLLMServer:
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        assignment_count, server_id, server = heapq.heappop(self.weighted_servers)
        heapq.heappush(self.weighted_servers, [assignment_count + 1, server_id, server])
        self.request_id_to_server[request_id] = server

        print(f"  [ServerManager] {request_id} -> server {server_id}")
        return server

    async def generate(self, request_id: str, prompt: str) -> str:
        server = self._choose_server(request_id)
        return await server.generate(request_id=request_id, prompt=prompt)


async def search(query: str) -> str:
    """Simulated external search tool."""
    print(f"    [Tool] search({query!r})")
    await asyncio.sleep(0.2)
    return (
        "Agentic RL optimizes an agent based on rewards obtained from "
        "multi-step interactions with an environment."
    )


class ToolAgentLoop:
    """Minimal Think -> Act -> Observe agent loop."""

    def __init__(self, server_manager: AsyncLLMServerManager) -> None:
        self.server_manager = server_manager

    async def run(self, request_id: str, user_prompt: str) -> AgentLoopOutput:
        context = user_prompt

        for step in range(1, MAX_AGENT_STEPS + 1):
            print(f"[Agent {request_id}] step={step}")

            output = await self.server_manager.generate(
                request_id=request_id,
                prompt=context,
            )

            if output.startswith("TOOL:"):
                _, tool_name, argument = output.split(":", maxsplit=2)

                if tool_name == "search":
                    observation = await search(argument)
                    context += (
                        f"\n\nTool call: search({argument})\nObservation: {observation}"
                    )
                    continue

            if output.startswith("FINAL:"):
                return AgentLoopOutput(
                    prompt=user_prompt,
                    answer=output.removeprefix("FINAL:").strip(),
                    steps=step,
                )

        return AgentLoopOutput(
            prompt=user_prompt,
            answer="Agent exceeded maximum steps.",
            steps=MAX_AGENT_STEPS,
        )


class AgentLoopWorker:
    """Runs multiple agent trajectories concurrently."""

    def __init__(self, worker_id: int, server_manager: AsyncLLMServerManager) -> None:
        self.worker_id = worker_id
        self.server_manager = server_manager

    async def _run_agent_loop(self, index: int, prompt: str) -> AgentLoopOutput:
        request_id = f"sample-{index}"
        agent_loop = ToolAgentLoop(self.server_manager)

        return await agent_loop.run(
            request_id=request_id,
            user_prompt=prompt,
        )

    async def generate_sequences(
        self, samples: Sequence[tuple[int, str]]
    ) -> list[AgentLoopOutput]:
        print(f"[Worker {self.worker_id}] received {len(samples)} samples")

        tasks = [
            asyncio.create_task(self._run_agent_loop(index, prompt))
            for index, prompt in samples
        ]

        return await asyncio.gather(*tasks)


class AgentLoopManager:
    """Initializes servers/workers, distributes prompts, and collects rollouts."""

    def __init__(self, num_workers: int = 2, num_servers: int = 2) -> None:
        self.servers = [AsyncLLMServer(i) for i in range(num_servers)]
        self.server_manager = AsyncLLMServerManager(self.servers)

        self.agent_loop_workers = [
            AgentLoopWorker(
                worker_id=i,
                server_manager=self.server_manager,
            )
            for i in range(num_workers)
        ]

    async def wake_up(self) -> None:
        print("\n=== Wake up rollout servers ===")

    async def sleep(self) -> None:
        print("\n=== Sleep rollout servers ===")

    async def generate_sequences(self, prompts: Sequence[str]) -> list[AgentLoopOutput]:
        """Distribute prompts round-robin and return results grouped by worker."""
        await self.wake_up()

        samples = list(enumerate(prompts))

        num_workers = len(self.agent_loop_workers)
        chunks = [samples[i::num_workers] for i in range(num_workers)]

        tasks = [
            worker.generate_sequences(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ]

        worker_outputs = await asyncio.gather(*tasks)

        results = [output for outputs in worker_outputs for output in outputs]

        await self.sleep()
        return results


class PPOTrainer:
    """Simplified stand-in for RayPPOTrainer rollout generation."""

    def __init__(self) -> None:
        self.async_rollout_manager = AgentLoopManager(
            num_workers=2,
            num_servers=2,
        )

    async def fit(self) -> None:
        prompts = [
            "Explain agentic reinforcement learning.",
            "What is GRPO?",
            "How does tool calling work?",
            "Why do agents need state?",
        ]

        print("=== PPO rollout ===")

        outputs = await self.async_rollout_manager.generate_sequences(prompts)

        print("\n=== Rollout results ===")

        for result in outputs:
            print()
            print("Prompt:", result.prompt)
            print("Answer:", result.answer)
            print("Steps:", result.steps)

        # A real veRL training loop would continue with reward
        # computation, advantages, PPO/GRPO loss, and optimizer.step().


async def main() -> None:
    trainer = PPOTrainer()
    await trainer.fit()


if __name__ == "__main__":
    asyncio.run(main())
