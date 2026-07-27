from unittest.mock import MagicMock, patch

from config import Config, LLMConfig


def test_agent_constructs_direct_openai_client():
    from agent import AgenticRAG

    config = Config()
    config.llm = LLMConfig(api_key="test-key")
    with patch("agent.OpenAI") as openai:
        agent = AgenticRAG(config)
    openai.assert_called_once_with(api_key="test-key")
    assert agent.model == "gpt-5.6-terra"


def test_contextual_chunker_constructs_direct_openai_client():
    from contextual_chunking import ContextualChunker

    with patch("contextual_chunking.OpenAI") as openai:
        chunker = ContextualChunker(
            llm_config=LLMConfig(api_key="test-key"),
            use_contextual=True,
        )
    openai.assert_called_once_with(api_key="test-key")
    assert chunker.model == "gpt-5.6-terra"


def test_gpt5_temperature_remains_reasoning_safe():
    from agent import _reasoning_safe_temperature as agent_temperature
    from contextual_chunking import _reasoning_safe_temperature as chunk_temperature

    assert agent_temperature("gpt-5.6-terra", 0.3) == 1
    assert chunk_temperature("gpt-5.6-terra", 0.3) == 1
    assert agent_temperature("gpt-4.1-mini", 0.3) == 0.3


def test_contextual_chunker_uses_configured_generation_settings():
    from contextual_chunking import ContextualChunker

    response = MagicMock()
    response.choices[0].message.content = "Document context"
    llm_config = LLMConfig(
        model="gpt-4.1-mini",
        api_key="test-key",
        temperature=0.42,
        max_tokens=321,
    )

    with patch("contextual_chunking.OpenAI") as openai:
        openai.return_value.chat.completions.create.return_value = response
        chunker = ContextualChunker(llm_config=llm_config)
        chunker._generate_chunk_context("chunk text", "full document")

    openai.return_value.chat.completions.create.assert_called_once()
    request = openai.return_value.chat.completions.create.call_args.kwargs
    assert request["temperature"] == 0.42
    assert request["max_tokens"] == 321
