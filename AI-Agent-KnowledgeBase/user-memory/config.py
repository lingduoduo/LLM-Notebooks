"""
Configuration module for User Memory System
"""

import logging
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv()

# Set once by Config.setup_logging() so repeated calls don't stack handlers
_LOGGING_CONFIGURED = False


# Default OpenAI chat model. Override per run with MODEL_NAME in .env.
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"


def _is_reasoning_model(model) -> bool:
    """True for OpenAI reasoning models (GPT-5 family, o-series), including
    their OpenRouter-style ids (openai/gpt-5...)."""
    m = str(model or "").lower().replace("/", "-")
    m = m[len("openai-"):] if m.startswith("openai-") else m
    return "gpt-5" in m or m.startswith(("o1", "o3", "o4"))


def _reasoning_safe_temperature(model, requested=1.0):
    """Reasoning models (GPT-5, o-series, ...) only accept temperature=1.
    Return 1 for those; otherwise the requested value so non-reasoning
    providers (Doubao, DeepSeek, Qwen) are unchanged."""
    return 1 if _is_reasoning_model(model) else requested


def max_tokens_param(model, limit) -> Dict[str, Any]:
    """Output-token limit as kwargs for chat.completions.create().

    Reasoning models reject `max_tokens` ("Unsupported parameter: 'max_tokens'
    is not supported with this model. Use 'max_completion_tokens' instead."),
    while the other providers here only understand `max_tokens`."""
    key = "max_completion_tokens" if _is_reasoning_model(model) else "max_tokens"
    return {key: limit}


def openrouter_model_id(model) -> str:
    """Map a provider-native model name to an OpenRouter model id, used by the
    universal OpenRouter fallback. An explicit OPENROUTER_MODEL env var wins."""
    override = os.getenv("OPENROUTER_MODEL")
    if override:
        return override
    m = (model or "").strip()
    if not m:
        return "openai/" + DEFAULT_OPENAI_MODEL
    if "/" in m:
        return m  # already an OpenRouter-style id (e.g. openai/gpt-5.6-luna)
    ml = m.lower()
    if ml.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai/" + m
    if ml.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    # Provider-native ids (doubao-*/qwen/deepseek-*) not hosted on OpenRouter
    # under the same name -> a widely-available OpenAI chat model.
    return "openai/" + DEFAULT_OPENAI_MODEL


# Default model per provider, used to map onto an OpenRouter model id when the
# primary provider key is missing but OPENROUTER_API_KEY is present.
PROVIDER_DEFAULT_MODELS = {
    "openai": DEFAULT_OPENAI_MODEL,
    "siliconflow": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "doubao": "doubao-seed-1-6-thinking-250715",
}


class MemoryMode(Enum):
    """Memory management modes"""
    NOTES = "notes"  # Simple notes/facts (basic)
    ENHANCED_NOTES = "enhanced_notes"  # Enhanced notes with paragraphs containing full context
    JSON_CARDS = "json_cards"  # Hierarchical JSON memory cards (basic)
    ADVANCED_JSON_CARDS = "advanced_json_cards"  # Advanced JSON cards with complete card objects


class Config:
    """Configuration settings for the user memory system"""
    
    # Provider Configuration
    PROVIDER: str = os.getenv("PROVIDER", "openai").lower()  # Default to openai

    # API Keys for different providers
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    DOUBAO_API_KEY: str = os.getenv("DOUBAO_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # Base URLs for different providers
    # OPENAI_BASE_URL lets you point at Azure/proxy/compatible gateways
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"
    DOUBAO_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "")  # Can override default model for provider
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
    MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "4096"))
    
    # Memory Configuration
    MEMORY_MODE: MemoryMode = MemoryMode(os.getenv("MEMORY_MODE", "notes").lower())
    MAX_MEMORY_ITEMS: int = int(os.getenv("MAX_MEMORY_ITEMS", "100"))
    MEMORY_UPDATE_TEMPERATURE: float = float(os.getenv("MEMORY_UPDATE_TEMPERATURE", "0.2"))
    
    # Dify Configuration for conversation history search
    DIFY_API_KEY: str = os.getenv("DIFY_API_KEY", "")
    DIFY_BASE_URL: str = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1")
    DIFY_DATASET_ID: str = os.getenv("DIFY_DATASET_ID", "")
    ENABLE_HISTORY_SEARCH: bool = os.getenv("ENABLE_HISTORY_SEARCH", "false").lower() == "true"
    
    # Session Configuration
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))  # seconds
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))  # tokens
    
    # LOCOMO Benchmark Configuration
    LOCOMO_DATASET_PATH: str = os.getenv("LOCOMO_DATASET_PATH", "data/locomo")
    LOCOMO_OUTPUT_DIR: str = os.getenv("LOCOMO_OUTPUT_DIR", "results/locomo")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "logs/user_memory.log")
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    # Storage paths
    MEMORY_STORAGE_DIR: str = os.getenv("MEMORY_STORAGE_DIR", "data/memories")
    CONVERSATION_HISTORY_DIR: str = os.getenv("CONVERSATION_HISTORY_DIR", "data/conversations")
    
    @classmethod
    def get_api_key(cls, provider: Optional[str] = None) -> Optional[str]:
        """
        Get API key for the specified provider
        
        Args:
            provider: Provider name (defaults to configured provider)
            
        Returns:
            API key or None if not found
        """
        provider = (provider or cls.PROVIDER).lower()
        
        if provider == "openai":
            return cls.OPENAI_API_KEY
        elif provider == "siliconflow":
            return cls.SILICONFLOW_API_KEY
        elif provider == "doubao":
            return cls.DOUBAO_API_KEY
        elif provider == "openrouter":
            return cls.OPENROUTER_API_KEY
        else:
            return None
    
    @classmethod
    def validate(cls, provider: Optional[str] = None) -> bool:
        """
        Validate required configuration
        
        Args:
            provider: Provider to validate (defaults to configured provider)
        
        Returns:
            True if configuration is valid
        """
        provider = (provider or cls.PROVIDER).lower()
        api_key = cls.get_api_key(provider)
        
        if not api_key:
            print(f"ERROR: API key for provider '{provider}' is not set")
            if provider == "openai":
                print("Please set OPENAI_API_KEY in .env file or as environment variable")
            elif provider == "siliconflow":
                print("Please set SILICONFLOW_API_KEY in .env file or as environment variable")
            elif provider == "doubao":
                print("Please set DOUBAO_API_KEY in .env file or as environment variable")
            elif provider == "openrouter":
                print("Please set OPENROUTER_API_KEY in .env file or as environment variable")
            else:
                print(f"Unknown provider: {provider}")
            return False
        
        if cls.ENABLE_HISTORY_SEARCH and not cls.DIFY_API_KEY:
            print("WARNING: History search enabled but DIFY_API_KEY not set")
            print("History search will be disabled")
            cls.ENABLE_HISTORY_SEARCH = False
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.MEMORY_STORAGE_DIR, exist_ok=True)
        os.makedirs(cls.CONVERSATION_HISTORY_DIR, exist_ok=True)
        os.makedirs(cls.LOCOMO_OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.LOG_FILE) or "logs", exist_ok=True)
    
    @classmethod
    def setup_logging(cls):
        """Configure root logging from LOG_LEVEL / LOG_FORMAT / LOG_FILE.

        Modules call this instead of logging.basicConfig() so those settings
        are actually honored. Only the first call configures handlers.
        """
        global _LOGGING_CONFIGURED
        if _LOGGING_CONFIGURED:
            return
        level = getattr(logging, str(cls.LOG_LEVEL).upper(), logging.INFO)
        handlers = [logging.StreamHandler()]
        if cls.LOG_FILE:
            try:
                log_dir = os.path.dirname(cls.LOG_FILE)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                handlers.append(logging.FileHandler(cls.LOG_FILE, encoding="utf-8"))
            except OSError as e:
                print(f"WARNING: could not open log file '{cls.LOG_FILE}': {e}")
        logging.basicConfig(level=level, format=cls.LOG_FORMAT, handlers=handlers, force=True)
        _LOGGING_CONFIGURED = True

    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """
        Get model configuration as dictionary

        Returns:
            Model configuration dict
        """
        return {
            "model": cls.MODEL_NAME,
            "temperature": _reasoning_safe_temperature(cls.MODEL_NAME, cls.MODEL_TEMPERATURE),
            "max_tokens": cls.MODEL_MAX_TOKENS
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("\n" + "="*50)
        print("USER MEMORY SYSTEM CONFIGURATION")
        print("="*50)
        print(f"Provider: {cls.PROVIDER}")
        print(f"Model: {cls.MODEL_NAME or '(using provider default)'}")
        print(f"Memory Mode: {cls.MEMORY_MODE.value}")
        print(f"Max Memory Items: {cls.MAX_MEMORY_ITEMS}")
        print(f"History Search: {'Enabled' if cls.ENABLE_HISTORY_SEARCH else 'Disabled'}")
        
        # Show which API keys are set
        print("\nAPI Keys:")
        print(f"  OpenAI: {'✓ Set' if cls.OPENAI_API_KEY else '✗ Not set'}")
        print(f"  SiliconFlow: {'✓ Set' if cls.SILICONFLOW_API_KEY else '✗ Not set'}")
        print(f"  Doubao: {'✓ Set' if cls.DOUBAO_API_KEY else '✗ Not set'}")
        print(f"  OpenRouter: {'✓ Set' if cls.OPENROUTER_API_KEY else '✗ Not set'}")
        print(f"  Dify: {'✓ Set' if cls.DIFY_API_KEY else '✗ Not set'}")
        
        print(f"\nLog Level: {cls.LOG_LEVEL}")
        print("="*50 + "\n")
