"""
Abstract base class for LLM providers.

Allows pluggable LLM implementations (OpenAI, Anthropic, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from ..constants import DEFAULT_OPENAI_MODEL, DEFAULT_ANTHROPIC_MODEL


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str):
        """
        Initialize LLM provider.
        
        Args:
            api_key: API key for the provider
        """
        self.api_key = api_key
    
    @abstractmethod
    def analyze_comment(self, prompt: str) -> Dict[str, Any]:
        """
        Send prompt to LLM and get analysis.
        
        Args:
            prompt: Full analysis prompt with comment context
        
        Returns:
            Dictionary with scores (parsed from LLM response)
        """
        pass
    
    @classmethod
    def from_config(cls, provider: str, api_key: str = None, model: str = None) -> 'LLMProvider':
        """
        Factory method to create provider from config.
        
        Args:
            provider: 'openai' or 'anthropic'
            api_key: API key (if not provided, loaded from env)
            model: Model name (uses default if not provided)
        
        Returns:
            LLMProvider instance
        """
        if provider.lower() == 'openai':
            from .openai_provider import OpenAIProvider
            return OpenAIProvider(api_key, model=model)
        elif provider.lower() == 'anthropic':
            from .anthropic_provider import AnthropicProvider
            return AnthropicProvider(api_key, model=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
