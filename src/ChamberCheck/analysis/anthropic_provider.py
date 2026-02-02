"""
Anthropic LLM provider implementation.
"""

import os
import json
from typing import Dict, Any

from .llm_provider import LLMProvider
from ..utils import setup_logger
from ..constants import DEFAULT_ANTHROPIC_MODEL


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for comment analysis."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name (defaults to DEFAULT_ANTHROPIC_MODEL)
        """
        if not api_key:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
        
        super().__init__(api_key)
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.logger = setup_logger("AnthropicProvider")
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def analyze_comment(self, prompt: str) -> Dict[str, Any]:
        """
        Send prompt to Anthropic and get analysis with prompt caching.
        
        Args:
            prompt: Full analysis prompt with comment context
        
        Returns:
            Dictionary with scores
        """
        try:
            from ..constants import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_INSTRUCTIONS_PROMPT, LLM_MAX_TOKENS
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=LLM_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": ANALYSIS_SYSTEM_PROMPT
                    },
                    {
                        "type": "text",
                        "text": ANALYSIS_INSTRUCTIONS_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response_text = response.content[0].text
            self.logger.debug(f"Raw API response: {response_text}")
            result = json.loads(response_text)
            
            # Log cache usage for monitoring
            usage = response.usage
            if hasattr(usage, 'cache_creation_input_tokens') and usage.cache_creation_input_tokens:
                self.logger.info(f"Cache created: {usage.cache_creation_input_tokens} tokens")
            if hasattr(usage, 'cache_read_input_tokens') and usage.cache_read_input_tokens:
                self.logger.info(f"Cache hit: {usage.cache_read_input_tokens} tokens saved")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}. Response was: {response_text if 'response_text' in locals() else 'N/A'}")
            raise
        except Exception as e:
            self.logger.error(f"Anthropic API error: {type(e).__name__}: {e}")
            raise
