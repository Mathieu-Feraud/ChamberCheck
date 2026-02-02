"""
OpenAI LLM provider implementation.
"""

import os
import json
from typing import Dict, Any

from .llm_provider import LLMProvider
from ..utils import setup_logger
from ..constants import DEFAULT_OPENAI_MODEL


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for comment analysis."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (defaults to DEFAULT_OPENAI_MODEL)
        """
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
        
        super().__init__(api_key)
        self.model = model or DEFAULT_OPENAI_MODEL
        self.logger = setup_logger("OpenAIProvider")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def analyze_comment(self, prompt: str) -> Dict[str, Any]:
        """
        Send prompt to OpenAI and get analysis with prompt caching.
        
        Args:
            prompt: Full analysis prompt with comment context
        
        Returns:
            Dictionary with scores
        """
        try:
            from ..constants import ANALYSIS_INSTRUCTIONS_PROMPT, LLM_TEMPERATURE, LLM_MAX_TOKENS
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": ANALYSIS_INSTRUCTIONS_PROMPT,
                                "cache_control": {"type": "ephemeral"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            
            response_text = response.choices[0].message.content
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
            self.logger.error(f"OpenAI API error: {type(e).__name__}: {e}")
            raise
