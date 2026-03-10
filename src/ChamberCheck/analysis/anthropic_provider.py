"""
Anthropic LLM provider implementation.
"""

import os
import json
import re
from typing import Dict, Any

from .llm_provider import LLMProvider, NonRetryableError
from ..utils import setup_logger
from ..constants import DEFAULT_ANTHROPIC_MODEL


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for comment analysis."""
    
    def __init__(self, api_key: str = None, model: str = None, include_rationale: bool = True):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name (defaults to DEFAULT_ANTHROPIC_MODEL)
            include_rationale: When False, the no-rationale prompt variant is used,
                reducing output tokens by ~25-30 % at the cost of losing the
                per-score rationale strings.
        """
        if not api_key:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
        
        super().__init__(api_key)
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.include_rationale = include_rationale
        self.last_response_model = None
        self.last_response_usage = None
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
        self.last_response_model = None
        self.last_response_usage = None
        try:
            from ..constants import (
                ANALYSIS_SYSTEM_PROMPT,
                COMMENT_ANALYSIS_STATIC_INSTRUCTIONS,
                COMMENT_ANALYSIS_STATIC_INSTRUCTIONS_NO_RATIONALE,
                LLM_MAX_TOKENS,
            )

            static_instructions = (
                COMMENT_ANALYSIS_STATIC_INSTRUCTIONS
                if self.include_rationale
                else COMMENT_ANALYSIS_STATIC_INSTRUCTIONS_NO_RATIONALE
            )

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
                        "text": static_instructions,
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

            self._record_response_metadata(response)
            
            response_text = response.content[0].text
            self.logger.debug(f"Raw API response: {response_text}")

            json_candidate = response_text.strip()
            if json_candidate.startswith("```"):
                json_candidate = re.sub(r"^```(?:json)?", "", json_candidate).strip()
                if json_candidate.endswith("```"):
                    json_candidate = json_candidate[:-3].strip()

            # Always extract first JSON object — strips any trailing text
            # (e.g. ```-fence remnants, "Reasoning:" blocks) after closing brace.
            start = json_candidate.find("{")
            end = json_candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_candidate = json_candidate[start:end + 1]

            result = json.loads(json_candidate)
            
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
            # Re-raise billing / auth errors as NonRetryableError so the
            # caller can abort immediately instead of burning retries.
            err_str = str(e)
            if (
                'credit balance is too low' in err_str
                or 'insufficient_quota' in err_str
                or 'access_denied' in err_str
                or ('400' in err_str and 'invalid_api_key' in err_str)
            ):
                raise NonRetryableError(err_str) from e
            self.logger.error(f"Anthropic API error: {type(e).__name__}: {e}")
            raise

    def analyze_with_text(self, prompt: str) -> str:
        """
        Analyze plain text with the Anthropic model.

        Args:
            prompt: Text prompt to send to the model

        Returns:
            String response from the model
        """
        self.last_response_model = None
        self.last_response_usage = None
        try:
            from ..constants import LLM_MAX_TOKENS

            response = self.client.messages.create(
                model=self.model,
                max_tokens=LLM_MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            self._record_response_metadata(response)

            response_text = response.content[0].text
            self.logger.debug(f"Text API response: {response_text[:200]}...")
            return response_text
        except Exception as e:
            self.logger.error(f"Anthropic Text API error: {type(e).__name__}: {e}")
            raise

    def _record_response_metadata(self, response: Any) -> None:
        self.last_response_model = getattr(response, "model", None) or self.model
        usage = getattr(response, "usage", None)
        if not usage:
            self.last_response_usage = None
            return

        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)

        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        self.last_response_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
