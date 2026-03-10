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
        self.last_response_model = None
        self.last_response_usage = None
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
        self.last_response_model = None
        self.last_response_usage = None
        try:
            from ..constants import ANALYSIS_INSTRUCTIONS_PROMPT, LLM_TEMPERATURE, LLM_MAX_TOKENS
            
            response = self._create_chat_completion(
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
                max_tokens=LLM_MAX_TOKENS,
            )

            self._record_response_metadata(response)
            
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
    
    def analyze_with_text(self, prompt: str) -> str:
        """
        Analyze plain text with the OpenAI model.
        
        Args:
            prompt: Text prompt to send to the model
        
        Returns:
            String response from the model
        """
        self.last_response_model = None
        self.last_response_usage = None
        try:
            from ..constants import LLM_TEMPERATURE, LLM_MAX_TOKENS
            
            response = self._create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )

            self._record_response_metadata(response)
            
            response_text = response.choices[0].message.content
            self.logger.debug(f"Text API response: {response_text[:200]}...")
            return response_text
        except Exception as e:
            self.logger.error(f"OpenAI Text API error: {type(e).__name__}: {e}")
            raise

    def analyze_with_vision(self, prompt: str, image_url: str) -> str:
        """
        Analyze an image using OpenAI's vision capabilities.
        
        Args:
            prompt: Analysis prompt
            image_url: URL of the image to analyze
        
        Returns:
            String response from the model
        """
        self.last_response_model = None
        self.last_response_usage = None
        try:
            from ..constants import LLM_TEMPERATURE, LLM_MAX_TOKENS
            
            response = self._create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )

            self._record_response_metadata(response)
            
            response_text = response.choices[0].message.content
            self.logger.debug(f"Vision API response: {response_text[:200]}...")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"OpenAI Vision API error: {type(e).__name__}: {e}")
            raise

    def _create_chat_completion(self, messages: list, temperature: float, max_tokens: int):
        request = {
            "model": self.model,
            "messages": messages,
        }

        model_lower = (self.model or "").lower()
        is_reasoning_model = (
            model_lower.startswith("gpt-5")
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
        )

        if is_reasoning_model:
            request["max_completion_tokens"] = max(max_tokens, 1200)
            request["reasoning_effort"] = "medium" if model_lower == "gpt-5.2" else "minimal"
        else:
            request["temperature"] = temperature
            request["max_tokens"] = max_tokens

        try:
            return self.client.chat.completions.create(**request)
        except Exception as e:
            message = str(e)

            if "reasoning_effort" in message and "unsupported" in message.lower():
                fallback_request = dict(request)
                fallback_request["reasoning_effort"] = "low"
                try:
                    return self.client.chat.completions.create(**fallback_request)
                except Exception as second_error:
                    second_message = str(second_error)
                    if "reasoning_effort" in second_message and "unsupported" in second_message.lower():
                        fallback_request["reasoning_effort"] = "none"
                        return self.client.chat.completions.create(**fallback_request)
                    raise

            if "max_tokens" in message or "max_completion_tokens" in message:
                request.pop("max_tokens", None)
                request["max_completion_tokens"] = max_tokens
                try:
                    return self.client.chat.completions.create(**request)
                except Exception as second_error:
                    message = str(second_error)
                    if "temperature" not in message:
                        raise

            if "temperature" in message and "unsupported" in message.lower():
                request.pop("temperature", None)
                return self.client.chat.completions.create(**request)

            raise

    def _record_response_metadata(self, response: Any) -> None:
        self.last_response_model = getattr(response, "model", None) or self.model
        usage = getattr(response, "usage", None)
        if not usage:
            self.last_response_usage = None
            return

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        self.last_response_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


