"""
Analysis modules for discourse metrics.

Includes LLM-based comment analysis and echo chamber scoring.
"""

from .comment_analyzer import CommentAnalyzer
from .llm_provider import LLMProvider

__all__ = ["CommentAnalyzer", "LLMProvider"]
