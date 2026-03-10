"""
Analysis modules for discourse metrics.

Includes LLM-based comment analysis and echo chamber scoring.
"""

from .comment_analyzer import CommentAnalyzer
from .llm_provider import LLMProvider, NonRetryableError
from .openai_provider import OpenAIProvider
from .batch_analyzer import batch_analyze_comments
from .prompt_exporter import export_comment_prompts
from .post_analyzer import analyze_posts
from .analyze_comments import run_comment_analysis


def generate_abn_test_set(*args, **kwargs):
	from ..model_analysis.abn_test import generate_abn_test_set as _generate_abn_test_set
	return _generate_abn_test_set(*args, **kwargs)


def extract_abn_user_entries(*args, **kwargs):
	from ..model_analysis.abn_test import extract_abn_user_entries as _extract_abn_user_entries
	return _extract_abn_user_entries(*args, **kwargs)


def run_abn_llm_analysis(*args, **kwargs):
	from ..model_analysis.abn_test import run_abn_llm_analysis as _run_abn_llm_analysis
	return _run_abn_llm_analysis(*args, **kwargs)

__all__ = [
	"CommentAnalyzer",
	"LLMProvider",
	"analyze_posts",
	"OpenAIProvider",
	"batch_analyze_comments",
	"export_comment_prompts",
	"generate_abn_test_set",
	"extract_abn_user_entries",
	"run_abn_llm_analysis",
	"run_comment_analysis",
]
