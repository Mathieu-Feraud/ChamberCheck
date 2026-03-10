"""
Comment analyzer using LLM providers.

Analyzes comments for discourse dynamics and echo chamber metrics.
"""

import json
from typing import List, Dict, Any, Optional

from .llm_provider import LLMProvider
from ..models import Comment
from ..utils import setup_logger
from .prompt_builder import build_comment_prompt


class CommentAnalyzer:
    """Analyzes comments using LLM for discourse dynamics metrics."""

    def __init__(self, provider: Optional[LLMProvider], subreddit: str = "samharris"):
        """
        Initialize analyzer.
        
        Args:
            provider: LLMProvider instance (optional for prompt-only use)
            subreddit: Subreddit context for analysis
        """
        self.provider = provider
        self.subreddit = subreddit
        self.logger = setup_logger("CommentAnalyzer")
    
    def build_prompt(
        self,
        comment: Dict[str, Any],
        parent: Optional[Dict[str, Any]] = None,
        parent_is_post: bool = False,
        source_file: Optional[str] = None,
        posts_map: Optional[Dict[str, Dict[str, Any]]] = None,
        comments_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Build analysis prompt for a comment."""
        return build_comment_prompt(
            comment=comment,
            parent=parent,
            parent_is_post=parent_is_post,
            source_file=source_file,
            posts_map=posts_map,
            comments_map=comments_map,
        )
    
    def analyze(
        self,
        comment: Dict[str, Any],
        parent: Optional[Dict[str, Any]] = None,
        parent_is_post: bool = False,
        source_file: Optional[str] = None,
        posts_map: Optional[Dict[str, Dict[str, Any]]] = None,
        comments_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single comment.
        
        Args:
            comment: Comment data
            parent: Optional parent comment or post context
            parent_is_post: Whether parent is a post (True) or comment (False)
        
        Returns:
            Analysis results with discourse dynamics scores
        """
        if not self.provider:
            raise ValueError("LLM provider is not configured for analysis")

        prompt = self.build_prompt(
            comment,
            parent,
            parent_is_post,
            source_file=source_file,
            posts_map=posts_map,
            comments_map=comments_map,
        )
        # analyze_comment uses the structured system prompt with prompt caching
        # (cache_control headers on the system block) — do not use analyze_with_text
        analysis = self.provider.analyze_comment(prompt)

        # Add comment_id to result
        analysis['comment_id'] = comment.get('comment_id', 'UNKNOWN')
        
        return analysis
    
    def analyze_batch(
        self,
        comments: List[Dict[str, Any]],
        parent_map: Dict[str, Dict] = None,
        posts_map: Dict[str, Dict] = None,
        source_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple comments.
        
        Args:
            comments: List of comment data
            parent_map: Mapping of comment_id to parent comment (optional)
            posts_map: Mapping of post_id to post data (optional)
        
        Returns:
            List of analysis results
        """
        results = []
        parent_map = parent_map or {}
        posts_map = posts_map or {}
        
        for i, comment in enumerate(comments, 1):
            try:
                self.logger.info(f"Analyzing comment {i}/{len(comments)}: {comment.get('comment_id')}")
                
                parent = None
                parent_is_post = False
                
                # First try to find parent comment
                if comment.get('parent_id') and comment['parent_id'] in parent_map:
                    parent = parent_map[comment['parent_id']]
                    parent_is_post = False
                # Then try to find parent post
                elif comment.get('post_id') and comment['post_id'] in posts_map:
                    parent = posts_map[comment['post_id']]
                    parent_is_post = True
                
                result = self.analyze(
                    comment,
                    parent,
                    parent_is_post,
                    source_file=source_file,
                    posts_map=posts_map,
                    comments_map=parent_map,
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error analyzing comment {comment.get('comment_id')}: {e}")
                # Add error result
                results.append({
                    "comment_id": comment.get('comment_id'),
                    "error": str(e)
                })
        
        return results
