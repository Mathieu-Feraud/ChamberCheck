"""
Comment analyzer using LLM providers.

Analyzes comments for echo chamber metrics.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from .llm_provider import LLMProvider
from ..models import Comment
from ..utils import setup_logger
from ..constants import ANALYSIS_INSTRUCTIONS_PROMPT


class CommentAnalyzer:
    """Analyzes comments using LLM for echo chamber metrics."""
    
    def __init__(self, provider: LLMProvider, subreddit: str = "samharris"):
        """
        Initialize analyzer.
        
        Args:
            provider: LLMProvider instance
            subreddit: Subreddit context for analysis
        """
        self.provider = provider
        self.subreddit = subreddit
        self.logger = setup_logger("CommentAnalyzer")
    
    def build_prompt(self, comment: Dict[str, Any], parent_comment: Optional[Dict[str, Any]] = None) -> str:
        """
        Build analysis prompt for a comment.
        
        Args:
            comment: Comment data (from JSON)
            parent_comment: Optional parent comment (for context)
        
        Returns:
            Full prompt for LLM
        """
        prompt = ANALYSIS_INSTRUCTIONS_PROMPT + f"""

Subreddit context: r/{self.subreddit} (philosophy, neuroscience, politics, morality discussion)
Comment depth: {comment.get('depth', 0)}
Comment text: {comment.get('content', '')}
"""
        
        if parent_comment:
            prompt += f"\n[PARENT COMMENT CONTEXT (depth {parent_comment.get('depth', 0)}):\n"
            prompt += f"Author: u/{parent_comment.get('author', '[deleted]')}\n"
            prompt += f"Score: {parent_comment.get('metadata', {}).get('score', 0)} upvotes\n"
            prompt += f"Text: {parent_comment.get('content', '')}]\n"
        
        prompt += """
[COMMENT TO ANALYZE (depth {comment.get('depth', 0)}):\n"""
        prompt += f"Author: u/{comment.get('author', '[deleted]')}\n"
        
        prompt += """
Return JSON only (no markdown, no code blocks):
{
  "argument_narrowness": 0-10,
  "hostility": 0-10,
  "suppression": 0-10,
  "epistemic_closure": 0-10,
  "argument_avoidance": "0-10 or N/A",
  "echo_chamber_score": 0-50,
  "reasoning": "brief explanation"
}"""
        
        return prompt
    
    def analyze(self, comment: Dict[str, Any], parent_comment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a single comment.
        
        Args:
            comment: Comment data
            parent_comment: Optional parent comment context
        
        Returns:
            Analysis results with scores
        """
        prompt = self.build_prompt(comment, parent_comment)
        result = self.provider.analyze_comment(prompt)
        
        # Add comment_id back to result (not provided to LLM to avoid bias)
        if 'error' not in result:
            result['comment_id'] = comment.get('comment_id', 'UNKNOWN')
        
        return result
    
    def analyze_batch(self, comments: List[Dict[str, Any]], parent_map: Dict[str, Dict] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple comments.
        
        Args:
            comments: List of comment data
            parent_map: Mapping of comment_id to parent comment (optional)
        
        Returns:
            List of analysis results
        """
        results = []
        parent_map = parent_map or {}
        
        for i, comment in enumerate(comments, 1):
            try:
                self.logger.info(f"Analyzing comment {i}/{len(comments)}: {comment.get('comment_id')}")
                
                parent = None
                if comment.get('parent_id') and comment['parent_id'] in parent_map:
                    parent = parent_map[comment['parent_id']]
                
                result = self.analyze(comment, parent)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error analyzing comment {comment.get('comment_id')}: {e}")
                # Add error result
                results.append({
                    "comment_id": comment.get('comment_id'),
                    "error": str(e)
                })
        
        return results
