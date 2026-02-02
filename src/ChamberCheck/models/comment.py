"""
Data model for social media comments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Comment:
    """
    Represents a comment on a social media post.
    
    Attributes:
        comment_id: Unique identifier for the comment
        post_id: ID of the parent post
        platform: Platform name
        content: Comment text
        author: Author username/identifier
        created_at: Comment creation timestamp
        upvotes: Number of upvotes/likes
        downvotes: Number of downvotes (if available)
        parent_id: ID of parent comment (if this is a reply)
        depth: Depth in comment tree (0 = top-level)
        metadata: Additional platform-specific metadata
    """
    
    comment_id: str
    post_id: str
    platform: str
    content: str
    author: str
    created_at: datetime
    upvotes: int
    downvotes: Optional[int] = None
    parent_id: Optional[str] = None
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert comment to dictionary."""
        return {
            'comment_id': self.comment_id,
            'post_id': self.post_id,
            'platform': self.platform,
            'content': self.content,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'upvotes': self.upvotes,
            'downvotes': self.downvotes,
            'parent_id': self.parent_id,
            'depth': self.depth,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Comment':
        """Create Comment from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def is_top_level(self) -> bool:
        """Check if comment is top-level (direct reply to post)."""
        return self.depth == 0 or self.parent_id is None
    
    def get_score(self) -> int:
        """
        Calculate net score.
        
        Returns:
            Net score (upvotes - downvotes)
        """
        score = self.upvotes
        if self.downvotes:
            score -= self.downvotes
        return score
