"""
Data model for social media posts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Post:
    """
    Represents a social media post.
    
    Attributes:
        post_id: Unique identifier for the post
        platform: Platform name ('reddit', 'facebook', etc.)
        community: Community identifier (subreddit, group, etc.)
        title: Post title
        content: Post text content
        author: Author username/identifier
        created_at: Post creation timestamp
        upvotes: Number of upvotes/likes
        downvotes: Number of downvotes (if available)
        num_comments: Number of comments
        url: URL to the post
        metadata: Additional platform-specific metadata
    """
    
    post_id: str
    platform: str
    community: str
    title: str
    content: str
    author: str
    created_at: datetime
    upvotes: int
    downvotes: Optional[int] = None
    num_comments: int = 0
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert post to dictionary."""
        return {
            'post_id': self.post_id,
            'platform': self.platform,
            'community': self.community,
            'title': self.title,
            'content': self.content,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'upvotes': self.upvotes,
            'downvotes': self.downvotes,
            'num_comments': self.num_comments,
            'url': self.url,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Post':
        """Create Post from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def get_engagement_score(self, comment_weight: float = 2.0) -> float:
        """
        Calculate engagement score.
        
        Args:
            comment_weight: Weight for comment count
            
        Returns:
            Engagement score
        """
        score = self.upvotes
        if self.downvotes:
            score -= self.downvotes
        score += self.num_comments * comment_weight
        return score
