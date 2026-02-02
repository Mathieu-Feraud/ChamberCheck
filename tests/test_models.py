"""
Unit tests for data models.
"""

import pytest
from datetime import datetime
from chambercheck.models import Post, Comment, AnalysisResult


class TestPost:
    """Test suite for Post model."""
    
    def test_post_creation(self):
        """Test creating a Post instance."""
        post = Post(
            post_id='test123',
            platform='reddit',
            community='test',
            title='Test Post',
            content='Test content',
            author='test_user',
            created_at=datetime.now(),
            upvotes=100,
            num_comments=10
        )
        assert post.post_id == 'test123'
        assert post.platform == 'reddit'
    
    def test_post_to_dict(self):
        """Test converting Post to dictionary."""
        now = datetime.now()
        post = Post(
            post_id='test123',
            platform='reddit',
            community='test',
            title='Test',
            content='Content',
            author='user',
            created_at=now,
            upvotes=100,
            num_comments=10
        )
        post_dict = post.to_dict()
        assert post_dict['post_id'] == 'test123'
        assert 'created_at' in post_dict
    
    def test_engagement_score(self):
        """Test engagement score calculation."""
        post = Post(
            post_id='test',
            platform='reddit',
            community='test',
            title='Test',
            content='Content',
            author='user',
            created_at=datetime.now(),
            upvotes=100,
            num_comments=10
        )
        score = post.get_engagement_score(comment_weight=2.0)
        assert score == 120  # 100 + (10 * 2)


class TestComment:
    """Test suite for Comment model."""
    
    def test_comment_creation(self):
        """Test creating a Comment instance."""
        comment = Comment(
            comment_id='c123',
            post_id='p123',
            platform='reddit',
            content='Test comment',
            author='user',
            created_at=datetime.now(),
            upvotes=10
        )
        assert comment.comment_id == 'c123'
        assert comment.post_id == 'p123'
    
    def test_is_top_level(self):
        """Test checking if comment is top-level."""
        top_level = Comment(
            comment_id='c1',
            post_id='p1',
            platform='reddit',
            content='Top',
            author='user',
            created_at=datetime.now(),
            upvotes=5,
            depth=0
        )
        assert top_level.is_top_level() == True
        
        reply = Comment(
            comment_id='c2',
            post_id='p1',
            platform='reddit',
            content='Reply',
            author='user2',
            created_at=datetime.now(),
            upvotes=3,
            parent_id='c1',
            depth=1
        )
        assert reply.is_top_level() == False


if __name__ == '__main__':
    pytest.main([__file__])
