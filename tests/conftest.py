"""
Pytest configuration and fixtures.
"""

import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_reddit_config():
    """Provide sample Reddit configuration."""
    return {
        'client_id': 'test_client_id',
        'client_secret': 'test_client_secret',
        'user_agent': 'ChamberCheck/0.1 Test'
    }


@pytest.fixture
def sample_date_range():
    """Provide sample date range for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date, end_date


@pytest.fixture
def sample_post_data():
    """Provide sample post data."""
    return {
        'post_id': 'test123',
        'platform': 'reddit',
        'community': 'test',
        'title': 'Test Post Title',
        'content': 'This is test content for a post.',
        'author': 'test_user',
        'created_at': datetime.now(),
        'upvotes': 100,
        'downvotes': 10,
        'num_comments': 25,
        'url': 'https://reddit.com/r/test/comments/test123',
        'metadata': {
            'upvote_ratio': 0.9,
            'is_self': True,
            'flair': 'Discussion'
        }
    }


@pytest.fixture
def sample_comment_data():
    """Provide sample comment data."""
    return {
        'comment_id': 'c123',
        'post_id': 'test123',
        'platform': 'reddit',
        'content': 'This is a test comment.',
        'author': 'commenter',
        'created_at': datetime.now(),
        'upvotes': 15,
        'downvotes': 2,
        'parent_id': None,
        'depth': 0,
        'metadata': {
            'is_submitter': False,
            'stickied': False
        }
    }
