"""
Unit tests for Reddit scraper.
"""

import pytest
from datetime import datetime, timedelta
from chambercheck.scrapers import RedditScraper
from chambercheck import Config


class TestRedditScraper:
    """Test suite for RedditScraper."""
    
    def test_scraper_initialization(self):
        """Test that scraper initializes correctly."""
        config = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'user_agent': 'test_agent'
        }
        scraper = RedditScraper(config)
        assert scraper.platform_name == 'reddit'
        assert scraper.config == config
    
    def test_required_config_fields(self):
        """Test that required config fields are correct."""
        config = {'client_id': 'test'}
        scraper = RedditScraper(config)
        required = scraper.get_required_config_fields()
        assert 'client_id' in required
        assert 'client_secret' in required
        assert 'user_agent' in required
    
    def test_validate_config_missing_fields(self):
        """Test that validation fails with missing fields."""
        config = {'client_id': 'test'}
        scraper = RedditScraper(config)
        with pytest.raises(ValueError):
            scraper.validate_config()
    
    def test_validate_config_success(self):
        """Test that validation succeeds with all fields."""
        config = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'user_agent': 'test_agent'
        }
        scraper = RedditScraper(config)
        assert scraper.validate_config() == True


if __name__ == '__main__':
    pytest.main([__file__])
