"""
Unit tests for configuration management.
"""

import pytest
import json
import tempfile
from pathlib import Path
from chambercheck import Config


class TestConfig:
    """Test suite for Config class."""
    
    def test_config_get_with_dot_notation(self):
        """Test getting config values with dot notation."""
        config = Config()
        config.config_data = {
            'reddit': {
                'client_id': 'test123'
            }
        }
        assert config.get('reddit.client_id') == 'test123'
    
    def test_config_get_default(self):
        """Test getting config with default value."""
        config = Config()
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_config_load_from_file(self):
        """Test loading config from JSON file."""
        # Create temporary config file
        config_data = {
            'reddit': {
                'client_id': 'file_id',
                'client_secret': 'file_secret',
                'user_agent': 'file_agent'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config(config_path=temp_path)
            assert config.get('reddit.client_id') == 'file_id'
        finally:
            Path(temp_path).unlink()
    
    def test_get_scraper_config(self):
        """Test getting scraper-specific config."""
        config = Config()
        config.config_data = {
            'reddit': {
                'client_id': 'test',
                'client_secret': 'secret'
            }
        }
        reddit_config = config.get_scraper_config('reddit')
        assert 'client_id' in reddit_config
        assert reddit_config['client_id'] == 'test'


if __name__ == '__main__':
    pytest.main([__file__])
