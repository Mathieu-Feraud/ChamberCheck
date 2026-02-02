"""
Configuration management for ChamberCheck.

Handles loading and validation of application configuration.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Central configuration manager for ChamberCheck.
    
    Loads configuration from environment variables or config files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to JSON config file
        """
        self.config_data = {}
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self.load_from_env()
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration JSON file
        """
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        self.config_data = {
            'reddit': {
                'client_id': os.getenv('REDDIT_CLIENT_ID', ''),
                'client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
                'user_agent': os.getenv('REDDIT_USER_AGENT', 'ChamberCheck/0.1')
            },
            'facebook': {
                'access_token': os.getenv('FB_ACCESS_TOKEN', ''),
                'app_id': os.getenv('FB_APP_ID', ''),
                'app_secret': os.getenv('FB_APP_SECRET', '')
            },
            'llm': {
                'provider': os.getenv('LLM_PROVIDER', 'openai'),
                'api_key': os.getenv('LLM_API_KEY', ''),
                'model': os.getenv('LLM_MODEL', 'gpt-4')
            },
            'data': {
                'raw_data_dir': os.getenv('RAW_DATA_DIR', 'data/raw'),
                'processed_data_dir': os.getenv('PROCESSED_DATA_DIR', 'data/processed'),
                'output_dir': os.getenv('OUTPUT_DIR', 'data/output')
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'reddit.client_id')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_scraper_config(self, platform: str) -> Dict[str, Any]:
        """
        Get scraper-specific configuration.
        
        Args:
            platform: Platform name ('reddit', 'facebook', etc.)
            
        Returns:
            Dictionary with scraper configuration
        """
        return self.config_data.get(platform, {})
    
    def save_to_file(self, config_path: str):
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
    
    def validate_scraper_config(self, platform: str) -> bool:
        """
        Validate that required configuration exists for a platform.
        
        Args:
            platform: Platform name
            
        Returns:
            bool: True if configuration is valid
        """
        platform_config = self.config_data.get(platform, {})
        
        if platform == 'reddit':
            required = ['client_id', 'client_secret', 'user_agent']
        elif platform == 'facebook':
            required = ['access_token', 'app_id', 'app_secret']
        else:
            return False
        
        return all(platform_config.get(field) for field in required)
