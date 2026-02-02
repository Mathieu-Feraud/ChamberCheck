"""
Base scraper class for all platform-specific scrapers.

This abstract class defines the interface that all scrapers must implement,
ensuring consistent behavior across different social media platforms.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional, Any
from ..models import Post, Comment


class BaseScraper(ABC):
    """
    Abstract base class for social media platform scrapers.
    
    All platform-specific scrapers (Reddit, Facebook, etc.) should inherit
    from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the scraper with configuration.
        
        Args:
            config: Dictionary containing scraper-specific configuration
                   (API keys, rate limits, etc.)
        """
        self.config = config
        self.platform_name = "base"
        
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the platform's API.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        pass
    
    @abstractmethod
    def fetch_posts(
        self,
        community: str,
        start_date: datetime,
        end_date: datetime,
        keywords: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Post]:
        """
        Fetch posts from a specific community within a time range.
        
        Args:
            community: Community identifier (e.g., subreddit name)
            start_date: Start of time range
            end_date: End of time range
            keywords: Optional list of keywords to filter posts
            limit: Optional maximum number of posts to fetch
            
        Returns:
            List of Post objects
        """
        pass
    
    @abstractmethod
    def fetch_comments(
        self,
        post_ids: List[str],
        limit: Optional[int] = None
    ) -> List[Comment]:
        """
        Fetch comments for given post IDs.
        
        Args:
            post_ids: List of post identifiers
            limit: Optional maximum number of comments per post
            
        Returns:
            List of Comment objects
        """
        pass
    
    @abstractmethod
    def fetch_post_metadata(self, post_id: str) -> Dict[str, Any]:
        """
        Fetch metadata for a specific post.
        
        Args:
            post_id: Post identifier
            
        Returns:
            Dictionary containing post metadata (scores, timestamps, etc.)
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate that required configuration parameters are present.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If required configuration is missing
        """
        required_fields = self.get_required_config_fields()
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
        return True
    
    @abstractmethod
    def get_required_config_fields(self) -> List[str]:
        """
        Get list of required configuration fields for this scraper.
        
        Returns:
            List of required field names
        """
        pass
    
    def save_data(self, data: List[Any], output_path: str, format: str = "json"):
        """
        Save scraped data to file.
        
        Args:
            data: List of data objects to save
            output_path: Path to output file
            format: Output format ('json' or 'csv')
        """
        # Implementation will be in utils
        pass
