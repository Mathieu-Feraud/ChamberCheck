"""
Scraper modules for different social media platforms.

This package provides a base scraper interface and platform-specific
implementations for collecting discourse data.
"""

from .base_scraper import BaseScraper
from .reddit_json_scraper import RedditJSONScraper
from .batch_scraper import batch_scrape, batch_scrape_posts_only, scrape_subreddit
from .comment_scraper import scrape_comments

__all__ = [
    "BaseScraper",
    "RedditJSONScraper",
    "batch_scrape",
    "batch_scrape_posts_only",
    "scrape_subreddit",
    "scrape_comments",
]
