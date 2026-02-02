"""
Scraper modules for different social media platforms.

This package provides a base scraper interface and platform-specific
implementations for collecting discourse data.
"""

from .base_scraper import BaseScraper
from .reddit_json_scraper import RedditJSONScraper

__all__ = ["BaseScraper", "RedditJSONScraper"]
