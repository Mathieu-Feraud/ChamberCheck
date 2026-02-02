"""
ChamberCheck: A tool to analyze echo chamber dynamics in online communities.

This package provides modular components for scraping, analyzing, and scoring
discourse patterns across social media platforms like Reddit, Facebook, etc.
"""

__version__ = "0.1.0"
__author__ = "ChamberCheck Team"
__license__ = "MIT"

from .config import Config

__all__ = ["Config"]
