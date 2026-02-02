"""
Validation utilities for ChamberCheck.
"""

import re
from datetime import datetime
from typing import Tuple


def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """
    Validate that date range is valid.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If date range is invalid
    """
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    
    if end_date > datetime.now():
        raise ValueError("End date cannot be in the future")
    
    return True


def validate_subreddit_name(subreddit: str) -> bool:
    """
    Validate subreddit name format.
    
    Args:
        subreddit: Subreddit name
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If subreddit name is invalid
    """
    # Remove r/ prefix if present
    subreddit = subreddit.replace('r/', '')
    
    # Reddit subreddit name rules:
    # - 3-21 characters
    # - Only letters, numbers, and underscores
    # - Must start with a letter
    pattern = r'^[a-zA-Z][a-zA-Z0-9_]{2,20}$'
    
    if not re.match(pattern, subreddit):
        raise ValueError(
            f"Invalid subreddit name: {subreddit}. "
            "Must be 3-21 characters, start with a letter, "
            "and contain only letters, numbers, and underscores."
        )
    
    return True


def validate_keywords(keywords: list) -> bool:
    """
    Validate keywords list.
    
    Args:
        keywords: List of keywords
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If keywords are invalid
    """
    if not isinstance(keywords, list):
        raise ValueError("Keywords must be a list")
    
    if not keywords:
        raise ValueError("Keywords list cannot be empty")
    
    for keyword in keywords:
        if not isinstance(keyword, str):
            raise ValueError("All keywords must be strings")
        if not keyword.strip():
            raise ValueError("Keywords cannot be empty strings")
    
    return True
