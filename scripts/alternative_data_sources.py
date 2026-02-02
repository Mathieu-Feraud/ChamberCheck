"""
Alternative data sources for ChamberCheck when Reddit API is unavailable.

This document describes alternatives to Reddit's official API.
"""

# ALTERNATIVE DATA SOURCES FOR REDDIT DATA

## 1. Pushshift API (Historical Data)
# Pushshift archives Reddit data and provides a free API
# Note: As of 2024, Pushshift API access is limited

from datetime import datetime
import requests

def fetch_from_pushshift(subreddit, limit=100):
    """
    Fetch historical Reddit data from Pushshift.
    
    Note: Pushshift may have access limitations. Check status at:
    https://pushshift.io/
    """
    url = "https://api.pushshift.io/reddit/search/submission/"
    params = {
        'subreddit': subreddit,
        'size': limit,
        'sort': 'desc',
        'sort_type': 'score'
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"Pushshift API returned status: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error accessing Pushshift: {e}")
        return []


## 2. Academic Reddit Datasets
"""
Several academic datasets are available:

1. Reddit Comments Dataset (2015-2023)
   - Available on archive.org and academic data repositories
   - Compressed JSON files by month/year

2. Kaggle Datasets
   - Various Reddit datasets available on Kaggle
   - Search: "reddit comments dataset"

3. Stanford SNAP
   - Reddit hyperlinks dataset
   - Reddit submissions dataset
"""


## 3. Using Reddit's RSS Feeds (No API Required)
import feedparser

def fetch_from_rss(subreddit, sort='hot'):
    """
    Fetch recent posts using Reddit's RSS feeds (no authentication needed).
    
    Limitations:
    - Only recent posts (typically last 25-100)
    - No comment data
    - Limited metadata
    """
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.rss"
    
    try:
        feed = feedparser.parse(url)
        posts = []
        
        for entry in feed.entries:
            post_data = {
                'title': entry.title,
                'url': entry.link,
                'published': entry.published,
                'author': entry.author if hasattr(entry, 'author') else 'unknown',
                'summary': entry.summary if hasattr(entry, 'summary') else ''
            }
            posts.append(post_data)
        
        return posts
    except Exception as e:
        print(f"Error fetching RSS: {e}")
        return []


## 4. Web Scraping (Last Resort)
"""
Web scraping Reddit directly is:
- Against Reddit's Terms of Service for automated collection
- Unreliable (HTML structure changes)
- May result in IP bans

Only use for:
- Personal, small-scale research
- With proper rate limiting
- As absolute last resort

Consider using libraries like:
- praw (official, but requires API access)
- PSAW (Pushshift wrapper)
- Manual RSS parsing (shown above)
"""


## RECOMMENDED APPROACH FOR TESTING
"""
For immediate testing without Reddit API:

1. Use the mock data generator (scripts/test_mock_scraper.py)
2. Try RSS feeds for real (but limited) Reddit data
3. Download academic datasets for historical analysis
4. Apply for Reddit API access for production use

For Research/Academic Use:
- Contact Reddit for academic API access
- Use archived datasets from academic repositories
- Cite data sources properly in publications
"""


if __name__ == '__main__':
    print("Alternative Data Sources for ChamberCheck")
    print("=" * 60)
    
    # Test RSS feed (no authentication needed)
    print("\nTesting Reddit RSS feed access...")
    posts = fetch_from_rss('python', sort='hot')
    
    if posts:
        print(f"✓ Successfully fetched {len(posts)} posts via RSS")
        print(f"\nSample post:")
        print(f"Title: {posts[0]['title']}")
        print(f"URL: {posts[0]['url']}")
    else:
        print("✗ Unable to fetch RSS feed")
    
    print("\n" + "=" * 60)
    print("See script comments for more data source options")
