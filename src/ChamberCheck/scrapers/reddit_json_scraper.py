"""
Reddit JSON scraper - no authentication required.

Uses Reddit's public JSON endpoints which don't require API credentials.
Just add .json to any Reddit URL to get structured data.
"""

import requests
import time
from datetime import datetime
from typing import List, Optional

from .base_scraper import BaseScraper
from ..models import Post, Comment
from ..utils import setup_logger


class RedditJSONScraper(BaseScraper):
    """
    Reddit scraper using public JSON endpoints (no API key needed).
    
    Rate limits: ~60 requests per 10 minutes per IP address.
    Good for: Testing, development, small datasets
    For production: Use RedditScraper with API credentials for higher limits.
    """
    
    def __init__(self, config: dict = None, user_agent: str = None):
        """
        Initialize JSON scraper.
        
        Args:
            config: Optional config dict with 'user_agent' key
            user_agent: Optional user agent string (overrides config)
        """
        if config is None:
            config = {}
        
        if user_agent:
            config['user_agent'] = user_agent
        
        if 'user_agent' not in config:
            config['user_agent'] = 'ChamberCheck/0.1'
        
        super().__init__(config)
        self.platform_name = "reddit"
        self.user_agent = config['user_agent']
        self.logger = setup_logger("RedditJSONScraper")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent
        })
        self.rate_limit_delay = 3  # seconds between requests (increased to avoid 429)
        self.retry_on_429 = config.get('retry_on_429', True)
        self.max_retries = config.get('max_retries', 3)
        self.retry_wait_seconds = config.get('retry_wait_seconds', 60)
    
    def authenticate(self) -> bool:
        """
        No authentication needed for JSON endpoints.
        
        Returns:
            Always returns True (no auth required)
        """
        return True
    
    def _make_request(self, url: str) -> Optional[dict]:
        """Make request with rate limiting and retry on 429."""
        for attempt in range(self.max_retries + 1):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url, timeout=10)
                
                # Handle 429 rate limit
                if response.status_code == 429:
                    if attempt < self.max_retries and self.retry_on_429:
                        wait_time = self.retry_wait_seconds * (attempt + 1)
                        self.logger.warning(f"Rate limit hit (429). Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        self.logger.error(f"Rate limit hit (429) - max retries exhausted")
                        response.raise_for_status()
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries and "429" in str(e):
                    wait_time = self.retry_wait_seconds * (attempt + 1)
                    self.logger.warning(f"Request error with 429. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                    
                self.logger.error(f"Request error: {e}")
                return None
        
        return None
    
    def fetch_posts(
        self,
        community: str,
        start_date: datetime,
        end_date: datetime,
        keywords: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Post]:
        """
        Fetch posts from subreddit using JSON API.
        
        Note: Date filtering happens client-side as JSON API doesn't support
        date range queries directly.
        
        Args:
            community: Subreddit name (without 'r/')
            start_date: Start of time range
            end_date: End of time range
            keywords: Optional keywords to filter posts
            limit: Max posts to fetch
        
        Returns:
            List of Post objects
        """
        return self.fetch_posts_by_engagement(
            community=community,
            start_date=start_date,
            end_date=end_date,
            sort_by='hot',
            limit=limit or 100
        )
    
    def fetch_posts_by_engagement(
        self,
        community: str,
        start_date: datetime = None,
        end_date: datetime = None,
        sort_by: str = 'hot',
        time_filter: str = 'week',
        limit: int = 100,
        keywords: Optional[List[str]] = None
    ) -> List[Post]:
        """
        Fetch posts sorted by engagement.
        
        Args:
            community: Subreddit name
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            sort_by: 'hot', 'new', 'top', 'rising', 'controversial'
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            limit: Max posts
            keywords: Optional list of keywords to search for
        
        Returns:
            List of Post objects
        """
        # If keywords provided, use search endpoint
        if keywords:
            return self._search_posts(
                community=community,
                keywords=keywords,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_by,
                time_filter=time_filter,
                limit=limit
            )
        
        posts = []
        after = None
        
        start_timestamp = start_date.timestamp() if start_date else 0
        end_timestamp = end_date.timestamp() if end_date else float('inf')
        
        while len(posts) < limit:
            # Build URL
            if sort_by in ['top', 'controversial']:
                url = f"https://www.reddit.com/r/{community}/{sort_by}.json?t={time_filter}&limit=100"
            else:
                url = f"https://www.reddit.com/r/{community}/{sort_by}.json?limit=100"
            
            if after:
                url += f"&after={after}"
            
            self.logger.info(f"Fetching: {url}")
            data = self._make_request(url)
            
            if not data or 'data' not in data:
                self.logger.warning(f"No data returned, stopping pagination")
                break
            
            children = data['data'].get('children', [])
            if not children:
                self.logger.warning(f"No children in response, stopping pagination")
                break
            
            self.logger.info(f"Processing {len(children)} posts from this batch")
            
            for child in children:
                if len(posts) >= limit:
                    break
                
                post_data = child['data']
                post_timestamp = post_data['created_utc']
                
                # Filter by date range
                if not (start_timestamp <= post_timestamp <= end_timestamp):
                    continue
                
                post = Post(
                    post_id=post_data['id'],
                    platform='reddit',
                    community=community,
                    title=post_data['title'],
                    content=post_data.get('selftext', ''),
                    author=post_data.get('author', '[deleted]'),
                    created_at=datetime.fromtimestamp(post_timestamp),
                    upvotes=post_data.get('ups', 0),
                    downvotes=post_data.get('downs', 0),
                    num_comments=post_data.get('num_comments', 0),
                    url=f"https://www.reddit.com{post_data['permalink']}",
                    metadata={
                        'upvote_ratio': post_data.get('upvote_ratio', 0),
                        'score': post_data.get('score', 0),
                        'is_self': post_data.get('is_self', False),
                        'flair': post_data.get('link_flair_text'),
                        'permalink': post_data['permalink'],
                        'external_url': post_data.get('url', None),  # For link posts
                        'gilded': post_data.get('gilded', 0),
                        'stickied': post_data.get('stickied', False),
                        'engagement_score': post_data.get('score', 0) + post_data.get('num_comments', 0) * 2
                    }
                )
                posts.append(post)
            
            # Get pagination token
            after = data['data'].get('after')
            self.logger.info(f"Collected {len(posts)} posts so far, after token: {after}")
            if not after:
                self.logger.warning(f"No 'after' token, reached end of available posts")
                break
        
        # Sort by engagement
        posts.sort(key=lambda p: p.metadata.get('engagement_score', 0), reverse=True)
        return posts
    
    def _search_posts(
        self,
        community: str,
        keywords: List[str],
        start_date: datetime = None,
        end_date: datetime = None,
        sort_by: str = 'relevance',
        time_filter: str = 'all',
        limit: int = 100
    ) -> List[Post]:
        """
        Search for posts by keywords in a subreddit.
        
        Args:
            community: Subreddit name
            keywords: List of keywords to search for
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            sort_by: 'relevance', 'hot', 'top', 'new', 'comments'
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            limit: Max posts
        
        Returns:
            List of Post objects matching keywords
        """
        posts = []
        after = None
        
        # Combine keywords into search query
        search_query = ' '.join(keywords)
        
        start_timestamp = start_date.timestamp() if start_date else 0
        end_timestamp = end_date.timestamp() if end_date else float('inf')
        
        self.logger.info(f"Searching r/{community} for: {search_query}")
        
        while len(posts) < limit:
            # Build search URL
            url = f"https://www.reddit.com/r/{community}/search.json?"
            url += f"q={search_query}&restrict_sr=on&sort={sort_by}&t={time_filter}&limit=100"
            
            if after:
                url += f"&after={after}"
            
            self.logger.info(f"Fetching: {url}")
            data = self._make_request(url)
            
            if not data or 'data' not in data:
                break
            
            children = data['data'].get('children', [])
            if not children:
                break
            
            for child in children:
                if len(posts) >= limit:
                    break
                
                post_data = child['data']
                post_timestamp = post_data['created_utc']
                
                # Filter by date range
                if not (start_timestamp <= post_timestamp <= end_timestamp):
                    continue
                
                post = Post(
                    post_id=post_data['id'],
                    platform='reddit',
                    community=community,
                    title=post_data['title'],
                    content=post_data.get('selftext', ''),
                    author=post_data.get('author', '[deleted]'),
                    created_at=datetime.fromtimestamp(post_timestamp),
                    upvotes=post_data.get('ups', 0),
                    downvotes=post_data.get('downs', 0),
                    num_comments=post_data.get('num_comments', 0),
                    url=f"https://www.reddit.com{post_data['permalink']}",
                    metadata={
                        'upvote_ratio': post_data.get('upvote_ratio', 0),
                        'score': post_data.get('score', 0),
                        'is_self': post_data.get('is_self', False),
                        'flair': post_data.get('link_flair_text'),
                        'permalink': post_data['permalink'],
                        'gilded': post_data.get('gilded', 0),
                        'stickied': post_data.get('stickied', False),
                        'engagement_score': post_data.get('score', 0) + post_data.get('num_comments', 0) * 2,
                        'search_keywords': keywords
                    }
                )
                posts.append(post)
            
            # Get pagination token
            after = data['data'].get('after')
            if not after:
                break
        
        # Sort by engagement
        posts.sort(key=lambda p: p.metadata.get('engagement_score', 0), reverse=True)
        return posts
    
    def fetch_comments(
        self,
        post_ids: List[str],
        limit: Optional[int] = None
    ) -> List[Comment]:
        """
        Fetch comments for given post IDs.
        
        Args:
            post_ids: List of Reddit post IDs
            limit: Max comments per post
        
        Returns:
            List of Comment objects
        """
        all_comments = []
        
        for post_id in post_ids:
            comments = self._fetch_post_comments(post_id, limit=limit)
            all_comments.extend(comments)
        
        return all_comments
    
    def _fetch_post_comments(self, post_id: str, limit: int = None) -> List[Comment]:
        """Fetch comments for a single post with pagination support.
        
        Args:
            post_id: Reddit post ID
            limit: Max comments to fetch. If None, fetches all available comments.
        
        Returns:
            List of Comment objects
        """
        # If limit is 0, return empty list (don't fetch any comments)
        if limit == 0:
            return []
        
        all_comments = []
        after = None
        request_count = 0
        
        while True:
            request_count += 1
            
            # Build URL for this request
            url = f"https://www.reddit.com/comments/{post_id}.json?limit=100"
            if after:
                url += f"&after={after}"
            
            self.logger.info(f"Fetching comments for post {post_id} (request {request_count}, collected: {len(all_comments)})")
            data = self._make_request(url)
            
            if not data or len(data) < 2:
                self.logger.warning(f"No data returned for post {post_id}")
                break
            
            # Reddit returns [post_data, comments_data]
            comments_data = data[1]
            
            def extract_comments(items, depth=0):
                """Recursively extract comments from nested structure."""
                for item in items:
                    # Stop if we've reached the limit
                    if limit is not None and len(all_comments) >= limit:
                        return False  # Signal to stop
                    
                    if item.get('kind') != 't1':  # t1 = comment
                        continue
                    
                    comment_data = item['data']
                    
                    # Skip deleted/removed
                    if comment_data.get('body') in [None, '[deleted]', '[removed]']:
                        continue
                    
                    comment = Comment(
                        comment_id=comment_data['id'],
                        post_id=post_id,
                        platform='reddit',
                        content=comment_data.get('body', ''),
                        author=comment_data.get('author', '[deleted]'),
                        created_at=datetime.fromtimestamp(comment_data['created_utc']),
                        upvotes=comment_data.get('ups', 0),
                        downvotes=comment_data.get('downs', 0),
                        parent_id=comment_data.get('parent_id', '').replace('t3_', '').replace('t1_', ''),
                        depth=depth,
                        metadata={
                            'score': comment_data.get('score', 0),
                            'is_submitter': comment_data.get('is_submitter', False),
                            'stickied': comment_data.get('stickied', False),
                            'gilded': comment_data.get('gilded', 0)
                        }
                    )
                    all_comments.append(comment)
                    
                    # Process replies
                    if 'replies' in comment_data and comment_data['replies']:
                        if isinstance(comment_data['replies'], dict):
                            replies = comment_data['replies'].get('data', {}).get('children', [])
                            if not extract_comments(replies, depth + 1):
                                return False  # Stop if limit reached
                
                return True  # Continue
            
            children = comments_data.get('data', {}).get('children', [])
            if not extract_comments(children):
                # Reached limit while processing
                break
            
            # Check for pagination token
            after = comments_data.get('data', {}).get('after')
            
            if not after:
                # No more pages available
                self.logger.info(f"Reached end of comments for post {post_id} (total: {len(all_comments)})")
                break
            
            # Stop if we've reached the limit
            if limit is not None and len(all_comments) >= limit:
                self.logger.info(f"Reached comment limit for post {post_id} (total: {len(all_comments)})")
                break
        
        return all_comments
    
    def fetch_post_metadata(self, post_id: str) -> dict:
        """
        Fetch metadata for a specific post.
        
        Args:
            post_id: Reddit post ID
        
        Returns:
            Dictionary with post metadata
        """
        url = f"https://www.reddit.com/comments/{post_id}.json"
        data = self._make_request(url)
        
        if not data or len(data) < 1:
            return {}
        
        post_data = data[0]['data']['children'][0]['data']
        
        return {
            'id': post_data['id'],
            'title': post_data['title'],
            'score': post_data.get('score', 0),
            'upvote_ratio': post_data.get('upvote_ratio', 0),
            'num_comments': post_data.get('num_comments', 0),
            'created_utc': post_data['created_utc'],
            'author': post_data.get('author', '[deleted]'),
            'subreddit': post_data.get('subreddit'),
            'url': post_data.get('url'),
            'permalink': post_data['permalink']
        }

    def fetch_subreddit_info(self, community: str) -> dict:
        """
        Fetch basic metadata about a subreddit.

        Args:
            community: Subreddit name (without 'r/')

        Returns:
            Dictionary with subreddit metadata, or error info if unavailable
        """
        url = f"https://www.reddit.com/r/{community}/about.json"
        data = self._make_request(url)

        if not data or 'data' not in data:
            return {
                "subreddit": community,
                "error": "No data returned",
                "source_url": url
            }

        info = data.get('data', {})
        created_utc = info.get('created_utc')
        created_at = None
        if isinstance(created_utc, (int, float)):
            created_at = datetime.fromtimestamp(created_utc).isoformat()

        return {
            "subreddit": info.get('display_name', community),
            "title": info.get('title'),
            "public_description": info.get('public_description'),
            "description": info.get('description'),
            "subscribers": info.get('subscribers'),
            "active_user_count": info.get('active_user_count'),
            "created_utc": created_utc,
            "created_at": created_at,
            "over18": info.get('over18'),
            "lang": info.get('lang'),
            "subreddit_type": info.get('subreddit_type'),
            "url": info.get('url'),
            "community_icon": info.get('community_icon') or info.get('icon_img'),
            "banner_img": info.get('banner_img'),
            "source_url": url
        }
    
    def get_required_config_fields(self) -> List[str]:
        """
        Get required config fields (none required for JSON scraper).
        
        Returns:
            Empty list (no required fields)
        """
        return []
