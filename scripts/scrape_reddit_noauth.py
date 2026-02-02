"""
Reddit JSON API scraper script - no authentication required.

Uses the RedditJSONScraper class which accesses Reddit's public JSON endpoints.
Can be run interactively or with a config file for batch scraping multiple subreddits.

Usage:
    Interactive:  python scrape_reddit_noauth.py
    Config-based: python scrape_reddit_noauth.py --config config/scraper_config.json
"""

from pathlib import Path
import json
from datetime import datetime, timedelta
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ChamberCheck.scrapers import RedditJSONScraper
from ChamberCheck.constants import (
    DEFAULT_SUBREDDIT,
    DEFAULT_NUM_POSTS,
    DEFAULT_SORT_METHOD,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_USER_AGENT,
    DEFAULT_COMMENTS_PER_POST,
)


def load_config(config_path: str) -> dict:
    """Load scraper configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def scrape_subreddit(scraper: RedditJSONScraper, subreddit: str, num_posts: int, 
                     comments_per_post: int, sort_method: str, keywords: list = None) -> dict:
    """Scrape a single subreddit and return metadata."""
    print(f"\n{'='*70}")
    print(f"Scraping r/{subreddit}")
    print(f"{'='*70}")
    
    if keywords:
        print(f"Keywords: {', '.join(keywords)}")
    
    # Use fetch_posts_by_engagement which supports optional dates
    posts = scraper.fetch_posts_by_engagement(
        community=subreddit,
        start_date=None,
        end_date=None,
        sort_by=sort_method,
        time_filter='all',
        limit=num_posts,
        keywords=keywords
    )
    print(f"✓ Fetched {len(posts)} posts")
    
    all_comments = []
    for i, post in enumerate(posts, 1):
        try:
            # Posts are Post objects, use post_id for fetching comments
            if hasattr(post, 'post_id'):
                post_id = post.post_id
                post_title = post.title
            else:
                post_id = post.get('post_id')
                post_title = post.get('title', 'Untitled')
            
            # fetch_comments expects a LIST of post IDs
            comments = scraper.fetch_comments([post_id], comments_per_post)
            all_comments.extend(comments)
            print(f"  [{i}/{len(posts)}] {post_title[:50]}... → {len(comments)} comments")
        except Exception as e:
            print(f"  [{i}/{len(posts)}] Error fetching comments: {e}")
    
    print(f"✓ Total comments collected: {len(all_comments)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    keyword_suffix = ""
    if keywords:
        keyword_suffix = "_" + "_".join(keywords[:2])  # Use first 2 keywords
    
    filename = f"data/raw/{subreddit}_comments_json{keyword_suffix}_{timestamp}.json"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Comment objects to dicts for JSON serialization
    comments_dicts = [c.to_dict() if hasattr(c, 'to_dict') else c for c in all_comments]
    
    with open(filename, 'w') as f:
        json.dump(comments_dicts, f, indent=2)
    
    print(f"✓ Saved to: {filename}")
    
    return {
        "subreddit": subreddit,
        "keywords": keywords,
        "posts_count": len(posts),
        "comments_count": len(all_comments),
        "output_file": filename,
        "timestamp": timestamp
    }


def main_interactive():
    """Run Reddit JSON scraping interactively."""
    print("=" * 70)
    print("ChamberCheck - Reddit JSON API Scraper (No Auth Required)")
    print("=" * 70)
    
    scraper = RedditJSONScraper(user_agent=DEFAULT_USER_AGENT)
    
    # Get input
    subreddit = input(f"\n📝 Enter subreddit name (default: {DEFAULT_SUBREDDIT}): ").strip() or DEFAULT_SUBREDDIT
    
    # Ask about keyword search
    search_keywords = input("\n🔍 Search by keywords? (leave blank for all posts, or enter keywords separated by commas): ").strip()
    keywords = None
    if search_keywords:
        keywords = [k.strip() for k in search_keywords.split(',') if k.strip()]
        print(f"   Searching for: {', '.join(keywords)}")
    
    try:
        num_posts = int(input(f"\n📊 Number of posts (default: {DEFAULT_NUM_POSTS}): ").strip() or str(DEFAULT_NUM_POSTS))
    except ValueError:
        num_posts = DEFAULT_NUM_POSTS
    
    sort = input(f"🔢 Sort by (hot/new/top/rising/controversial, default: {DEFAULT_SORT_METHOD}): ").strip() or DEFAULT_SORT_METHOD
    
    # Get date range
    print("\n📅 Date Range (leave blank for defaults)")
    
    default_end = DEFAULT_END_DATE
    default_start = DEFAULT_START_DATE
    
    start_input = input(f"   Start date (YYYY-MM-DD, default: {DEFAULT_START_DATE.date()}): ").strip() or str(DEFAULT_START_DATE.date())
    if start_input:
        try:
            start_date = datetime.strptime(start_input, "%Y-%m-%d")
        except ValueError:
            print(f"   Invalid date format, using default: {DEFAULT_START_DATE.date()}")
            start_date = default_start
    else:
        start_date = default_start
    
    end_input = input(f"   End date (YYYY-MM-DD, default: {DEFAULT_END_DATE.date()}): ").strip() or str(DEFAULT_END_DATE.date())
    if end_input:
        try:
            end_date = datetime.strptime(end_input, "%Y-%m-%d")
        except ValueError:
            print(f"   Invalid date format, using default: {DEFAULT_END_DATE.date()}")
            end_date = default_end
    else:
        end_date = default_end
    
    print(f"\n   Filtering posts from {start_date.date()} to {end_date.date()}")
    
    # Calculate appropriate time_filter based on date range
    days_ago = (datetime.now() - start_date).days
    
    if days_ago <= 1:
        time_filter = "day"
    elif days_ago <= 7:
        time_filter = "week"
    elif days_ago <= 31:
        time_filter = "month"
    elif days_ago <= 365:
        time_filter = "year"
    else:
        time_filter = "all"
    
    print(f"   Using Reddit time filter: {time_filter} (to cover {days_ago} days ago)")

    
    # Fetch posts
    if keywords:
        print(f"\n📥 Searching r/{subreddit} for keywords: {', '.join(keywords)}...")
    else:
        print(f"\n📥 Fetching {num_posts} posts from r/{subreddit}...")
    
    posts = scraper.fetch_posts_by_engagement(
        community=subreddit,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort,
        time_filter=time_filter,
        limit=num_posts,
        keywords=keywords
    )
    
    print(f"✓ Fetched {len(posts)} posts")
    
    if not posts:
        print("No posts found!")
        return
    
    # Display summary
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Subreddit: r/{subreddit}")
    if keywords:
        print(f"Keywords: {', '.join(keywords)}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Posts: {len(posts)}")
    
    avg_upvotes = sum(p.upvotes for p in posts) / len(posts)
    avg_comments = sum(p.num_comments for p in posts) / len(posts)
    
    print(f"Avg upvotes: {avg_upvotes:.1f}")
    print(f"Avg comments: {avg_comments:.1f}")
    
    # Show all posts with details
    print("\n" + "=" * 70)
    print(f"TOP {len(posts)} POSTS (sorted by engagement)")
    print("=" * 70)
    
    top_posts = sorted(posts, key=lambda x: x.upvotes, reverse=True)
    for i, post in enumerate(top_posts, 1):
        print(f"\n{'=' * 70}")
        print(f"#{i}. {post.title}")
        print(f"{'=' * 70}")
        print(f"👍 Upvotes: {post.upvotes} | 💬 Comments: {post.num_comments}")
        print(f"👤 Author: u/{post.author}")
        print(f"🕒 Posted: {post.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"🔗 Link: {post.url}")
        
        # Show post content if it's a text post
        if post.content and post.content.strip():
            print(f"\n📝 Post Text:")
            print("-" * 70)
            # Truncate very long posts
            content = post.content[:500]
            if len(post.content) > 500:
                content += f"... [truncated, {len(post.content)} chars total]"
            print(content)
        else:
            print("\n📝 Post Text: [No text content - link post or image]")
    
    # Fetch comments
    fetch_comments = input("\n💬 Fetch comments for all posts? (y/n, default: y): ").strip().lower() != 'n'
    
    comments = []
    if fetch_comments and posts:
        print(f"\n📥 Fetching comments from {len(posts)} posts (up to {DEFAULT_COMMENTS_PER_POST} per post)...")
        
        comments = scraper.fetch_comments([p.post_id for p in posts], limit=DEFAULT_COMMENTS_PER_POST)
        print(f"✓ Fetched {len(comments)} comments total")
        
        if comments:
            top_level = sum(1 for c in comments if c.depth == 0)
            print(f"   Top-level: {top_level} | Replies: {len(comments) - top_level}")
    
    # Save data
    print("\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)
    
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build keyword suffix for filename
    keyword_suffix = ""
    if keywords:
        keyword_suffix = "_" + "_".join(keywords)
    
    # Save posts
    posts_file = f'data/raw/{subreddit}_posts_json{keyword_suffix}_{timestamp}.json'
    with open(posts_file, 'w', encoding='utf-8') as f:
        json.dump([p.to_dict() for p in posts], f, indent=2, default=str)
    print(f"✓ Posts: {posts_file}")
    
    # Save comments
    if comments:
        comments_file = f'data/raw/{subreddit}_comments_json{keyword_suffix}_{timestamp}.json'
        with open(comments_file, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in comments], f, indent=2, default=str)
        print(f"✓ Comments: {comments_file}")
    
    print("\n" + "=" * 70)
    print("✅ Scraping completed!")
    print("=" * 70)
    print("\n💡 No API key needed - uses public JSON endpoints")
    print("⚠️  Rate limited to ~60 requests per 10 min per IP")
    print("📚 Good for: Testing, small datasets, development")
    print("🔑 For production: Get API key for higher limits")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reddit scraper - interactive or config-based")
    parser.add_argument('--config', type=str, default='config/scraper_config.json',
                        help='Path to scraper config JSON file (default: config/scraper_config.json)')
    args = parser.parse_args()
    
    use_config = args.config and Path(args.config).exists()
    
    if use_config:
        # Config-based batch scraping
        config = load_config(args.config)
        
        # Load global settings and pass to scraper
        global_settings = config.get('global_settings', {})
        scraper_config = {
            'user_agent': DEFAULT_USER_AGENT,
            'retry_on_429': global_settings.get('retry_on_429', True),
            'max_retries': global_settings.get('max_retries', 3),
            'retry_wait_seconds': global_settings.get('retry_wait_seconds', 60)
        }
        scraper = RedditJSONScraper(config=scraper_config)
        
        print("=" * 70)
        print("ChamberCheck - Batch Scraper (Config-Based)")
        print("=" * 70)
        
        default_num_posts = global_settings.get('num_posts', DEFAULT_NUM_POSTS)
        default_comments_per_post = global_settings.get('comments_per_post', DEFAULT_COMMENTS_PER_POST)
        default_sort_method = global_settings.get('sort_method', DEFAULT_SORT_METHOD)
        
        results = []
        subreddits = config.get('subreddits', [])
        print(f"\n🚀 Starting batch scrape of {len(subreddits)} subreddit(s)...\n")
        
        for sub_config in subreddits:
            try:
                result = scrape_subreddit(
                    scraper,
                    subreddit=sub_config['name'],
                    num_posts=sub_config.get('num_posts', default_num_posts),
                    comments_per_post=sub_config.get('comments_per_post', default_comments_per_post),
                    sort_method=sub_config.get('sort_method', default_sort_method),
                    keywords=sub_config.get('keywords')
                )
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error scraping r/{sub_config['name']}: {e}")
                results.append({
                    "subreddit": sub_config['name'],
                    "error": str(e)
                })
        
        # Summary
        print("\n" + "=" * 70)
        print("BATCH SCRAPE SUMMARY")
        print("=" * 70)
        
        successful = sum(1 for r in results if 'error' not in r)
        print(f"\n✅ Successfully scraped: {successful}/{len(results)} subreddit(s)")
        
        total_comments = sum(r.get('comments_count', 0) for r in results)
        print(f"📊 Total comments collected: {total_comments}")
        
        for r in results:
            if 'error' not in r:
                print(f"\n  • r/{r['subreddit']}: {r['comments_count']} comments → {r['output_file']}")
            else:
                print(f"\n  • r/{r['subreddit']}: ❌ {r['error']}")
    else:
        # Interactive mode
        main_interactive()
