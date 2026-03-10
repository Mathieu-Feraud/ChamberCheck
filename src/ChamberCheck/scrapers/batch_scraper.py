"""
Batch scraper for orchestrating multi-subreddit scraping operations.

Handles configuration loading, batch processing, file management, and metadata tracking.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from .reddit_json_scraper import RedditJSONScraper


def remove_derived_metrics(posts: list) -> list:
    """Remove derived metrics from post dicts before writing raw output."""
    cleaned = []
    for post in posts:
        if isinstance(post, dict):
            metadata = post.get("metadata")
            if isinstance(metadata, dict):
                metadata = dict(metadata)
                metadata.pop("engagement_score", None)
                metadata.pop("duration_seconds", None)
                post = dict(post)
                post["metadata"] = metadata
        cleaned.append(post)
    return cleaned


def load_config(config_path: str) -> dict:
    """Load scraper configuration from a JSON or YAML file."""
    path = Path(config_path)
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError as exc:
                raise ImportError("pyyaml is required for YAML config files: pip install pyyaml") from exc
        return json.load(f)


def get_next_scrape_folder_number(data_dir: str = "data/raw") -> int:
    """Get the next available scrape folder number.
    
    Looks for existing folders like 'scrape_001', 'scrape_002', etc.
    Returns 1 if no folders exist, otherwise returns max + 1.
    """
    path = Path(data_dir)
    if not path.exists():
        return 1
    
    # Find all scrape_NNN folders
    pattern = "scrape_*"
    existing_folders = [f for f in path.glob(pattern) if f.is_dir()]
    
    max_num = 0
    for folder in existing_folders:
        # Extract number from folder name (e.g., "scrape_001" -> 1)
        folder_name = folder.name
        if folder_name.startswith("scrape_"):
            try:
                num = int(folder_name.split('_')[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                pass
    
    return max_num + 1 if max_num > 0 else 1


def save_metadata(subreddit: str, mode: str, start_time: datetime, 
                  end_time: datetime, output_folder: str, config: dict = None, 
                  posts_count: int = 0, comments_count: int = 0) -> str:
    """Save metadata about the scraping operation.
    
    Args:
        subreddit: Subreddit name
        mode: Scraping mode (posts_only, comments_only, posts_and_comments)
        start_time: Start time of scraping
        end_time: End time of scraping
        output_folder: Output folder path
        config: Scraper configuration parameters
        posts_count: Number of posts scraped
        comments_count: Number of comments scraped
    
    Returns:
        str: Path to metadata file
    """
    metadata = {
        "subreddit": subreddit,
        "mode": mode,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "posts_count": posts_count,
        "comments_count": comments_count,
        "total_items": posts_count + comments_count,
        "config": config or {}
    }
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    metadata_file = f"{output_folder}/{subreddit}_scraper_metadata.json"
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def save_subreddit_info(subreddits_info: List[Dict], output_folder: str) -> str:
    """Save aggregated subreddit metadata for the scrape run."""
    payload = {
        "generated_at": datetime.now().isoformat(),
        "subreddits": subreddits_info
    }

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    info_file = f"{output_folder}/subreddits_info.json"

    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return info_file


def scrape_subreddits_info(
    scraper: RedditJSONScraper,
    subreddits: List[Dict],
    output_folder: str
) -> str:
    """Fetch and save subreddit metadata for all configured subreddits."""
    subreddits_info = []
    for sub_config in subreddits:
        name = sub_config.get('name')
        if not name:
            continue
        try:
            info = scraper.fetch_subreddit_info(name)
        except Exception as e:
            info = {
                "subreddit": name,
                "error": str(e)
            }
        subreddits_info.append(info)

    return save_subreddit_info(subreddits_info, output_folder)


def scrape_subreddit(scraper: RedditJSONScraper, subreddit: str, num_posts: int, 
                     comments_per_post, sort_method: str, output_folder: str,
                     keywords: list = None, min_comments_per_post: int = 0) -> dict:
    """Scrape a single subreddit (posts and comments) and return metadata.
    
    Args:
        scraper: RedditJSONScraper instance
        subreddit: Subreddit name
        num_posts: Number of posts to fetch
        comments_per_post: Number of comments to fetch per post, or "all" to fetch all comments
        sort_method: Sort method (hot/new/top/rising/controversial)
        output_folder: Output folder path for saving files
        keywords: Optional list of keywords to search for
        min_comments_per_post: Minimum number of comments a post must have to be included (default: 0)
    
    Returns:
        dict: Metadata about the scrape operation
    """
    start_time = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"Scraping r/{subreddit} (Posts + Comments)")
    print(f"{'='*70}")
    
    if keywords:
        print(f"Keywords: {', '.join(keywords)}")
    
    # Convert "all" to None for fetching all comments
    if comments_per_post == "all":
        limit_per_post = None
        print(f"Comments per post: all")
    else:
        limit_per_post = int(comments_per_post)
        print(f"Comments per post: {limit_per_post}")
    
    if min_comments_per_post > 0:
        print(f"Minimum comments per post: {min_comments_per_post}")
    
    # Fetch more posts than needed to ensure we have enough after filtering
    fetch_limit = num_posts * 3 if min_comments_per_post > 0 else num_posts
    
    all_fetched_posts = scraper.fetch_posts_by_engagement(
        community=subreddit,
        start_date=None,
        end_date=None,
        sort_by=sort_method,
        time_filter='all',
        limit=fetch_limit,
        keywords=keywords
    )
    print(f"✓ Fetched {len(all_fetched_posts)} posts")
    
    # Filter posts by minimum comment count if specified
    if min_comments_per_post > 0:
        posts = [p for p in all_fetched_posts if p.num_comments >= min_comments_per_post]
        filtered_count = len(all_fetched_posts) - len(posts)
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} posts with < {min_comments_per_post} comments")
        
        # Limit to num_posts after filtering
        posts = posts[:num_posts]
        
        if len(posts) < num_posts:
            print(f"  ⚠️  Warning: Only found {len(posts)} posts meeting minimum threshold (requested {num_posts})")
    else:
        posts = all_fetched_posts[:num_posts]
    
    print(f"✓ Processing {len(posts)} posts")
    
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
            comments = scraper.fetch_comments([post_id], limit_per_post)
            all_comments.extend(comments)
            print(f"  [{i}/{len(posts)}] {post_title[:50]}... → {len(comments)} comments")
        except Exception as e:
            print(f"  [{i}/{len(posts)}] Error fetching comments: {e}")
    
    print(f"✓ Total comments collected: {len(all_comments)}")
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Save data file with both posts and comments (no number suffix)
    data_file = f"{output_folder}/{subreddit}.json"
    
    posts_dicts = [p.to_dict() if hasattr(p, 'to_dict') else p for p in posts]
    posts_dicts = remove_derived_metrics(posts_dicts)
    comments_dicts = [c.to_dict() if hasattr(c, 'to_dict') else c for c in all_comments]
    
    # Combined data structure with both posts and comments
    combined_data = {
        "posts": posts_dicts,
        "comments": comments_dicts
    }
    
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"✓ Saved to: {data_file}")
    print(f"   └─ {len(posts)} posts + {len(all_comments)} comments")
    
    # Save metadata
    end_time = datetime.now()
    config = {
        "num_posts": num_posts,
        "comments_per_post": comments_per_post,  # Save original value ("all" or number)
        "sort_method": sort_method,
        "keywords": keywords,
        "min_comments_per_post": min_comments_per_post
    }
    metadata_file = save_metadata(
        subreddit, "posts_and_comments", start_time, end_time, output_folder,
        config=config, posts_count=len(posts), comments_count=len(all_comments)
    )
    print(f"✓ Metadata: {metadata_file}")
    
    return {
        "subreddit": subreddit,
        "keywords": keywords,
        "posts_count": len(posts),
        "comments_count": len(all_comments),
        "output_file": data_file,
        "metadata_file": metadata_file
    }


def batch_scrape_posts_only(config_path: str, output_folder: str = None) -> str:
    """Scrape posts only (no comments) for all subreddits in config, saving to a single posts.json.

    Args:
        config_path: Path to scraper configuration JSON file
        output_folder: Optional output folder under data/raw/. Auto-increments if omitted.

    Returns:
        Path to the saved posts.json file
    """
    config = load_config(config_path)
    global_settings = config.get("scraping", {})

    scraper_config = {
        "retry_on_429": global_settings.get("retry_on_429", True),
        "max_retries": global_settings.get("max_retries", 3),
        "retry_wait_seconds": global_settings.get("retry_wait_seconds", 60),
    }
    scraper = RedditJSONScraper(config=scraper_config)

    if output_folder is None:
        next_num = get_next_scrape_folder_number()
        output_folder = f"data/raw/scrape_{next_num:03d}"
    elif not output_folder.startswith("data/raw/"):
        output_folder = f"data/raw/{output_folder}"

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    default_num_posts = global_settings.get("num_posts", 100)
    default_sort_method = global_settings.get("sort_method", "top")
    default_time_filter = global_settings.get("time_filter", "all")
    default_min_comments = global_settings.get("min_comments", global_settings.get("min_comments_per_post", 0))

    def _parse_date(val):
        if not val:
            return None
        from datetime import datetime as _dt
        return _dt.strptime(str(val), "%Y-%m-%d")

    default_start_date = _parse_date(global_settings.get("start_date"))
    default_end_date   = _parse_date(global_settings.get("end_date"))

    # subreddits list: new format has it nested under scraping, legacy format is top-level
    subreddits_raw = global_settings.get("subreddits", config.get("subreddits", []))
    # Normalise: YAML list of strings → list of dicts with 'name' key
    subreddits = [
        {"name": s} if isinstance(s, str) else s
        for s in subreddits_raw
    ]

    print("=" * 70)
    print("ChamberCheck - Posts-Only Scraper")
    print("=" * 70)
    print(f"Output folder : {output_folder}")
    print(f"Subreddits    : {len(subreddits)}")
    print()

    all_posts = []
    per_subreddit: List[Dict] = []
    run_start = datetime.now()

    for sub_config in subreddits:
        name = sub_config["name"]
        num_posts = sub_config.get("num_posts", default_num_posts)
        sort_method = sub_config.get("sort_method", default_sort_method)
        time_filter = sub_config.get("time_filter", default_time_filter)
        min_comments = sub_config.get("min_comments", sub_config.get("min_comments_per_post", default_min_comments))
        keywords = sub_config.get("keywords")
        start_date = _parse_date(sub_config.get("start_date")) if "start_date" in sub_config else default_start_date
        end_date   = _parse_date(sub_config.get("end_date"))   if "end_date"   in sub_config else default_end_date
        fetch_limit = num_posts * 3 if min_comments > 0 else num_posts

        sub_start = datetime.now()
        print(f"Scraping r/{name}  (posts only, limit={num_posts}, sort={sort_method})")
        try:
            fetched = scraper.fetch_posts_by_engagement(
                community=name,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_method,
                time_filter=time_filter,
                limit=fetch_limit,
                keywords=keywords,
            )
            if min_comments > 0:
                fetched = [p for p in fetched if p.num_comments >= min_comments]
            posts = fetched[:num_posts]
            posts_dicts = [p.to_dict() if hasattr(p, "to_dict") else p for p in posts]
            posts_dicts = remove_derived_metrics(posts_dicts)
            all_posts.extend(posts_dicts)
            comment_counts = [p["num_comments"] for p in posts_dicts]
            sub_summary = {
                "subreddit": name,
                "posts_scraped": len(posts_dicts),
                "num_comments_min": min(comment_counts) if comment_counts else 0,
                "num_comments_max": max(comment_counts) if comment_counts else 0,
                "num_comments_total": sum(comment_counts),
                "sort_method": sort_method,
                "min_comments_filter": min_comments,
                "keywords": keywords,
                "duration_seconds": (datetime.now() - sub_start).total_seconds(),
                "status": "ok",
            }
            print(f"  ✓ {len(posts_dicts)} posts  (num_comments range: "
                  f"{sub_summary['num_comments_min']}–{sub_summary['num_comments_max']})")
        except Exception as exc:
            print(f"  ❌ Error: {exc}")
            sub_summary = {"subreddit": name, "status": "error", "error": str(exc)}
        per_subreddit.append(sub_summary)

    run_end = datetime.now()

    out_path = f"{output_folder}/posts.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"posts": all_posts}, fh, indent=2)

    metadata = {
        "generated_at": run_start.isoformat(),
        "completed_at": run_end.isoformat(),
        "duration_seconds": (run_end - run_start).total_seconds(),
        "output_file": out_path,
        "total_posts": len(all_posts),
        "config": {
            "config_path": config_path,
            "num_posts": default_num_posts,
            "sort_method": default_sort_method,
            "time_filter": default_time_filter,
            "start_date": default_start_date.isoformat() if default_start_date else None,
            "end_date": default_end_date.isoformat() if default_end_date else None,
            "min_comments": default_min_comments,
        },
        "subreddits": per_subreddit,
    }
    metadata_path = f"{output_folder}/posts_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print()
    print(f"✓ Saved {len(all_posts)} posts   → {out_path}")
    print(f"✓ Metadata                      → {metadata_path}")
    return out_path


def batch_scrape(config_path: str, scraper_config: dict = None, output_folder: str = None) -> List[Dict]:
    """Perform batch scraping of multiple subreddits from config file.
    
    Args:
        config_path: Path to scraper configuration JSON file
        scraper_config: Optional scraper configuration parameters
        output_folder: Optional output folder name. If not provided, generates scrape_001, scrape_002, etc.
                       Folder will be created under data/raw/
    
    Returns:
        List of result dictionaries for each subreddit
    """
    config = load_config(config_path)
    global_settings = config.get("scraping", {})
    if scraper_config is None:
        scraper_config = {
            'retry_on_429': global_settings.get('retry_on_429', True),
            'max_retries': global_settings.get('max_retries', 3),
            'retry_wait_seconds': global_settings.get('retry_wait_seconds', 60)
        }
    
    scraper = RedditJSONScraper(config=scraper_config)
    
    print("=" * 70)
    print("ChamberCheck - Batch Scraper (Config-Based)")
    print("=" * 70)
    
    # Determine output folder
    if output_folder is None:
        # Generate next scrape_NNN folder
        next_num = get_next_scrape_folder_number()
        output_folder = f"data/raw/scrape_{next_num:03d}"
    else:
        # Ensure output folder is under data/raw/
        if not output_folder.startswith("data/raw/"):
            output_folder = f"data/raw/{output_folder}"
    
    print(f"📁 Output folder: {output_folder}\n")
    
    default_num_posts = global_settings.get("num_posts", 200)
    default_comments_per_post = config.get("comment_scraping", {}).get("max_comments_per_post", 2000)
    default_sort_method = global_settings.get("sort_method", "top")
    default_min_comments_per_post = global_settings.get("min_comments_per_post", 0)

    subreddits_raw = global_settings.get("subreddits", [])
    subreddits = [{"name": s} if isinstance(s, str) else s for s in subreddits_raw]

    results = []
    print(f"🚀 Starting batch scrape of {len(subreddits)} subreddit(s)...\n")

    if subreddits:
        info_file = scrape_subreddits_info(scraper, subreddits, output_folder)
        print(f"✓ Subreddit info: {info_file}\n")
    
    for sub_config in subreddits:
        try:
            # Always scrape posts and comments
            result = scrape_subreddit(
                scraper,
                subreddit=sub_config['name'],
                num_posts=sub_config.get('num_posts', default_num_posts),
                comments_per_post=sub_config.get('comments_per_post', default_comments_per_post),
                sort_method=sub_config.get('sort_method', default_sort_method),
                output_folder=output_folder,
                keywords=sub_config.get('keywords'),
                min_comments_per_post=sub_config.get('min_comments_per_post', default_min_comments_per_post)
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
    
    return results
