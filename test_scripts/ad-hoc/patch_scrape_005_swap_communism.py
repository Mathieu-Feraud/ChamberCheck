"""
Ad-hoc: patch scrape_005/posts.json
  - Remove all communism posts
  - Scrape decodingthegurus (500 posts, same settings as scrape_005)
  - Append to posts.json in-place
  - Rewrite posts_metadata.json to reflect the change
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ChamberCheck.scrapers.reddit_json_scraper import RedditJSONScraper
from ChamberCheck.scrapers.batch_scraper import remove_derived_metrics

SCRAPE_DIR   = Path("data/raw/scrape_005")
TARGET_SUB   = "decodingthegurus"
REMOVE_SUB   = "communism"
NUM_POSTS    = 500
SORT_METHOD  = "top"

posts_file    = SCRAPE_DIR / "posts.json"
metadata_file = SCRAPE_DIR / "posts_metadata.json"

# ── load existing data ────────────────────────────────────────────────────────
data  = json.loads(posts_file.read_text(encoding="utf-8"))
posts = data["posts"]
before = len(posts)

# ── remove communism ──────────────────────────────────────────────────────────
posts = [p for p in posts if p.get("community") != REMOVE_SUB]
removed = before - len(posts)
print(f"Removed {removed} posts from r/{REMOVE_SUB}  ({before} → {len(posts)})")

# ── scrape decodingthegurus ───────────────────────────────────────────────────
scraper = RedditJSONScraper(config={
    "retry_on_429": True,
    "max_retries": 10,
    "retry_wait_seconds": 60,
})

print(f"\nScraping r/{TARGET_SUB}  (limit={NUM_POSTS}, sort={SORT_METHOD})")
t0 = time.time()
fetched = scraper.fetch_posts_by_engagement(
    community=TARGET_SUB,
    start_date=None,
    end_date=None,
    sort_by=SORT_METHOD,
    time_filter="all",
    limit=NUM_POSTS,
)
new_posts = [p.to_dict() if hasattr(p, "to_dict") else p for p in fetched]
new_posts = remove_derived_metrics(new_posts)
elapsed = round(time.time() - t0, 1)

comment_counts = [p["num_comments"] for p in new_posts]
print(f"  ✓ {len(new_posts)} posts  "
      f"(num_comments range: {min(comment_counts)}–{max(comment_counts)})  "
      f"[{elapsed}s]")

# ── merge and save ────────────────────────────────────────────────────────────
posts.extend(new_posts)
posts_file.write_text(
    json.dumps({"posts": posts}, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nSaved {len(posts)} posts → {posts_file}")

# ── patch metadata ────────────────────────────────────────────────────────────
meta = json.loads(metadata_file.read_text(encoding="utf-8"))

# Remove communism entry, add decodingthegurus entry
meta["subreddits"] = [s for s in meta.get("subreddits", []) if s.get("subreddit") != REMOVE_SUB]
meta["subreddits"].append({
    "subreddit":           TARGET_SUB,
    "posts_scraped":       len(new_posts),
    "num_comments_min":    min(comment_counts),
    "num_comments_max":    max(comment_counts),
    "num_comments_total":  sum(comment_counts),
    "sort_method":         SORT_METHOD,
    "min_comments_filter": 0,
    "keywords":            None,
    "duration_seconds":    elapsed,
    "status":              "ok",
    "note":                "patched in post-hoc to replace communism",
})
meta["total_posts"]  = len(posts)
meta["patched_at"]   = datetime.now().isoformat()
meta["patch_note"]   = f"Removed r/{REMOVE_SUB}, added r/{TARGET_SUB}"

metadata_file.write_text(
    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"Updated metadata → {metadata_file}")
