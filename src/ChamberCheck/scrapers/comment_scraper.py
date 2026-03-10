"""
Comment scraper: loads selected posts from pre_process output and scrapes their comments.

Post selection (score, topic peers, top-N per subreddit) is handled upstream
by the preprocessing stage. This module reads the pre_process_NNN.json file
directly and scrapes comments for every post it contains.

Comment collection strategy (per post):
- Fetch the full comment tree sorted by oldest-first (sort=old) so ordering
  is deterministic and independent of Reddit's engagement ranking
- Follow "more" stubs via the morechildren API to collect the full thread
- Keep every comment posted within `window_days` days of the original post
- No per-post count cap — all qualifying comments are stored
- Subsampling for analysis is deferred to analysis time

Output: scrape_NNN/comments/comments_NNN.json  +  comments_NNN_metadata.json
Supports resume: if latest comments file has no companion metadata, continues
from where it left off.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..config import Config
from ..constants import COMMENT_SCRAPING_WINDOW_DAYS
from ..models import Comment
from ..utils import setup_logger

# Type alias: a tree node is (Comment, list_of_child_nodes)
_Node = Tuple[Comment, list]

logger = setup_logger("CommentScraper")


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_scrape_dir(scrape_dir: Optional[Union[str, Path]]) -> Path:
    """Return scrape directory, defaulting to the latest data/raw/scrape_*."""
    if scrape_dir is not None:
        target = Path(scrape_dir)
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Scrape directory not found: {target}")
        return target
    raw_dir = Path("data/raw")
    candidates = sorted([p for p in raw_dir.glob("scrape_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No scrape_* folders found in {raw_dir}")
    return candidates[-1]


def _latest_pre_process(scrape_dir: Path) -> Path:
    """Return the latest pre_process_NNN.json (excludes *_metadata.json)."""
    pre_dir = scrape_dir / "pre_process"
    candidates = sorted([
        f for f in pre_dir.glob("pre_process_*.json")
        if "_metadata" not in f.name
    ])
    if not candidates:
        raise FileNotFoundError(f"No pre_process_*.json found in {pre_dir}")
    return candidates[-1]


# ── comment tree helpers ──────────────────────────────────────────────────────

def _parse_comment_tree(
    items: list,
    post_id: str,
    post_utc: float,
    window_seconds: float,
    depth: int = 0,
) -> List[_Node]:
    """Recursively parse Reddit JSON comment items into a list of tree nodes.

    Each node is ``(Comment, [child_nodes])``.  Comments posted more than
    ``window_seconds`` after the post are silently dropped, as are
    deleted/removed comments.
    """
    result: List[_Node] = []
    for item in items:
        if item.get("kind") != "t1":
            continue
        d = item["data"]
        if d.get("body") in (None, "[deleted]", "[removed]"):
            continue

        c_utc = float(d.get("created_utc") or 0)
        if window_seconds > 0 and (c_utc - post_utc) > window_seconds:
            continue

        comment = Comment(
            comment_id=d["id"],
            post_id=post_id,
            platform="reddit",
            content=d.get("body", ""),
            author=d.get("author", "[deleted]"),
            created_at=datetime.fromtimestamp(c_utc),
            upvotes=d.get("ups", 0),
            downvotes=d.get("downs", 0),
            parent_id=d.get("parent_id", "").replace("t3_", "").replace("t1_", ""),
            depth=depth,
            metadata={
                "score": d.get("score", 0),
                "is_submitter": d.get("is_submitter", False),
                "stickied": d.get("stickied", False),
                "gilded": d.get("gilded", 0),
            },
        )

        children: List[_Node] = []
        if d.get("replies") and isinstance(d["replies"], dict):
            reply_items = d["replies"].get("data", {}).get("children", [])
            children = _parse_comment_tree(
                reply_items, post_id, post_utc, window_seconds, depth + 1
            )

        result.append((comment, children))
    return result


def _flatten_thread(node: _Node) -> List[Comment]:
    """DFS-flatten a tree node into a flat list of Comment objects."""
    comment, children = node
    flat = [comment]
    for child in children:
        flat.extend(_flatten_thread(child))
    return flat


def _collect_more_ids(items: list) -> List[str]:
    """Walk a raw Reddit JSON item list and collect all IDs from ``kind: "more"`` stubs.

    These stubs appear whenever Reddit has truncated a branch of the comment tree.
    The IDs can be passed to the ``morechildren`` API to expand them.
    """
    ids: List[str] = []
    for item in items:
        if item.get("kind") == "more":
            ids.extend(item.get("data", {}).get("children", []))
        elif item.get("kind") == "t1":
            replies = item.get("data", {}).get("replies")
            if replies and isinstance(replies, dict):
                child_items = replies.get("data", {}).get("children", [])
                ids.extend(_collect_more_ids(child_items))
    return ids


def _fetch_morechildren(
    scraper,
    post_id: str,
    ids: List[str],
) -> List[dict]:
    """Call Reddit's ``morechildren`` API for a batch of comment IDs.

    Args:
        scraper:  A ``RedditJSONScraper`` instance.
        post_id:  Reddit post ID (without ``t3_`` prefix).
        ids:      Up to 100 comment IDs to expand.

    Returns:
        Raw list of ``{kind, data}`` items, or empty list on failure.
    """
    id_str = ",".join(ids)
    url = (
        f"https://www.reddit.com/api/morechildren.json"
        f"?api_type=json&link_id=t3_{post_id}&children={id_str}&sort=old"
    )
    data = scraper._make_request(url)
    if not data:
        return []
    return data.get("json", {}).get("data", {}).get("things", [])


def _fetch_windowed_comments(
    scraper,
    post_id: str,
    post_created_utc: float,
    window_days: int,
) -> List[Comment]:
    """Fetch all comments for a single Reddit post that fall within the time window.

    Uses ``sort=old`` so comments are returned in chronological order —
    deterministic and free of Reddit's engagement-based ranking.

    The initial request returns up to 500 top-level threads.  Any ``kind: "more"``
    stubs (truncated branches) are followed via the ``morechildren`` API in
    batches of 100 IDs.  Pagination stops early once a full batch falls entirely
    outside the time window — because ``sort=old`` guarantees remaining stubs
    represent even newer comments.

    No per-post count limit is applied; subsampling is deferred to analysis time.

    Args:
        scraper:           A ``RedditJSONScraper`` instance.
        post_id:           Reddit post ID (without ``t3_`` prefix).
        post_created_utc:  Unix timestamp of the post creation time.
        window_days:       Discard comments posted more than this many days after the post.

    Returns:
        All qualifying ``Comment`` objects in roughly chronological order.
    """
    window_seconds = window_days * 86_400

    # ─ initial tree request ───────────────────────────────────────────────
    # sort=old → oldest comments first; limit=500 is the Reddit API maximum
    url = f"https://www.reddit.com/comments/{post_id}.json?sort=old&limit=500&depth=10"
    data = scraper._make_request(url)

    if not data or len(data) < 2:
        logger.warning(f"No comment data returned for post {post_id}")
        return []

    raw_children = data[1].get("data", {}).get("children", [])
    roots        = _parse_comment_tree(raw_children, post_id, post_created_utc, window_seconds)

    all_comments: List[Comment] = []
    for root in roots:
        all_comments.extend(_flatten_thread(root))

    # ─ follow "more" stubs ──────────────────────────────────────────────
    pending_ids: List[str] = _collect_more_ids(raw_children)
    _BATCH = 100

    while pending_ids:
        batch       = pending_ids[:_BATCH]
        pending_ids = pending_ids[_BATCH:]

        more_items = _fetch_morechildren(scraper, post_id, batch)
        if not more_items:
            break

        in_window = 0
        for item in more_items:
            if item.get("kind") == "more":
                # Stubs can appear inside morechildren responses too
                pending_ids.extend(item.get("data", {}).get("children", []))
                continue
            if item.get("kind") != "t1":
                continue
            d = item["data"]
            if d.get("body") in (None, "[deleted]", "[removed]"):
                continue
            c_utc = float(d.get("created_utc") or 0)
            if window_seconds > 0 and (c_utc - post_created_utc) > window_seconds:
                continue
            all_comments.append(Comment(
                comment_id=d["id"],
                post_id=post_id,
                platform="reddit",
                content=d.get("body", ""),
                author=d.get("author", "[deleted]"),
                created_at=datetime.fromtimestamp(c_utc),
                upvotes=d.get("ups", 0),
                downvotes=d.get("downs", 0),
                parent_id=d.get("parent_id", "").replace("t3_", "").replace("t1_", ""),
                depth=d.get("depth", 0),
                metadata={
                    "score":        d.get("score", 0),
                    "is_submitter": d.get("is_submitter", False),
                    "stickied":     d.get("stickied", False),
                    "gilded":       d.get("gilded", 0),
                },
            ))
            in_window += 1

        # Early exit: sort=old means remaining stubs are newer; if none of this
        # batch was within the window the rest won't be either.
        if in_window == 0 and pending_ids:
            logger.info(
                f"morechildren batch for {post_id}: 0 comments in window — stopping."
            )
            break

    logger.info(f"Post {post_id}: {len(all_comments)} comments within {window_days}d window")
    return all_comments



# ── post loading ──────────────────────────────────────────────────────────────

def _load_pre_process(pre_process_path: Path) -> list:
    """Load all posts from a pre_process_NNN.json file.

    All filtering (score, topic peers, top-N per subreddit) has already been
    applied during preprocessing. This function simply deserialises the file.
    """
    return json.loads(pre_process_path.read_text(encoding="utf-8"))


# ── main entry ────────────────────────────────────────────────────────────────

def scrape_comments(
    scrape_dir: Optional[Union[str, Path]] = None,
    config_path: str = "config/config.yaml",
) -> str:
    """Scrape comments for posts selected from pre_process output.

    For each selected post, fetches comments sorted oldest-first and retains
    every comment posted within ``window_days`` days of the original post.
    No per-post comment cap is applied — all qualifying comments are stored so
    that downstream analysis can apply its own unbiased sampling.

    Supports resume: if the latest ``comments_NNN.json`` has no companion
    metadata file the run is considered incomplete and continued from where
    it stopped.

    Args:
        scrape_dir:  Path to the scrape folder. Defaults to latest ``data/raw/scrape_*``.
        config_path: Path to the YAML config file.

    Returns:
        Path string of the written ``comments_NNN.json`` file.
    """
    from .reddit_json_scraper import RedditJSONScraper

    cfg         = Config(config_path)
    window_days = cfg.get("comment_scraping.window_days", COMMENT_SCRAPING_WINDOW_DAYS)

    rate_limit  = cfg.get("scraping.rate_limit_delay",   2)
    max_retries = cfg.get("scraping.max_retries",        10)
    retry_wait  = cfg.get("scraping.retry_wait_seconds", 60)

    scrape_dir       = _resolve_scrape_dir(scrape_dir)
    pre_process_path = _latest_pre_process(scrape_dir)
    out_dir          = scrape_dir / "comments"
    out_dir.mkdir(exist_ok=True)

    print(f"Using scrape directory  : {scrape_dir}")
    print(f"Using pre-process file  : {pre_process_path.name}")
    print(f"Comment window          : {window_days} days from post")

    # ── resolve run number (resume if latest has no metadata) ─────────────────
    existing = sorted([f for f in out_dir.glob("comments_*.json") if "_metadata" not in f.name])
    if existing:
        last      = existing[-1]
        num       = int(last.stem.split("_")[-1])
        last_meta = out_dir / f"{last.stem}_metadata.json"
        if not last_meta.exists():
            out_path  = last
            meta_path = last_meta
            next_num  = num
            results   = json.loads(last.read_text(encoding="utf-8"))
            done_ids  = {r["post_id"] for r in results}
            print(f"Resuming run {next_num:03d} - {len(done_ids)} posts already scraped")
        else:
            next_num  = len(existing) + 1
            out_path  = out_dir / f"comments_{next_num:03d}.json"
            meta_path = out_dir / f"comments_{next_num:03d}_metadata.json"
            results   = []
            done_ids  = set()
    else:
        next_num  = 1
        out_path  = out_dir / f"comments_{next_num:03d}.json"
        meta_path = out_dir / f"comments_{next_num:03d}_metadata.json"
        results   = []
        done_ids  = set()

    # ── load posts (all filtering done at preprocess time) ─────────────────────
    selected  = _load_pre_process(pre_process_path)
    to_scrape = [p for p in selected if p["post_id"] not in done_ids]
    print(f"Posts to scrape         : {len(to_scrape)} (skipping {len(done_ids)} already done)")

    # ── scraper setup ─────────────────────────────────────────────────────────
    scraper = RedditJSONScraper(config={
        "user_agent":         "ChamberCheck/0.1 Research",
        "retry_on_429":       True,
        "max_retries":        max_retries,
        "retry_wait_seconds": retry_wait,
    })
    scraper.rate_limit_delay = rate_limit

    total_comments = sum(len(r["comments"]) for r in results)
    errors         = 0
    run_start      = time.time()

    for idx, post in enumerate(to_scrape, start=1):
        post_id = post["post_id"]
        title   = post.get("title", "")[:70]
        print(f"  [{idx:03d}/{len(to_scrape)}] {post['community']:<20} {title}")

        # Resolve post creation time for the time-window filter
        post_created_utc: float = 0.0
        raw_created_at = post.get("created_at")
        if raw_created_at:
            try:
                post_created_utc = datetime.fromisoformat(str(raw_created_at)).timestamp()
            except (ValueError, TypeError):
                logger.warning(f"Could not parse created_at for post {post_id}: {raw_created_at!r}")

        try:
            comments = _fetch_windowed_comments(
                scraper,
                post_id=post_id,
                post_created_utc=post_created_utc,
                window_days=window_days,
            )
            comment_dicts = [c.to_dict() for c in comments]
        except Exception as exc:
            logger.error(f"Failed to scrape post {post_id}: {exc}")
            comment_dicts = []
            errors += 1

        record = {
            "post_id":           post_id,
            "community":         post.get("community"),
            "title":             post.get("title"),
            "url":               post.get("url"),
            "created_at":        post.get("created_at"),
            "topic":             post.get("topic"),
            "secondary_topics":  post.get("secondary_topics", []),
            "discussion_score":  post.get("discussion_score"),
            "topic_group":       post.get("topic_group"),
            "topic_group_level": post.get("topic_group_level"),
            "comments":          comment_dicts,
            "comment_count":     len(comment_dicts),
        }
        total_comments += len(comment_dicts)
        results.append(record)
        print(f"          -> {len(comment_dicts)} comments  (window: {window_days}d)")

        # Atomic write via temp file — avoids Windows file-lock errors
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_path)

    run_end = time.time()

    metadata = {
        "run":               next_num,
        "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start)),
        "completed_at":      time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_end)),
        "duration_seconds":  round(run_end - run_start, 1),
        "config_file":       str(config_path),
        "scrape_dir":        str(scrape_dir),
        "pre_process_file":  str(pre_process_path),
        "output_file":       str(out_path),
        "params": {
            "window_days":              window_days,
            "rate_limit_delay_seconds": rate_limit,
            "max_retries_on_429":       max_retries,
            "retry_wait_seconds":       retry_wait,
            "api_initial_limit":        500,
            "api_initial_depth":        10,
            "api_morechildren_batch":   100,
            "api_sort":                 "old",
            "note_post_selection":      "post selection filters applied at preprocessing stage",
        },
        "counts": {
            "posts_selected":  len(selected),
            "posts_scraped":   len(results),
            "total_comments":  total_comments,
            "errors":          errors,
        },
        "subreddits": sorted({r["community"] for r in results if r.get("community")}),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Results  -> {out_path}")
    print(f"  Metadata -> {meta_path}")
    print(f"  Posts    : {len(results)}  |  Comments: {total_comments:,}  |  Errors: {errors}")
    return str(out_path)
