"""
Post preprocessing: filter and rank posts per subreddit for comment scraping.

Joins posts.json with the latest analysis_NNN.json, applies hard filters
(num_comments threshold, topic clarity, discussion score, topic-group peers),
and selects the top-N posts per subreddit ranked by discussion_score.

The ``topic_group`` and ``topic_group_level`` fields written here tell the
comment scraper which peer group each post belongs to — no re-filtering needed
at scrape time.
"""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Union

from ..config import Config
from ..constants import PREPROCESSING_MIN_DISCUSSION_SCORE, PREPROCESSING_MIN_TOPIC_PEERS


def _latest_analysis(analysis_dir: Path) -> Path:
    """Return the path to the most recent ``analysis_NNN.json`` file.

    Excludes ``*_metadata.json`` companions from consideration.
    """
    candidates = sorted(
        [f for f in analysis_dir.glob("analysis_*.json") if "_metadata" not in f.name]
    )
    if not candidates:
        raise FileNotFoundError(f"No analysis_*.json found in {analysis_dir}")
    return candidates[-1]


def preprocess_posts(scrape_dir: Union[str, Path], config_path: str = "config/config.yaml") -> str:
    """Filter and rank posts, writing ``pre_process.json`` for a scrape folder.

    Reads:
        ``<scrape_dir>/posts.json``
        ``<scrape_dir>/posts_analysis/analysis_NNN.json``  (latest run)

    Writes:
        ``<scrape_dir>/pre_process/pre_process.json``
        ``<scrape_dir>/pre_process/pre_process_metadata.json``

    Args:
        scrape_dir:  Path to the scrape folder (e.g. ``"data/raw/scrape_004"``).
        config_path: Path to the unified YAML config file.

    Returns:
        Path to the written ``pre_process_NNN.json`` file.
    """
    cfg             = Config(config_path)
    min_comments    = cfg.get("scraping.min_comments", cfg.get("preprocessing.min_comments", 0))
    top_n           = cfg.get("preprocessing.top_n_per_subreddit", 30)
    min_score       = cfg.get("preprocessing.min_discussion_score", PREPROCESSING_MIN_DISCUSSION_SCORE)
    min_peers       = cfg.get("preprocessing.min_topic_peers",      PREPROCESSING_MIN_TOPIC_PEERS)

    scrape_dir    = Path(scrape_dir)
    posts_file    = scrape_dir / "posts.json"
    analysis_dir  = scrape_dir / "posts_analysis"
    analysis_file = _latest_analysis(analysis_dir)
    out_dir       = scrape_dir / "pre_process"
    out_dir.mkdir(exist_ok=True)

    # Auto-increment run number (exclude *_metadata.json from count)
    existing  = [f for f in out_dir.glob("pre_process_*.json") if "_metadata" not in f.name]
    next_num  = len(existing) + 1
    out_path  = out_dir / f"pre_process_{next_num:03d}.json"
    meta_path = out_dir / f"pre_process_{next_num:03d}_metadata.json"

    print(f"Posts file   : {posts_file}")
    print(f"Analysis file: {analysis_file}")

    run_start = time.time()

    posts    = json.loads(posts_file.read_text(encoding="utf-8"))["posts"]
    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))

    analysis_by_id: dict = {
        rec["_post_id"]: rec for rec in analysis if rec.get("_post_id")
    }
    print(f"\nLoaded {len(posts)} posts, {len(analysis_by_id)} analysis records")

    # ── merge ─────────────────────────────────────────────────────────────────
    merged    = []
    unmatched = 0
    for post in posts:
        pid = post.get("post_id")
        ann = analysis_by_id.get(pid)
        if ann is None:
            unmatched += 1
            continue
        merged.append({
            "post_id":           pid,
            "community":         post.get("community"),
            "title":             post.get("title"),
            "content":           post.get("content"),
            "author":            post.get("author"),
            "created_at":        post.get("created_at"),
            "upvotes":           post.get("upvotes"),
            "num_comments":      post.get("num_comments"),
            "url":               post.get("url"),
            "topic":             ann.get("topic"),
            "secondary_topics":  ann.get("secondary_topics", []),
            "topic_confidence":  ann.get("topic_confidence"),
            "secondary_confidence": ann.get("secondary_confidence", []),
            "discussion_score":  ann.get("discussion_score"),
            "discussion_reason": ann.get("discussion_reason"),
        })

    print(f"Merged: {len(merged)} records  ({unmatched} unmatched / skipped)")

    # ── filter ────────────────────────────────────────────────────────────────
    filtered = [
        r for r in merged
        if (r["num_comments"] or 0) >= min_comments
        and (r.get("topic") or {}).get("top", "UNCLEAR") != "UNCLEAR"
        and (r.get("discussion_score") or 0) >= min_score
    ]
    print(f"After filter (num_comments>={min_comments}, topic!=UNCLEAR, score>={min_score}): {len(filtered)} records")

    # ── topic-group peer filter ───────────────────────────────────────────────
    # Leaf group preferred; fall back to mid if leaf count is below min_peers.
    leaf_counts: Counter = Counter(
        (r.get("topic") or {}).get("leaf")
        for r in filtered
        if (r.get("topic") or {}).get("leaf") is not None
    )
    mid_counts: Counter = Counter(
        (r.get("topic") or {}).get("mid")
        for r in filtered
        if (r.get("topic") or {}).get("mid") is not None
    )

    peer_filtered = []
    for r in filtered:
        leaf = (r.get("topic") or {}).get("leaf")
        mid  = (r.get("topic") or {}).get("mid")
        if leaf is not None and leaf_counts[leaf] >= min_peers:
            r = dict(r)
            r["topic_group"]       = leaf
            r["topic_group_level"] = "leaf"
            peer_filtered.append(r)
        elif mid is not None and mid_counts[mid] >= min_peers:
            r = dict(r)
            r["topic_group"]       = mid
            r["topic_group_level"] = "mid"
            peer_filtered.append(r)
    print(f"After peer filter (>={min_peers} posts sharing topic group): {len(peer_filtered)} records")

    # ── select top-N per subreddit ────────────────────────────────────────────
    by_sub: dict = {}
    for r in peer_filtered:
        by_sub.setdefault(r["community"], []).append(r)

    selected      = []
    stats_per_sub = {}
    for sub, posts_sub in sorted(by_sub.items()):
        ranked = sorted(posts_sub, key=lambda r: r["discussion_score"], reverse=True)
        top    = ranked[:top_n]
        selected.extend(top)
        stats_per_sub[sub] = {
            "eligible":  len(posts_sub),
            "selected":  len(top),
            "score_min": round(top[-1]["discussion_score"], 3) if top else None,
            "score_max": round(top[0]["discussion_score"],  3) if top else None,
        }
        if top:
            print(
                f"  {sub:<30} eligible={len(posts_sub):3d}  selected={len(top):2d}"
                f"  score range [{top[-1]['discussion_score']:.2f}–{top[0]['discussion_score']:.2f}]"
            )
        else:
            print(f"  {sub:<30} eligible={len(posts_sub):3d}  selected=0")

    print(f"\nTotal selected: {len(selected)}")

    out_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    run_end  = time.time()
    metadata = {
        "run":              next_num,
        "generated_at":     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start)),
        "completed_at":     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_end)),
        "duration_seconds": round(run_end - run_start, 2),
        "scrape_dir":       str(scrape_dir),
        "sources": {
            "posts_file":    str(posts_file),
            "analysis_file": str(analysis_file),
        },
        "filters": {
            "min_comments":         min_comments,
            "min_discussion_score": min_score,
            "min_topic_peers":      min_peers,
            "exclude_topic_top":    ["UNCLEAR"],
        },
        "selection": {
            "top_n_per_subreddit": top_n,
            "rank_by":             "discussion_score",
        },
        "counts": {
            "posts_total":       len(posts),
            "posts_merged":      len(merged),
            "posts_filtered":    len(filtered),
            "posts_peer_filter": len(peer_filtered),
            "posts_selected":    len(selected),
            "unmatched":         unmatched,
        },
        "per_subreddit": stats_per_sub,
        "output_file":   str(out_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Results  → {out_path}")
    print(f"  Metadata → {meta_path}")
    return str(out_path)
