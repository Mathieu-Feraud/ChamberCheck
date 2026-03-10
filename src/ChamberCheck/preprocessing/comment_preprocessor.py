"""
Comment preprocessing: trunk-based random sampling with content-length filtering.

Reads the raw ``comments_NNN.json`` produced by the comment scraper, applies a
minimum content-length filter (pruning short comments **and their entire reply
subtrees**), then samples up to ``max_sample_per_post`` comments per post.

Sampling strategy
-----------------
Top-level ("trunk") comments are shuffled in random order (seeded for
reproducibility).  Trunks are accumulated one at a time — each trunk together
with every descendant reply counts toward the target.  Sampling stops once the
running total reaches ``max_sample_per_post`` or all trunks have been included.
A trunk is **never** split: the final trunk may push the total past the target.

Output
------
``<scrape_dir>/comments/comments_filtered_NNN.json``
``<scrape_dir>/comments/comments_filtered_NNN_metadata.json``
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import Config
from ..constants import (
    COMMENT_PREPROCESSING_MAX_SAMPLE,
    COMMENT_PREPROCESSING_MIN_CONTENT_LENGTH,
)


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def _build_tree(comments: List[dict], post_id: str):
    """Build mutable tree structures from a flat comment list.

    Returns:
        comment_map:  {comment_id: comment_dict}
        children_map: {parent_id: [child_comment_id, ...]}
        parent_map:   {comment_id: parent_id}
    """
    comment_map: Dict[str, dict] = {}
    children_map: Dict[str, List[str]] = {post_id: []}
    parent_map: Dict[str, str] = {}

    for c in comments:
        cid = c["comment_id"]
        pid = c["parent_id"]
        comment_map[cid] = c
        children_map.setdefault(pid, []).append(cid)
        parent_map[cid] = pid

    return comment_map, children_map, parent_map


def _prune_node(
    node_id: str,
    comment_map: Dict[str, dict],
    children_map: Dict[str, List[str]],
    parent_map: Dict[str, str],
) -> None:
    """Remove *node_id* and all its descendants from the tree structures.

    Also removes the node from its parent's children list so subsequent
    traversals do not encounter stale references.
    """
    # Detach from parent
    pid = parent_map.get(node_id)
    if pid is not None and pid in children_map:
        try:
            children_map[pid].remove(node_id)
        except ValueError:
            pass

    # DFS removal of node + all descendants
    stack = [node_id]
    while stack:
        cid = stack.pop()
        comment_map.pop(cid, None)
        children = children_map.pop(cid, [])
        parent_map.pop(cid, None)
        stack.extend(children)


def _apply_length_filter(
    post_id: str,
    comment_map: Dict[str, dict],
    children_map: Dict[str, List[str]],
    parent_map: Dict[str, str],
    min_length: int,
) -> int:
    """BFS top-down: prune any subtree whose root comment is shorter than *min_length*.

    Processing top-down means we never waste time visiting descendants of an
    already-pruned node.

    Returns:
        Number of subtree roots pruned.
    """
    pruned = 0
    queue = list(children_map.get(post_id, []))

    while queue:
        next_queue: List[str] = []
        for cid in queue:
            if cid not in comment_map:
                continue
            content = comment_map[cid].get("content") or ""
            if len(content) < min_length:
                _prune_node(cid, comment_map, children_map, parent_map)
                pruned += 1
            else:
                next_queue.extend(list(children_map.get(cid, [])))
        queue = next_queue

    return pruned


def _collect_subtree(
    node_id: str,
    comment_map: Dict[str, dict],
    children_map: Dict[str, List[str]],
) -> List[dict]:
    """DFS pre-order collect of *node_id* and all its descendants."""
    result: List[dict] = []
    stack = [node_id]
    while stack:
        cid = stack.pop()
        if cid in comment_map:
            result.append(comment_map[cid])
            # Reverse so left-to-right sibling ordering is preserved after DFS
            for child_id in reversed(children_map.get(cid, [])):
                stack.append(child_id)
    return result


def _sample_post_comments(
    comments: List[dict],
    post_id: str,
    max_sample: int,
    min_length: int,
    rng: random.Random,
) -> List[dict]:
    """Sample up to *max_sample* comments for a single post.

    Steps:
        1. Build the comment tree.
        2. Prune every subtree whose root is shorter than *min_length*.
        3. Shuffle remaining trunk (depth-0) comments randomly (without
           replacement).
        4. Accumulate trunk + all descendants until total >= *max_sample* or
           all trunks are exhausted.  The final trunk is never split.

    Returns a flat ordered list of sampled comment dicts.
    """
    if not comments:
        return []

    comment_map, children_map, parent_map = _build_tree(comments, post_id)
    _apply_length_filter(post_id, comment_map, children_map, parent_map, min_length)

    # Trunk = top-level comments whose parent is the post itself
    trunks = [t for t in children_map.get(post_id, []) if t in comment_map]
    rng.shuffle(trunks)

    sampled: List[dict] = []
    for trunk_id in trunks:
        subtree = _collect_subtree(trunk_id, comment_map, children_map)
        sampled.extend(subtree)
        if len(sampled) >= max_sample:
            break

    return sampled


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _latest_raw_comments(comments_dir: Path) -> Path:
    """Return the most recent ``comments_NNN.json`` (excludes filtered files)."""
    candidates = sorted(
        f for f in comments_dir.glob("comments_*.json")
        if "_metadata" not in f.name and "filtered" not in f.name
    )
    if not candidates:
        raise FileNotFoundError(f"No comments_*.json found in {comments_dir}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_comments(
    scrape_dir: Optional[Union[str, Path]] = None,
    config_path: str = "config/config.yaml",
    seed: int = 42,
) -> str:
    """Sample and filter comments from a scrape folder.

    Reads:
        ``<scrape_dir>/comments/comments_NNN.json``  (latest raw scrape)

    Writes:
        ``<scrape_dir>/comments/comments_filtered_NNN.json``
        ``<scrape_dir>/comments/comments_filtered_NNN_metadata.json``

    Args:
        scrape_dir:  Path to the scrape folder.  Auto-detects the latest
                     ``data/raw/scrape_*`` folder if *None*.
        config_path: Path to the unified YAML config file.
        seed:        Random seed for reproducible sampling.

    Returns:
        Absolute path to the written ``comments_filtered_NNN.json`` file.
    """
    cfg        = Config(config_path)
    max_sample = cfg.get("comment_preprocessing.max_sample_per_post", COMMENT_PREPROCESSING_MAX_SAMPLE)
    min_length = cfg.get("comment_preprocessing.min_content_length",  COMMENT_PREPROCESSING_MIN_CONTENT_LENGTH)

    if scrape_dir is None:
        raw_dir    = Path("data/raw")
        scrape_dir = sorted(p for p in raw_dir.glob("scrape_*") if p.is_dir())[-1]
    scrape_dir = Path(scrape_dir)

    comments_dir = scrape_dir / "comments"
    raw_path     = _latest_raw_comments(comments_dir)

    # Auto-increment output run number (exclude *_metadata.json from count)
    existing  = [
        f for f in comments_dir.glob("comments_filtered_*.json")
        if "_metadata" not in f.name
    ]
    next_num  = len(existing) + 1
    out_path  = comments_dir / f"comments_filtered_{next_num:03d}.json"
    meta_path = comments_dir / f"comments_filtered_{next_num:03d}_metadata.json"

    print(f"Raw comments : {raw_path}")
    print(f"Output       : {out_path}")
    print(f"Config       : max_sample={max_sample}, min_content_length={min_length}, seed={seed}")
    print()

    rng       = random.Random(seed)
    run_start = time.time()

    raw_data = json.loads(raw_path.read_text(encoding="utf-8"))

    results:        List[dict] = []
    total_in:       int = 0
    total_out:      int = 0
    per_post_stats: List[dict] = []

    for i, post_entry in enumerate(raw_data, 1):
        post_id      = post_entry["post_id"]
        raw_comments = post_entry.get("comments", [])
        n_in         = len(raw_comments)
        total_in    += n_in

        sampled  = _sample_post_comments(raw_comments, post_id, max_sample, min_length, rng)
        n_out    = len(sampled)
        total_out += n_out

        community   = post_entry.get("community", "")
        title_short = (post_entry.get("title") or "")[:50]
        print(
            f"  [{i:03d}/{len(raw_data)}] {community:<25} "
            f"{n_in:5d} -> {n_out:4d} comments  ({title_short})"
        )

        results.append({
            **{k: v for k, v in post_entry.items() if k != "comments"},
            "comments":          sampled,
            "comment_count":     n_out,
            "comment_count_raw": n_in,
        })
        per_post_stats.append({
            "post_id":   post_id,
            "community": community,
            "raw":       n_in,
            "sampled":   n_out,
        })

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    run_end  = time.time()
    metadata = {
        "run":              next_num,
        "generated_at":     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start)),
        "completed_at":     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_end)),
        "duration_seconds": round(run_end - run_start, 2),
        "scrape_dir":       str(scrape_dir),
        "source_file":      str(raw_path),
        "seed":             seed,
        "filters": {
            "min_content_length":  min_length,
            "max_sample_per_post": max_sample,
            "sampling_strategy":   "trunk_random_without_replacement",
        },
        "counts": {
            "posts":            len(raw_data),
            "comments_raw":     total_in,
            "comments_sampled": total_out,
        },
        "per_post":    per_post_stats,
        "output_file": str(out_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"\nDone: {total_in:,} raw -> {total_out:,} sampled comments"
        f" across {len(raw_data)} posts"
    )
    print(f"  Results  -> {out_path}")
    print(f"  Metadata -> {meta_path}")
    return str(out_path)
