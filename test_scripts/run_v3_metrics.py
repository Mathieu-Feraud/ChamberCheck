"""
Run V3 echo-chamber metrics on a completed comment analysis.

Usage
-----
  python test_scripts/run_v3_metrics.py <scrape_dir> <analysis_file>

  scrape_dir      Path to the scrape folder, e.g. data/raw/scrape_006
  analysis_file   Path to the comment_analysis_NNN.json to score

Output is written to data/output/<scrape_name>/v3_metrics_NNN.json + metadata.

Examples
--------
  python test_scripts/run_v3_metrics.py data/raw/scrape_006 \
      data/output/scrape_006/comment_analysis_001.json
"""

import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

# ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from ChamberCheck.CC_derived_metrics.derived_metrics import MetricResult, V3Metrics


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """Recursively convert MetricResult / numpy scalars to plain Python."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(i) for i in obj]
    # numpy scalar guard
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    return obj


# ---------------------------------------------------------------------------
# Auto-increment output filename
# ---------------------------------------------------------------------------

def _next_run_number(output_dir: Path, prefix: str) -> int:
    existing = sorted(
        p for p in output_dir.glob(f"{prefix}_???.json")
        if "metadata" not in p.name
    )
    if not existing:
        return 1
    last = existing[-1].stem  # e.g. "v3_metrics_003"
    try:
        return int(last.split("_")[-1]) + 1
    except ValueError:
        return len(existing) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    scrape_dir = Path(sys.argv[1])
    analysis_file = Path(sys.argv[2])

    if not scrape_dir.is_dir():
        print(f"[ERROR] scrape_dir not found: {scrape_dir}")
        sys.exit(1)
    if not analysis_file.is_file():
        print(f"[ERROR] analysis_file not found: {analysis_file}")
        sys.exit(1)

    # locate the comments_filtered file
    comments_dir = scrape_dir / "comments"
    filtered_files = sorted(
        p for p in comments_dir.glob("comments_filtered_???.json")
        if p.is_file()
    )
    if not filtered_files:
        print(f"[ERROR] No comments_filtered_NNN.json found in {comments_dir}")
        sys.exit(1)
    filtered_file = filtered_files[-1]  # latest

    print(f"[INFO] Analysis file  : {analysis_file}")
    print(f"[INFO] Filtered file  : {filtered_file}")

    # output dir: data/output/<scrape_name>/
    scrape_name = scrape_dir.name  # e.g. "scrape_006"
    output_dir = scrape_dir.parent.parent.parent / "data" / "output" / scrape_name
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "v3_metrics"
    run_num = _next_run_number(output_dir, prefix)
    run_str = f"{run_num:03d}"
    output_file = output_dir / f"{prefix}_{run_str}.json"
    meta_file = output_dir / f"{prefix}_metadata_{run_str}.json"

    # load + compute
    print("[INFO] Loading data and building indexes …")
    t0 = time.time()
    metrics = V3Metrics.from_files(str(analysis_file), str(filtered_file))
    n_comments = len(metrics.comments)
    subreddits = metrics.get_subreddits()
    print(f"[INFO] {n_comments} merged comments across {len(subreddits)} subreddit(s)")

    print("[INFO] Computing aggregate metrics …")
    aggregate = metrics.compute_all()

    print("[INFO] Computing per-subreddit metrics …")
    by_subreddit = metrics.compute_all_by_subreddit()

    print("[INFO] Computing per-subreddit per-topic metrics …")
    by_subreddit_topic = metrics.compute_all_by_subreddit_topic()
    n_topics_total = sum(len(t) for t in by_subreddit_topic.values())
    print(f"[INFO] {n_topics_total} eligible topic(s) across {len(by_subreddit_topic)} subreddit(s)")

    elapsed = time.time() - t0

    results = {
        "aggregate": _jsonable(aggregate),
        "by_subreddit": _jsonable(by_subreddit),
        "by_subreddit_topic": _jsonable(by_subreddit_topic),
    }
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    metadata = {
        "analysis_file": str(analysis_file),
        "filtered_file": str(filtered_file),
        "n_comments": n_comments,
        "subreddits": subreddits,
        "n_topics_per_subreddit": {sr: list(topics.keys()) for sr, topics in by_subreddit_topic.items()},
        "elapsed_seconds": round(elapsed, 2),
    }
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[INFO] Results written  → {output_file}")
    print(f"[INFO] Metadata written → {meta_file}")
    print(f"[INFO] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
