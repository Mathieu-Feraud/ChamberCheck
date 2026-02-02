"""
Analyze comments for echo chamber metrics using LLM.

Supports analyzing one or more raw comment files and selecting either the most upvoted
or most downvoted comments for analysis.

Usage:
    python scripts/analyze_comments.py file1.json [file2.json ...] \
            --provider openai|anthropic --model MODEL --top-n N --mode top|bottom
"""

import json
import sys
import time
from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ChamberCheck.analysis.comment_analyzer import CommentAnalyzer
from ChamberCheck.analysis.llm_provider import LLMProvider
from ChamberCheck.utils import setup_logger
from ChamberCheck.constants import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_ANALYSIS_TOP_N,
    TIMESTAMP_FORMAT,
    PROCESSED_DATA_DIR,
)


def load_comments(file_path: str) -> List[Dict[str, Any]]:
    """Load comments from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_comments_by_score(
    comments: List[Dict[str, Any]],
    limit: int = 100,
    mode: str = "top",
) -> List[Dict[str, Any]]:
    """
    Filter comments by score, keeping either most upvoted (top) or most downvoted (bottom).

    Args:
        comments: List of comments
        limit: Number of comments to keep
        mode: "top" for most upvoted, "bottom" for most downvoted

    Returns:
        Top N comments sorted by score in chosen direction
    """
    reverse = True if mode == "top" else False
    sorted_comments = sorted(
        comments,
        key=lambda c: c.get('metadata', {}).get('score', 0),
        reverse=reverse,
    )
    return sorted_comments[:limit]


def build_parent_map(comments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping of comment_id -> comment for quick parent lookup.
    
    Args:
        comments: List of all comments
    
    Returns:
        Dictionary mapping comment_id to comment
    """
    return {c['comment_id']: c for c in comments}


def save_analysis(analysis_results: List[Dict[str, Any]], output_path: str):
    """Save analysis results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, default=str)


def save_metadata(metadata: Dict[str, Any], output_path: str):
    """Save run metadata to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Reddit comments for echo chamber metrics")
    parser.add_argument("comment_files", nargs="*", help="Path(s) to raw comments JSON files")
    parser.add_argument("--config", type=str, default="config/config.analyze.json",
                        help="Path to analysis config JSON (default: config/config.analyze.json)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None,
                        help="LLM provider to use (overrides config)")
    parser.add_argument("--model", default=None, help="Model name (optional, overrides config)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of comments to analyze per file (overrides config)")
    parser.add_argument("--mode", choices=["top", "bottom"], default=None,
                        help="Select most upvoted (top) or most downvoted (bottom) comments (overrides config)")
    parser.add_argument("--all-raw", action="store_true",
                        help="Process all JSON files under data/raw (overrides config)")
    return parser.parse_args()


def process_comments_file(
    comments_file: str,
    provider: LLMProvider,
    model_name: str,
    limit: int,
    mode: str,
    logger,
):
    """Process a single comments file and return metadata."""

    file_start_time = time.time()

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "script": "analyze_comments.py",
        "input_file": comments_file,
        "provider": provider.__class__.__name__,
        "model": model_name,
        "limit": limit,
        "mode": mode,
    }
    
    print("=" * 70)
    print("ChamberCheck - Comment Analysis")
    print("=" * 70)
    print(f"\n📂 Loading comments from: {comments_file}")
    
    # Load comments
    try:
        all_comments = load_comments(comments_file)
        print(f"✓ Loaded {len(all_comments)} comments")
        metadata["total_comments_in_file"] = len(all_comments)
    except FileNotFoundError:
        print(f"❌ File not found: {comments_file}")
        return None

    # Filter to top or bottom N by score
    direction = "most upvoted" if mode == "top" else "most downvoted"
    print(f"\n🔥 Selecting {limit} {direction} comments (by score)...")
    comments_to_analyze = filter_comments_by_score(all_comments, limit=limit, mode=mode)
    print(f"✓ Selected {len(comments_to_analyze)} comments for analysis")

    metadata["comments_selected_for_analysis"] = len(comments_to_analyze)
    metadata["filter_method"] = f"{direction.replace(' ', '_')}_{limit}"
    
    # Show top comments
    print("\nSelected comments (sample):")
    for i, c in enumerate(comments_to_analyze[:5], 1):
        score = c.get('metadata', {}).get('score', 0)
        print(f"  {i}. Score: {score:+6d} | {c['content'][:60]}...")
    
    # Initialize LLM provider
    # Extract subreddit from filename
    file_name = Path(comments_file).stem
    subreddit = file_name.split('_')[0]  # e.g., "samharris" from "samharris_comments_json_..."
    
    # Extract keyword from filename if present
    keywords = None
    if "_comments_json_" in file_name:
        parts = file_name.split("_comments_json_")
        if len(parts) > 1:
            middle = parts[1]
            middle_parts = middle.rsplit("_", 2)
            if len(middle_parts) > 0 and middle_parts[0]:
                keywords = middle_parts[0]
    
    metadata["subreddit"] = subreddit
    if keywords:
        metadata["keywords"] = keywords
    
    # Create analyzer
    analyzer = CommentAnalyzer(provider, subreddit=subreddit)
    
    # Build parent map for context
    parent_map = build_parent_map(all_comments)
    
    # Analyze comments
    print(f"\n📊 Analyzing {len(comments_to_analyze)} comments...")
    print("   (This may take a few minutes)\n")
    
    analysis_start_time = time.time()
    analysis_results = analyzer.analyze_batch(comments_to_analyze, parent_map)
    analysis_duration = time.time() - analysis_start_time
    
    metadata["analysis_duration_seconds"] = round(analysis_duration, 2)
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    
    # Extract keyword from filename if present
    keyword_part = ""
    if "_comments_json_" in file_name:
        # Extract part between "_comments_json_" and timestamp
        parts = file_name.split("_comments_json_")
        if len(parts) > 1:
            # Remove timestamp from end
            middle = parts[1]
            # Try to detect if there's a keyword (anything before the timestamp)
            # Timestamp format is like 20260119_142220
            middle_parts = middle.rsplit("_", 2)  # Split from right to separate timestamp
            if len(middle_parts) > 0 and middle_parts[0]:
                keyword_part = "_" + middle_parts[0]
    
    output_file = f'data/processed/{subreddit}_analysis{keyword_part}_{timestamp}.json'
    metadata_file = f'data/processed/{subreddit}_analysis{keyword_part}_{timestamp}_metadata.json'
    
    save_analysis(analysis_results, output_file)
    print(f"✓ Analysis saved: {output_file}")
    
    save_metadata(metadata, metadata_file)
    print(f"✓ Metadata saved: {metadata_file}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in analysis_results if 'error' not in r)
    errors = len(analysis_results) - successful
    
    print(f"Successfully analyzed: {successful}/{len(analysis_results)}")
    if errors > 0:
        print(f"Errors: {errors}")
    
    metadata["successful_analyses"] = successful
    metadata["failed_analyses"] = errors
    
    # Calculate averages
    def _numeric_or_none(val):
        try:
            if isinstance(val, str):
                return float(val) if val.strip().upper() != "N/A" else None
            return float(val)
        except Exception:
            return None
    
    valid_results = [r for r in analysis_results if 'error' not in r]
    if valid_results:
        def _avg(key):
            nums = [_numeric_or_none(r.get(key)) for r in valid_results]
            nums = [n for n in nums if n is not None]
            return sum(nums) / len(nums) if nums else 0.0
        
        avg_narrowness = _avg('argument_narrowness')
        avg_hostility = _avg('hostility')
        avg_suppression = _avg('suppression')
        avg_closure = _avg('epistemic_closure')
        avg_avoidance = _avg('argument_avoidance')
        avg_echo_score = _avg('echo_chamber_score')
        
        metadata["average_scores"] = {
            "argument_narrowness": round(avg_narrowness, 2),
            "hostility": round(avg_hostility, 2),
            "suppression": round(avg_suppression, 2),
            "epistemic_closure": round(avg_closure, 2),
            "argument_avoidance": round(avg_avoidance, 2),
            "echo_chamber_score": round(avg_echo_score, 2)
        }
        
        print(f"\nAverage scores (N/A skipped):")
        print(f"  Argument Narrowness:  {avg_narrowness:.1f}/10")
        print(f"  Hostility:            {avg_hostility:.1f}/10")
        print(f"  Suppression:          {avg_suppression:.1f}/10")
        print(f"  Epistemic Closure:    {avg_closure:.1f}/10")
        print(f"  Argument Avoidance:   {avg_avoidance:.1f}/10")
        print(f"  Echo Chamber Score:   {avg_echo_score:.1f}/50")
    
    # Total run time
    total_duration = time.time() - file_start_time
    metadata["total_run_duration_seconds"] = round(total_duration, 2)

    print("\n" + "=" * 70)
    print("✅ Analysis completed for file!")
    print("=" * 70)
    print(f"Total time: {total_duration/60:.1f} minutes")

    return metadata


def main():
    args = parse_args()
    logger = setup_logger("analyze_comments")
    run_start_time = time.time()

    # Load config if present
    config_data = {}
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config {args.config}: {e}")

    # Resolve settings: CLI override > config > defaults
    provider_name = args.provider or config_data.get("provider") or DEFAULT_LLM_PROVIDER
    model_name = args.model or config_data.get("model")
    limit = args.limit or config_data.get("limit") or 100
    mode = args.mode or config_data.get("mode") or "top"
    all_raw_flag = args.all_raw or config_data.get("all_raw", False)

    # Build file list from CLI + config
    file_list = list(dict.fromkeys(args.comment_files))  # preserve order, dedupe
    config_files = config_data.get("comment_files", []) if isinstance(config_data.get("comment_files"), list) else []
    file_list.extend([f for f in config_files if f not in file_list])

    if all_raw_flag:
        raw_dir = Path("data/raw")
        raw_files = sorted(raw_dir.glob("*.json"))
        file_list.extend([str(p) for p in raw_files if str(p) not in file_list])

    if not file_list:
        print("No input files specified. Provide paths, config comment_files, or use --all-raw.")
        sys.exit(1)

    # Initialize LLM provider once
    print(f"\n🤖 Initializing {provider_name.upper()} provider...")
    if model_name:
        print(f"   Model: {model_name}")
    try:
        provider = LLMProvider.from_config(provider_name, model=model_name)
        print(f"✓ {provider_name.upper()} provider ready")
    except ValueError as e:
        print(f"❌ Invalid provider: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Provider initialization error: {e}")
        print(f"   Make sure {provider_name.upper()}_API_KEY environment variable is set")
        sys.exit(1)

    all_metadata = []
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    for comments_file in file_list:
        metadata = process_comments_file(
            comments_file=comments_file,
            provider=provider,
            model_name=model_name,
            limit=limit,
            mode=mode,
            logger=logger,
        )
        if metadata:
            all_metadata.append(metadata)

    total_duration = time.time() - run_start_time
    print("\n" + "=" * 70)
    print("🏁 All files processed")
    print("=" * 70)
    print(f"Total runtime: {total_duration/60:.1f} minutes")


if __name__ == '__main__':
    main()
