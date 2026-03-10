"""
Batch comment analysis orchestrator.

Handles multi-file comment analysis with LLM metrics computation.
This is the core logic that can be imported and reused.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from .comment_analyzer import CommentAnalyzer
from .llm_provider import LLMProvider
from ..constants import (
    OUTPUT_DIR,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OPENAI_MODEL,
)
from ..config import Config


def load_comments(file_path: str) -> List[Dict[str, Any]]:
    """Load comments from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict) and "comments" in data:
            return data["comments"]
        return data


def load_comments_and_posts(file_path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load comments and posts from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("comments", []), data.get("posts", [])

    return data, []


def get_next_comment_analysis_run_number(output_dir: str) -> int:
    """Get the next run number for comment analysis outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob("comment_analysis_*.json"))
    run_numbers = []
    for file in existing:
        stem = file.stem
        if stem.startswith("comment_analysis_"):
            try:
                run_num = int(stem.split("comment_analysis_")[-1])
                run_numbers.append(run_num)
            except ValueError:
                continue

    return max(run_numbers, default=0) + 1


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


def _extract_subreddit_and_keywords(file_name: str) -> tuple:
    """Extract subreddit and keywords from filename."""
    subreddit = file_name.split('_')[0]
    keywords = None
    
    if "_comments_json_" in file_name:
        parts = file_name.split("_comments_json_")
        if len(parts) > 1:
            middle = parts[1]
            middle_parts = middle.rsplit("_", 2)
            if len(middle_parts) > 0 and middle_parts[0]:
                keywords = middle_parts[0]
    
    return subreddit, keywords


def _calculate_averages(analysis_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average scores from analysis results."""
    def _numeric_or_none(val):
        try:
            if isinstance(val, str):
                return float(val) if val.strip().upper() != "N/A" else None
            return float(val)
        except Exception:
            return None

    valid_results = [r for r in analysis_results if 'error' not in r]
    if not valid_results:
        return {}
    
    def _avg(key):
        nums = [_numeric_or_none(r.get(key)) for r in valid_results]
        nums = [n for n in nums if n is not None]
        return sum(nums) / len(nums) if nums else 0.0
    
    return {
        "argument_narrowness": round(_avg('argument_narrowness'), 2),
        "hostility": round(_avg('hostility'), 2),
        "suppression": round(_avg('suppression'), 2),
        "epistemic_closure": round(_avg('epistemic_closure'), 2),
        "argument_avoidance": round(_avg('argument_avoidance'), 2),
        "echo_chamber_score": round(_avg('echo_chamber_score'), 2)
    }


def analyze_comments_file(
    comments_file: str,
    provider: LLMProvider,
    model_name: str,
    limit: int = 100,
    mode: str = "top",
    comment_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single comments file and return comprehensive metadata.
    
    Args:
        comments_file: Path to JSON comments file
        provider: Initialized LLMProvider instance
        model_name: Name of the model being used
        limit: Number of comments to analyze
        mode: "top" for most upvoted, "bottom" for most downvoted
    
    Returns:
        Dictionary with metadata and results, or None if error
    """
    file_start_time = time.time()

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "module": "batch_analyzer.analyze_comments_file",
        "input_file": comments_file,
        "provider": provider.__class__.__name__,
        "model": model_name,
        "limit": limit,
        "mode": mode,
        "comment_ids": comment_ids or [],
    }
    
    # Load comments
    try:
        all_comments, all_posts = load_comments_and_posts(comments_file)
        metadata["total_comments_in_file"] = len(all_comments)
    except FileNotFoundError:
        metadata["error"] = f"File not found: {comments_file}"
        return metadata

    # Filter comments
    if comment_ids:
        comment_id_set = set(comment_ids)
        comments_to_analyze = [c for c in all_comments if c.get("comment_id") in comment_id_set]
        metadata["comments_selected_for_analysis"] = len(comments_to_analyze)
        metadata["filter_method"] = "comment_id_filter"
    else:
        direction = "most upvoted" if mode == "top" else "most downvoted"
        comments_to_analyze = filter_comments_by_score(all_comments, limit=limit, mode=mode)
        metadata["comments_selected_for_analysis"] = len(comments_to_analyze)
        metadata["filter_method"] = f"{direction.replace(' ', '_')}_{limit}"
    
    # Extract subreddit and keywords from filename
    file_name = Path(comments_file).stem
    subreddit, keywords = _extract_subreddit_and_keywords(file_name)
    
    metadata["subreddit"] = subreddit
    if keywords:
        metadata["keywords"] = keywords
    
    # Create analyzer and build parent map
    analyzer = CommentAnalyzer(provider, subreddit=subreddit)
    parent_map = build_parent_map(all_comments)
    posts_map = {p.get("post_id"): p for p in all_posts if p.get("post_id")}
    
    # Analyze comments
    analysis_start_time = time.time()
    try:
        analysis_results = analyzer.analyze_batch(
            comments_to_analyze,
            parent_map,
            posts_map=posts_map,
            source_file=comments_file,
        )
    except Exception as e:
        metadata["error"] = f"Analysis failed: {str(e)}"
        return metadata
    
    analysis_duration = time.time() - analysis_start_time
    metadata["analysis_duration_seconds"] = round(analysis_duration, 2)
    
    # Save results - preserve folder structure from input
    input_path = Path(comments_file)
    input_parent = input_path.parent.name  # e.g., "scrape_001"
    structured_output_dir = f"{OUTPUT_DIR}/{input_parent}"
    
    Path(structured_output_dir).mkdir(parents=True, exist_ok=True)

    run_number = get_next_comment_analysis_run_number(structured_output_dir)
    output_file = f"{structured_output_dir}/comment_analysis_{run_number:03d}.json"
    metadata_file = f"{structured_output_dir}/comment_analysis_metadata_{run_number:03d}.json"
    
    save_analysis(analysis_results, output_file)
    save_metadata(metadata, metadata_file)
    
    metadata["output_file"] = output_file
    metadata["metadata_file"] = metadata_file
    
    # Calculate summary statistics
    successful = sum(1 for r in analysis_results if 'error' not in r)
    errors = len(analysis_results) - successful
    
    metadata["successful_analyses"] = successful
    metadata["failed_analyses"] = errors
    
    average_scores = _calculate_averages(analysis_results)
    if average_scores:
        metadata["average_scores"] = average_scores
    
    # Total run time
    total_duration = time.time() - file_start_time
    metadata["total_run_duration_seconds"] = round(total_duration, 2)
    
    return metadata


def batch_analyze_comments(
    comment_files: Optional[List[str]] = None,
    provider: LLMProvider = None,
    model_name: str = None,
    limit: int = 100,
    mode: str = "top",
    comment_ids: Optional[List[str]] = None,
    input_folder: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze multiple comments files in batch.
    
    This is the main orchestrator function that users can call directly.
    
    Args:
        comment_files: List of paths to JSON comments files
        provider: Initialized LLMProvider instance
        model_name: Name of the model being used
        limit: Number of comments to analyze per file
        mode: "top" for most upvoted, "bottom" for most downvoted
        comment_ids: Optional list of comment IDs to analyze
        input_folder: Optional folder to scan for comment JSON files
    
    Returns:
        List of metadata dictionaries (one per file processed)
    """
    if not provider:
        config = Config()
        provider_name = config.get("llm.provider") or DEFAULT_LLM_PROVIDER
        api_key = config.get("llm.api_key")
        provider = LLMProvider.from_config(provider_name, api_key=api_key, model=model_name)

    if not model_name:
        model_name = getattr(provider, "model", None) or DEFAULT_OPENAI_MODEL

    if input_folder:
        folder = Path(input_folder)
        if not folder.exists():
            raise ValueError(f"input_folder not found: {input_folder}")

        discovered_files = []
        for path in sorted(folder.glob("*.json")):
            name = path.name
            if name.endswith("_metadata.json"):
                continue
            if name.startswith("comment_prompts_") or name.startswith("comment_analysis_"):
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("comments"), list):
                    discovered_files.append(str(path))
            except Exception:
                continue

        comment_files = discovered_files

    if not comment_files:
        raise ValueError("No comment files provided or discovered")

    all_metadata = []

    for comments_file in comment_files:
        metadata = analyze_comments_file(
            comments_file=comments_file,
            provider=provider,
            model_name=model_name,
            limit=limit,
            mode=mode,
            comment_ids=comment_ids,
        )
        if metadata:
            all_metadata.append(metadata)
    
    return all_metadata
