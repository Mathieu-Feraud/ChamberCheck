"""
Prompt exporter for human evaluation.

Creates prompt + empty JSON response templates for comments.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .prompt_builder import build_comment_prompt
from ..constants import OUTPUT_DIR


def _load_post_comment_data(file_path: str) -> Dict[str, Any]:
    """Load post/comment data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _get_next_prompt_run_number(output_dir: str) -> int:
    """Get the next run number for prompt export outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob("comment_prompts_*.json"))
    run_numbers = []
    for file in existing:
        stem = file.stem
        if stem.startswith("comment_prompts_"):
            try:
                run_num = int(stem.split("comment_prompts_")[-1])
                run_numbers.append(run_num)
            except ValueError:
                continue

    return max(run_numbers, default=0) + 1


def _empty_human_response_template() -> Dict[str, Any]:
    """Return an empty JSON template for human completion."""
    return {
        "topic": {
            "label": "",
            "stance": {
                "value": None
            }
        },
        "substantiveness": None,
        "discrediting": None,
        "defensive": None,
        "toxicity": None,
        "politeness": None,
        "emotion": {
            "anger": None,
            "fear": None,
            "outrage": None,
            "anxiety": None,
            "disgust": None
        },
        "epistemic risk": None
    }


def export_comment_prompts(
    input_file: str,
    subreddit: Optional[str] = None,
    comment_ids: Optional[List[str]] = None,
    output_dir: str = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Export prompts and empty JSON templates for human annotation.

    Args:
        input_file: Path to JSON file with posts/comments
        subreddit: Subreddit name override (optional)
        comment_ids: Optional list of comment IDs to export
        output_dir: Output directory for JSON files

    Returns:
        Metadata about the export and output file paths
    """
    data = _load_post_comment_data(input_file)

    posts = data.get("posts", []) if isinstance(data, dict) else []
    comments = data.get("comments", []) if isinstance(data, dict) else data

    posts_map = {p.get("post_id"): p for p in posts}
    comments_map = {c.get("comment_id"): c for c in comments}

    if comment_ids:
        comment_id_set = set(comment_ids)
        comments_to_export = [c for c in comments if c.get("comment_id") in comment_id_set]
    else:
        comments_to_export = comments

    inferred_subreddit = subreddit or (data.get("posts", [{}])[0].get("community") if posts else "unknown")

    export_rows = []
    for comment in comments_to_export:
        parent = None
        parent_is_post = False

        parent_id = comment.get("parent_id")
        if parent_id and parent_id in comments_map:
            parent = comments_map[parent_id]
            parent_is_post = False
        elif comment.get("post_id") in posts_map:
            parent = posts_map[comment.get("post_id")]
            parent_is_post = True

        prompt = build_comment_prompt(
            comment=comment,
            parent=parent,
            parent_is_post=parent_is_post,
            source_file=input_file,
            posts_map=posts_map,
            comments_map=comments_map,
        )
        prompt_lines = prompt.splitlines()

        export_rows.append({
            "comment_id": comment.get("comment_id"),
            "post_id": comment.get("post_id"),
            "parent_id": comment.get("parent_id"),
            "prompt_lines": prompt_lines,
            "human_response": _empty_human_response_template(),
        })

    run_number = _get_next_prompt_run_number(output_dir)
    output_path = f"{output_dir}/comment_prompts_{run_number:03d}.json"
    metadata_path = f"{output_dir}/comment_prompts_metadata_{run_number:03d}.json"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_rows, f, indent=2, default=str)

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "subreddit": inferred_subreddit,
        "comment_ids": comment_ids or [],
        "total_comments_in_file": len(comments),
        "comments_exported": len(export_rows),
        "output_file": output_path,
        "metadata_file": metadata_path,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata
