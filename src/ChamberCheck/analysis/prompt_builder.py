"""
Prompt builder for comment analysis.

Centralizes prompt construction for analysis, export, and A/B/n testing.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..constants import (
    COMMENT_ANALYSIS_PROMPT,
    COMMENT_ANALYSIS_DYNAMIC_TEMPLATE,
    COMMENT_ANALYSIS_DYNAMIC_TEMPLATE_NO_RATIONALE,
)


def _load_post_from_file(source_file: str, post_id: str) -> Optional[Dict[str, Any]]:
    if not source_file or not post_id:
        return None

    path = Path(source_file)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    posts = data.get("posts", []) if isinstance(data, dict) else []
    for post in posts:
        if post.get("post_id") == post_id:
            return post

    return None


def _ensure_post_preprocessed(
    post: Optional[Dict[str, Any]],
    post_id: str,
    source_file: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not source_file or not post_id:
        return post

    if isinstance(post, dict) and "extracted_media" in post:
        return post

    # Reload from file — media may have been enriched by the explicit preprocessing step
    return _load_post_from_file(source_file, post_id) or post


def _load_subreddits_info_map(source_file: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not source_file:
        return {}

    info_path = Path(source_file).parent / "subreddits_info.json"
    if not info_path.exists():
        return {}

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    subreddits = data.get("subreddits", []) if isinstance(data, dict) else []
    info_map = {}
    for entry in subreddits:
        if not isinstance(entry, dict):
            continue
        name = entry.get("subreddit")
        if isinstance(name, str) and name:
            info_map[name.lower()] = entry

    return info_map


def _prepare_parent_context(
    parent: Dict[str, Any],
    is_post: bool = False,
    source_file: Optional[str] = None,
    fallback_post_id: Optional[str] = None,
) -> Dict[str, str]:
    def _build_group_context_from_subreddit(subreddit_name: Optional[str]) -> str:
        if not subreddit_name:
            return ""

        info_map = _load_subreddits_info_map(source_file)
        info = info_map.get(str(subreddit_name).lower())

        subreddit_parts = []
        display_name = info.get("subreddit") if isinstance(info, dict) else subreddit_name
        subreddit_parts.append(f"Subreddit: r/{display_name}")

        if isinstance(info, dict) and info.get("title"):
            subreddit_parts.append(f"Subreddit title: {info.get('title')}")

        public_description = info.get("public_description") if isinstance(info, dict) else None
        if public_description:
            subreddit_parts.append(f"Subreddit description: {public_description}")

        return "\n".join(subreddit_parts).strip()

    if is_post:
        post_id = parent.get("post_id") if isinstance(parent, dict) else fallback_post_id
        if not parent and fallback_post_id and source_file:
            parent = _load_post_from_file(source_file, fallback_post_id) or {}
        if source_file and post_id:
            parent = _ensure_post_preprocessed(parent, post_id, source_file) or parent

        title = parent.get("title", "") if isinstance(parent, dict) else ""
        content = parent.get("content", "") if isinstance(parent, dict) else ""
        media_parts = []
        subreddit_name = None
        if isinstance(parent, dict):
            subreddit_name = parent.get("community") or parent.get("subreddit")
            if not subreddit_name:
                metadata = parent.get("metadata")
                if isinstance(metadata, dict):
                    subreddit_name = metadata.get("subreddit")

        topic_label = None
        media = parent.get("extracted_media") if isinstance(parent, dict) else None
        if isinstance(media, dict):
            analysis = media.get("analysis") if isinstance(media, dict) else None
            if isinstance(analysis, dict) and analysis.get("topic"):
                topic_label = analysis.get("topic")
            elif media.get("topic"):
                topic_label = media.get("topic")

            media_type = media.get("media_type")
            if isinstance(analysis, dict) and analysis.get("media_type"):
                media_type = analysis.get("media_type")
            if media_type:
                media_parts.append(f"Media type: {media_type}")

            description = media.get("description")
            if isinstance(analysis, dict) and analysis.get("description"):
                description = analysis.get("description")
            if description:
                media_parts.append(f"Description: {description}")

            extracted_text = media.get("extracted_text")
            if isinstance(analysis, dict) and analysis.get("text_content"):
                extracted_text = analysis.get("text_content")
            if extracted_text:
                media_parts.append(f"Content: {extracted_text}")

            text_context = media.get("text_context")
            if isinstance(analysis, dict) and analysis.get("text_context"):
                text_context = analysis.get("text_context")
            if text_context:
                media_parts.append(f"Context: {text_context}")

            key_points = media.get("key_points")
            if key_points:
                if isinstance(key_points, list):
                    points_str = "; ".join(key_points[:5])
                else:
                    points_str = str(key_points)
                media_parts.append(f"Key points: {points_str}")

        parent_lines = [f"Title: {title}"]
        if content:
            parent_lines.append(f"Content: {content}")
        if media_parts:
            parent_lines.extend(media_parts)

        parent_text = "\n".join(parent_lines)
        group_context = _build_group_context_from_subreddit(subreddit_name)

        if not topic_label:
            topic_label = ""

        return {
            "context_type": "post",
            "topic": topic_label,
            "text": parent_text.strip(),
            "group_context": group_context.strip(),
        }

    parent_text = parent.get("content", "") if isinstance(parent, dict) else ""
    topic_label = None
    subreddit_name = None
    if isinstance(parent, dict) and parent.get("topic"):
        analysis_topic = parent.get("topic")
        if isinstance(analysis_topic, dict):
            topic_label = analysis_topic.get("label")
        else:
            topic_label = str(analysis_topic)

    if isinstance(parent, dict):
        subreddit_name = parent.get("community") or parent.get("subreddit")
        if not subreddit_name:
            metadata = parent.get("metadata")
            if isinstance(metadata, dict):
                subreddit_name = metadata.get("subreddit")

    if not subreddit_name and fallback_post_id and source_file:
        parent_post = _load_post_from_file(source_file, fallback_post_id)
        if isinstance(parent_post, dict):
            subreddit_name = parent_post.get("community") or parent_post.get("subreddit")
            if not subreddit_name:
                metadata = parent_post.get("metadata")
                if isinstance(metadata, dict):
                    subreddit_name = metadata.get("subreddit")

    if not topic_label:
        topic_label = ""

    return {
        "context_type": "comment",
        "topic": topic_label,
        "text": parent_text,
        "group_context": _build_group_context_from_subreddit(subreddit_name),
    }


def _build_ancestor_chain(
    comment: Dict[str, Any],
    comments_map: Optional[Dict[str, Dict[str, Any]]] = None,
    max_depth: int = 100,
) -> List[Dict[str, Any]]:
    if not comments_map:
        return []

    ancestors: List[Dict[str, Any]] = []
    seen_ids = set()
    parent_id = comment.get("parent_id") if isinstance(comment, dict) else None
    depth = 0

    while parent_id and depth < max_depth:
        if parent_id in seen_ids:
            break
        seen_ids.add(parent_id)

        parent_comment = comments_map.get(parent_id)
        if not isinstance(parent_comment, dict):
            break

        ancestors.append(parent_comment)
        parent_id = parent_comment.get("parent_id")
        depth += 1

    ancestors.reverse()
    return ancestors


def _build_full_context_text(
    post_text: str,
    ancestor_comments: List[Dict[str, Any]],
) -> str:
    context_sections = []

    if post_text:
        context_sections.append(f"ORIGINAL POST:\n{post_text.strip()}")

    for idx, ancestor in enumerate(ancestor_comments, start=1):
        ancestor_text = ancestor.get("content", "") if isinstance(ancestor, dict) else ""
        context_sections.append(f"COMMENT {idx}:\n{ancestor_text}")

    return "\n\n".join(context_sections).strip()


def build_comment_prompt(
    comment: Dict[str, Any],
    parent: Optional[Dict[str, Any]] = None,
    parent_is_post: bool = False,
    source_file: Optional[str] = None,
    posts_map: Optional[Dict[str, Dict[str, Any]]] = None,
    comments_map: Optional[Dict[str, Dict[str, Any]]] = None,
    simplify_output_template: bool = False,
    dynamic_only: bool = False,
    include_rationale: bool = True,
) -> str:
    comment_text = comment.get("content", "")

    if parent_is_post:
        post_id = None
        if isinstance(parent, dict):
            post_id = parent.get("post_id")
        if not post_id:
            post_id = comment.get("post_id")

        if not parent and posts_map and post_id:
            parent = posts_map.get(post_id)

        parent_context = _prepare_parent_context(
            parent or {},
            is_post=True,
            source_file=source_file,
            fallback_post_id=post_id,
        )
    elif parent:
        parent_context = _prepare_parent_context(
            parent,
            is_post=False,
            source_file=source_file,
            fallback_post_id=comment.get("post_id"),
        )
    else:
        parent_context = {
            "context_type": "post",
            "topic": "General discussion",
            "text": "[No parent context available]",
        }

    post_id = comment.get("post_id")
    post_for_context = None
    if posts_map and post_id:
        post_for_context = posts_map.get(post_id)

    post_context = _prepare_parent_context(
        post_for_context or {},
        is_post=True,
        source_file=source_file,
        fallback_post_id=post_id,
    )

    ancestor_chain = _build_ancestor_chain(comment, comments_map=comments_map)
    if not ancestor_chain and parent and not parent_is_post:
        ancestor_chain = [parent]

    full_context_text = _build_full_context_text(
        post_text=post_context.get("text", ""),
        ancestor_comments=ancestor_chain,
    )
    if not full_context_text:
        full_context_text = parent_context.get("text", "[No parent context available]")

    # dynamic_only=True → use only the per-call section (GROUP CONTEXT onward).
    # The static definitions are cached in the Anthropic system block, so the
    # user message only needs to carry the dynamic context + output template.
    # dynamic_only=False (default) → full self-contained prompt (for export/tests).
    if dynamic_only:
        base_template = (
            COMMENT_ANALYSIS_DYNAMIC_TEMPLATE
            if include_rationale
            else COMMENT_ANALYSIS_DYNAMIC_TEMPLATE_NO_RATIONALE
        )
    else:
        from ..constants import COMMENT_ANALYSIS_PROMPT_NO_RATIONALE
        base_template = COMMENT_ANALYSIS_PROMPT if include_rationale else COMMENT_ANALYSIS_PROMPT_NO_RATIONALE

    prompt = base_template.replace(
        "{{PARENT_TOPIC}}", parent_context["topic"]
    ).replace(
        "{{PARENT_CONTEXT}}", parent_context["context_type"]
    ).replace(
        "{{GROUP_CONTEXT}}", post_context.get("group_context", parent_context.get("group_context", ""))
    ).replace(
        "{{PARENT_TEXT}}", full_context_text
    ).replace(
        "{{TEXT}}", comment_text
    )

    if simplify_output_template:
        prompt = _simplify_output_template(prompt)

    return prompt


def _simplify_output_template(prompt: str) -> str:
    output_index = prompt.find("Output:")
    if output_index == -1:
        return prompt

    header = prompt[:output_index]
    output_section = prompt[output_index:]

    output_section = output_section.replace(
        "\"string - the parent topic label provided above\"",
        "\"\""
    )
    output_section = output_section.replace("\"number | N/A\"", "")
    output_section = output_section.replace("\"string | N/A\"", "\"\"")
    output_section = output_section.replace("\"string\"", "\"\"")

    return header + output_section
