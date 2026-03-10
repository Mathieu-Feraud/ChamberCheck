import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _pick_int(rng: np.random.Generator, lower: int = 0, upper: int = 10) -> int:
    return int(rng.integers(lower, upper + 1))


def _topic_from_post(post: Dict[str, Any]) -> Optional[str]:
    for key in ("topic", "topic_label"):
        value = post.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = post.get("metadata")
    if isinstance(metadata, dict):
        for key in ("topic", "topic_label"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    title = post.get("title")
    if isinstance(title, str) and title.strip():
        clean = " ".join(title.strip().split())
        if len(clean) > 120:
            clean = clean[:117] + "..."
        return f"Post Topic: {clean}"

    post_id = post.get("post_id")
    community = post.get("community")
    if isinstance(post_id, str) and post_id:
        if isinstance(community, str) and community:
            return f"Post Topic: {community}/{post_id}"
        return f"Post Topic: {post_id}"

    return None


def _raw_files(raw_dir: Path) -> List[Path]:
    files = []
    for path in sorted(raw_dir.glob("*.json")):
        name = path.name
        if name.endswith("_scraper_metadata.json"):
            continue
        if name == "subreddits_info.json":
            continue
        files.append(path)
    return files


def build_fake_entries_from_raw(
    raw_dir: Path,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    entries: List[Dict[str, Any]] = []
    source_files = _raw_files(raw_dir)
    prompt_number = 1

    for raw_file in source_files:
        payload = _load_json(raw_file)
        if not isinstance(payload, dict):
            continue

        posts = payload.get("posts", []) if isinstance(payload.get("posts"), list) else []
        post_topic_map: Dict[str, str] = {}
        for post in posts:
            if not isinstance(post, dict):
                continue
            post_id = post.get("post_id")
            if not isinstance(post_id, str) or not post_id:
                continue
            topic = _topic_from_post(post)
            if topic:
                post_topic_map[post_id] = topic

        comments = payload.get("comments", []) if isinstance(payload.get("comments"), list) else []
        for comment in comments:
            if not isinstance(comment, dict):
                continue

            comment_id = comment.get("comment_id")
            post_id = comment.get("post_id")
            if not isinstance(comment_id, str) or not comment_id:
                continue
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_topic_map.get(post_id)
            if not topic_label:
                topic_label = f"Post Topic: {post_id}"

            stance_value = int(rng.integers(-10, 11))
            if stance_value == 0:
                stance_value = int(rng.choice([-1, 1]))

            claim_strength = _pick_int(rng)
            evidence_quality = _pick_int(rng)
            reasoning_depth = _pick_int(rng)
            toxicity = _pick_int(rng)
            discrediting = _pick_int(rng)
            defensive = _pick_int(rng)
            civility = max(0, 10 - toxicity)
            anger = _pick_int(rng)
            anxiety = _pick_int(rng)
            disgust = _pick_int(rng)

            if int(rng.integers(0, 11)) == 0:
                claim_strength_value: Any = "N/A"
                evidence_quality_value: Any = "N/A"
                reasoning_depth_value: Any = "N/A"
            else:
                claim_strength_value = claim_strength
                evidence_quality_value = evidence_quality
                reasoning_depth_value = reasoning_depth

            entry = {
                "prompt_number": prompt_number,
                "comment_id": comment_id,
                "parse_mode": "strict",
                "entry": {
                    "parent_topic": topic_label,
                    "comment_type": ["opinion"],
                    "topic": {
                        "label": topic_label,
                        "stance": {
                            "value": stance_value,
                            "rationale": "Synthetic stance for metric testing.",
                        },
                    },
                    "epistemic_risk": {
                        "claim_strength": claim_strength_value,
                        "evidence_quality": evidence_quality_value,
                        "reasoning_depth": reasoning_depth_value,
                        "rationale": "Synthetic epistemic profile for metric testing.",
                    },
                    "toxicity": toxicity,
                    "discrediting": discrediting,
                    "defensive": defensive,
                    "civility": civility,
                    "emotion": {
                        "anger": anger,
                        "anxiety": anxiety,
                        "disgust": disgust,
                    },
                },
            }

            entries.append(entry)
            prompt_number += 1

    return entries, source_files


def build_synthetic_run_outputs(
    raw_dir: Path,
    output_entries: Path,
    output_metadata: Path,
    output_prompts_metadata: Path,
    random_seed: int | None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    entries, source_files = build_fake_entries_from_raw(raw_dir, rng)

    _save_json(output_entries, entries)

    source_file_str = ", ".join(str(path).replace("/", "\\") for path in source_files)
    selected_comment_ids = []
    for item in entries:
        prompt_number = item.get("prompt_number")
        comment_id = item.get("comment_id")
        if not isinstance(comment_id, str):
            continue
        selected_comment_ids.append([prompt_number, comment_id, "synthetic"])

    prompts_metadata_payload = {
        "run_timestamp": datetime.now().isoformat(),
        "source_file": source_file_str,
        "random_seed": random_seed,
        "selected_comment_ids": selected_comment_ids,
        "notes": "Synthetic run metadata for full scrape_003 fake entries.",
    }
    _save_json(output_prompts_metadata, prompts_metadata_payload)

    run_metadata_payload = {
        "run_timestamp": datetime.now().isoformat(),
        "function": "run_make_fake_llm_entries_003",
        "prompts_txt_path": None,
        "metadata_json_path": str(output_prompts_metadata).replace("/", "\\"),
        "top": len(entries),
        "provider": "synthetic",
        "model": "synthetic-fake-entries-001",
        "model_requested": "synthetic-fake-entries-001",
        "model_used": "synthetic-fake-entries-001",
        "total_prompts_in_file": len(entries),
        "prompts_considered": len(entries),
        "entries_extracted": len(entries),
        "entries_parse_errors": 0,
        "token_usage_per_prompt": [],
        "validation_warnings": [],
        "parse_errors": [],
        "output_file": str(output_entries).replace("/", "\\"),
        "metadata_file": str(output_metadata).replace("/", "\\"),
        "notes": "Synthetic full-data run based on raw scrape_003 comments.",
    }
    _save_json(output_metadata, run_metadata_payload)

    return {
        "output_entries_file": str(output_entries),
        "output_metadata_file": str(output_metadata),
        "output_prompts_metadata_file": str(output_prompts_metadata),
        "entries_count": len(entries),
        "raw_files_count": len(source_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic full scrape_003 fake LLM entries with one topic per parent post."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/scrape_003",
        help="Directory containing raw scrape_003 JSON files.",
    )
    parser.add_argument(
        "--output-entries",
        default="data/output/scrape_003/fake_llm_entries_001.json",
        help="Output synthetic entries JSON path.",
    )
    parser.add_argument(
        "--output-metadata",
        default="data/output/scrape_003/fake_llm_entries_metadata_001.json",
        help="Output synthetic run metadata JSON path.",
    )
    parser.add_argument(
        "--output-prompts-metadata",
        default="data/output/scrape_003/fake_prompts_metadata_001.json",
        help="Output prompts metadata JSON path with source_file mapping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed. Omit for true IID redraws on each run.",
    )

    args = parser.parse_args()

    result = build_synthetic_run_outputs(
        raw_dir=Path(args.raw_dir),
        output_entries=Path(args.output_entries),
        output_metadata=Path(args.output_metadata),
        output_prompts_metadata=Path(args.output_prompts_metadata),
        random_seed=args.seed,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
