"""
Comment analysis pipeline.

Reads a preprocessed comments file (comments_filtered_NNN.json), calls the LLM
for every comment, and writes results incrementally to a .jsonl progress file.
On completion the .jsonl is promoted to a final .json + _metadata.json pair.

Resume support
--------------
If a previous run was interrupted, a .jsonl file without a matching .json will
be found on start-up.  The user is asked whether to continue that run or start
a new one.
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv

from ..config import Config
from ..utils import setup_logger
from ..constants import (
    RAW_DATA_DIR,
    COMMENT_ANALYSIS_PROVIDER,
    COMMENT_ANALYSIS_MODEL,
    COMMENT_ANALYSIS_RATE_LIMIT_DELAY,
    COMMENT_ANALYSIS_MAX_RETRIES,
    COMMENT_ANALYSIS_RETRY_WAIT_SECONDS,
    COMMENT_ANALYSIS_OUTPUT_BASE,
    COMMENT_ANALYSIS_FILE_PREFIX,
)
from .llm_provider import LLMProvider, NonRetryableError
from .prompt_builder import build_comment_prompt

load_dotenv()
logger = setup_logger("analyze_comments")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _latest_filtered_comments(comments_dir: Path) -> Path:
    """Return the path to the latest comments_filtered_NNN.json."""
    candidates = sorted(
        p for p in comments_dir.glob("comments_filtered_*.json")
        if not p.name.endswith("_metadata.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No comments_filtered_*.json found in {comments_dir}")
    return candidates[-1]


def _latest_enriched_posts(scrape_path: Path) -> Optional[Path]:
    """Return the path to the latest posts_context_NNN.json, or None."""
    comments_dir = scrape_path / "comments"
    if not comments_dir.is_dir():
        return None
    candidates = sorted(
        p for p in comments_dir.glob("posts_context_*.json")
        if not p.name.endswith("_metadata.json")
    )
    return candidates[-1] if candidates else None


def _next_run_number(output_dir: Path) -> int:
    """Return the next auto-incremented run number for this output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        p for p in output_dir.glob(f"{COMMENT_ANALYSIS_FILE_PREFIX}_*.json")
        if not p.name.endswith("_metadata.json")
        and not p.suffix == ".jsonl"
    ]
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split(f"{COMMENT_ANALYSIS_FILE_PREFIX}_")[-1]))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def _find_incomplete_run(output_dir: Path) -> Optional[Tuple[int, Path, set]]:
    """
    Look for an incomplete run: a .jsonl progress file with no matching .json.

    Returns (run_number, jsonl_path, done_comment_ids) or None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for jsonl_path in sorted(output_dir.glob(f"{COMMENT_ANALYSIS_FILE_PREFIX}_*.jsonl")):
        stem = jsonl_path.stem  # e.g. "comment_analysis_001"
        final_json = output_dir / f"{stem}.json"
        if final_json.exists():
            continue  # already completed
        try:
            run_num = int(stem.split(f"{COMMENT_ANALYSIS_FILE_PREFIX}_")[-1])
        except ValueError:
            continue
        # Load already-done comment IDs
        done_ids: set = set()
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        cid = obj.get("comment_id")
                        if cid:
                            done_ids.add(cid)
        except Exception as e:
            logger.warning(f"Could not read progress file {jsonl_path}: {e}")
            continue
        return run_num, jsonl_path, done_ids
    return None


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def _call_with_retry(
    provider: LLMProvider,
    prompt: str,
    comment_id: str,
    max_retries: int,
    retry_wait: int,
) -> Dict[str, Any]:
    """Call provider.analyze_comment with exponential-ish retry on failure.

    NonRetryableError (e.g. billing / auth failures) aborts immediately.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            return provider.analyze_comment(prompt)
        except NonRetryableError as exc:
            # Billing / auth errors — no point retrying
            logger.error(f"[{comment_id}] non-retryable error, aborting run: {exc}")
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = retry_wait * attempt
                logger.warning(
                    f"[{comment_id}] attempt {attempt}/{max_retries} failed: {exc}  "
                    f"— retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"[{comment_id}] all {max_retries} attempts failed: {exc}"
                )
    return {"error": str(last_exc), "comment_id": comment_id}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_comment_analysis(
    scrape_dir: Optional[str] = None,
    config_path: str = "config/config.yaml",
    max_comments: Optional[int] = None,
    start_offset: int = 0,
) -> None:
    """
    Run LLM analysis on preprocessed comments.

    Parameters
    ----------
    scrape_dir:
        Path to the scrape folder (e.g. ``data/raw/scrape_006``).
        Auto-detects the latest scrape folder when omitted.
    config_path:
        Path to YAML config file.
    max_comments:
        If set, cap the total number of comments analysed this run.
        Useful for smoke-tests before a full run.
    start_offset:
        Skip the first N comments in the global ordered list.  Use this to
        resume from where a previous completed run left off, e.g. pass 320
        after a run that processed 320 comments.
    """
    cfg = Config(config_path)
    start_time = datetime.now()

    # --- resolve scrape dir ---
    if scrape_dir:
        scrape_path = Path(scrape_dir)
    else:
        raw = Path(cfg.get("raw_data_dir") or RAW_DATA_DIR)
        folders = sorted(p for p in raw.glob("scrape_*") if p.is_dir())
        if not folders:
            raise FileNotFoundError(f"No scrape_XXX folders found in {raw}")
        scrape_path = folders[-1]
    logger.info(f"Scrape dir: {scrape_path}")

    # --- locate latest filtered comments file ---
    comments_dir = scrape_path / "comments"
    filtered_file = _latest_filtered_comments(comments_dir)
    logger.info(f"Input: {filtered_file}")

    # --- load data ---
    raw_data = json.loads(filtered_file.read_text(encoding="utf-8"))
    posts: List[Dict[str, Any]] = raw_data if isinstance(raw_data, list) else raw_data.get("posts", [])

    # Build lookup maps once
    posts_map: Dict[str, Dict] = {p["post_id"]: p for p in posts if p.get("post_id")}
    all_comments_map: Dict[str, Dict] = {}
    all_ordered_comments: List[Dict] = []
    for post in posts:
        for c in post.get("comments", []):
            cid = c.get("comment_id")
            if cid:
                all_comments_map[cid] = c
                all_ordered_comments.append(c)

    # Overlay enriched post data (extracted_media etc.) if available
    enriched_posts_file = _latest_enriched_posts(scrape_path)
    if enriched_posts_file:
        try:
            enriched_raw = json.loads(enriched_posts_file.read_text(encoding="utf-8"))
            enriched_posts = (
                enriched_raw if isinstance(enriched_raw, list)
                else enriched_raw.get("posts", [])
            )
            n_overlaid = 0
            for ep in enriched_posts:
                pid = ep.get("post_id")
                if pid and pid in posts_map:
                    posts_map[pid].update(ep)
                    n_overlaid += 1
            logger.info(
                f"Enriched posts overlaid: {n_overlaid}/{len(enriched_posts)} "
                f"from {enriched_posts_file}"
            )
        except Exception as exc:
            logger.warning(f"Could not load enriched posts file {enriched_posts_file}: {exc}")
    else:
        logger.info("No posts_context file found — proceeding without media enrichment")

    total_comments = len(all_ordered_comments)
    logger.info(f"Total comments to analyse: {total_comments} across {len(posts)} posts")

    if start_offset > 0:
        all_ordered_comments = all_ordered_comments[start_offset:]
        logger.info(f"Skipping first {start_offset} comments (already processed in a prior run)")

    if max_comments is not None:
        all_ordered_comments = all_ordered_comments[:max_comments]
        logger.info(f"[DEV] Capped to {max_comments} comments for this run")

    total_comments = len(all_ordered_comments)

    # --- output dir: data/output/scrape_NNN/ ---
    output_dir = Path(COMMENT_ANALYSIS_OUTPUT_BASE) / scrape_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- resume detection ---
    done_ids: set = set()
    incomplete = _find_incomplete_run(output_dir)
    run_num: int

    if incomplete:
        run_num, jsonl_path, done_ids = incomplete
        pct = len(done_ids) / total_comments * 100 if total_comments else 0
        print(
            f"\nIncomplete run {run_num:03d} found: "
            f"{len(done_ids)}/{total_comments} comments done ({pct:.0f}%)."
        )
        answer = input("Continue this run? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            logger.info(f"Resuming run {run_num:03d} — skipping {len(done_ids)} completed comments")
        else:
            done_ids = set()
            run_num = _next_run_number(output_dir)
            jsonl_path = output_dir / f"{COMMENT_ANALYSIS_FILE_PREFIX}_{run_num:03d}.jsonl"
            logger.info(f"Starting fresh run {run_num:03d}")
    else:
        run_num = _next_run_number(output_dir)
        jsonl_path = output_dir / f"{COMMENT_ANALYSIS_FILE_PREFIX}_{run_num:03d}.jsonl"
        logger.info(f"Starting new run {run_num:03d}")

    final_json_path = output_dir / f"{COMMENT_ANALYSIS_FILE_PREFIX}_{run_num:03d}.json"
    metadata_path = output_dir / f"{COMMENT_ANALYSIS_FILE_PREFIX}_metadata_{run_num:03d}.json"

    # --- config values ---
    provider_name: str = cfg.get("comment_analysis.provider") or COMMENT_ANALYSIS_PROVIDER
    model_name: str = cfg.get("comment_analysis.model") or COMMENT_ANALYSIS_MODEL
    rate_delay: float = float(cfg.get("comment_analysis.rate_limit_delay") or COMMENT_ANALYSIS_RATE_LIMIT_DELAY)
    max_retries: int = int(cfg.get("comment_analysis.max_retries") or COMMENT_ANALYSIS_MAX_RETRIES)
    retry_wait: int = int(cfg.get("comment_analysis.retry_wait_seconds") or COMMENT_ANALYSIS_RETRY_WAIT_SECONDS)
    # include_rationale defaults True for backward compat; set False in config to save ~25-30% output tokens
    _raw_rationale = cfg.get("comment_analysis.include_rationale")
    include_rationale: bool = True if _raw_rationale is None else bool(_raw_rationale)

    # --- build provider ---
    provider = LLMProvider.from_config(provider_name, model=model_name, include_rationale=include_rationale)
    logger.info(f"include_rationale: {include_rationale}")
    logger.info(f"Provider: {provider_name} / {model_name}")

    # --- run ---
    remaining = [c for c in all_ordered_comments if c.get("comment_id") not in done_ids]
    n_remaining = len(remaining)
    n_done_start = len(done_ids)
    errors = 0

    logger.info(f"Comments to process this session: {n_remaining}")

    source_file = str(filtered_file)

    with open(jsonl_path, "a", encoding="utf-8") as progress_fh:
        for idx, comment in enumerate(remaining, start=1):
            comment_id = comment.get("comment_id", "UNKNOWN")
            post_id = comment.get("post_id", "")
            display_num = n_done_start + idx

            # Resolve parent
            parent_id = comment.get("parent_id", "")
            if parent_id == post_id:
                parent = posts_map.get(post_id)
                parent_is_post = True
            else:
                parent = all_comments_map.get(parent_id)
                parent_is_post = False

            # Build prompt
            try:
                prompt = build_comment_prompt(
                    comment=comment,
                    parent=parent,
                    parent_is_post=parent_is_post,
                    source_file=None,  # skip media preprocessing — not needed for scoring
                    posts_map=posts_map,
                    comments_map=all_comments_map,
                    dynamic_only=True,   # static definitions are cached in system block
                    include_rationale=include_rationale,
                )
            except Exception as exc:
                logger.error(f"[{comment_id}] prompt build failed: {exc}")
                result: Dict[str, Any] = {"comment_id": comment_id, "post_id": post_id, "error": f"prompt_build: {exc}"}
                errors += 1
                progress_fh.write(json.dumps(result, default=str) + "\n")
                progress_fh.flush()
                continue

            # LLM call
            logger.info(f"[{display_num}/{total_comments}] {comment_id} (post {post_id})")
            try:
                result = _call_with_retry(provider, prompt, comment_id, max_retries, retry_wait)
            except NonRetryableError as exc:
                # Persist progress and exit cleanly so the .jsonl is resumable
                progress_fh.flush()
                logger.error("Aborting run — add credits and resume to continue.")
                print(f"\n[ABORTED] {exc}")
                print(f"Progress saved to {jsonl_path} — resume when credits are topped up.")
                return

            # Ensure IDs are always present in the record
            result["comment_id"] = comment_id
            result.setdefault("post_id", post_id)

            if "error" in result:
                errors += 1

            # Write immediately — one JSON object per line
            progress_fh.write(json.dumps(result, default=str) + "\n")
            progress_fh.flush()

            done_ids.add(comment_id)

            if idx < n_remaining:
                time.sleep(rate_delay)

    # --- finalise ---
    logger.info("All comments processed. Writing final output files...")

    # Read back all results from .jsonl (includes any from previous resume sessions)
    all_results: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))

    final_json_path.write_text(
        json.dumps(all_results, indent=2, default=str),
        encoding="utf-8",
    )

    completed_at = datetime.now()
    metadata: Dict[str, Any] = {
        "run": run_num,
        "generated_at": start_time.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round((completed_at - start_time).total_seconds(), 1),
        "scrape_dir": str(scrape_path),
        "input_file": str(filtered_file),
        "output_file": str(final_json_path),
        "config_file": config_path,
        "provider": provider_name,
        "model": model_name,
        "counts": {
            "posts": len(posts),
            "comments_total": total_comments,
            "comments_analysed": len(all_results),
            "errors": errors,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    # Remove progress file now that final JSON is written
    jsonl_path.unlink(missing_ok=True)

    logger.info(
        f"Done. {len(all_results)} results written to {final_json_path}  "
        f"({errors} errors)"
    )
    print(f"\nRun {run_num:03d} complete: {len(all_results)} comments analysed, {errors} errors.")
    print(f"Output: {final_json_path}")
