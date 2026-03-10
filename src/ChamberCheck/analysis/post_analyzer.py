"""
LLM-based post title analysis.

Analyzes post titles using the configured LLM provider (Anthropic or OpenAI)
to extract topic classification and discussion potential scores.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Union

from ..constants import (
    POST_ANALYSIS_RETRY_LIMIT,
    POST_ANALYSIS_RETRY_DELAY,
    POST_ANALYSIS_USER_TEMPLATE,
    PROMPT_POST_TITLE_ANALYSIS,
)
from ..config import Config


class CreditBalanceError(RuntimeError):
    """Raised when the API rejects requests due to insufficient credits."""


def _is_credit_error(exc: Exception) -> bool:
    return "credit balance" in str(exc).lower()


def _resolve_scrape_dir(scrape_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve scrape directory, defaulting to latest data/raw/scrape_* folder."""
    if scrape_dir is not None:
        target = Path(scrape_dir)
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Scrape directory not found: {target}")
        return target

    raw_dir = Path("data/raw")
    candidates = sorted([path for path in raw_dir.glob("scrape_*") if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No scrape_* folders found in {raw_dir}")
    return candidates[-1]


def _call_openai(client, title: str, model: str, temperature: float, max_tokens: int) -> dict:
    """Call the OpenAI API with retry logic. Returns a result dict."""
    for attempt in range(1, POST_ANALYSIS_RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PROMPT_POST_TITLE_ANALYSIS},
                    {"role": "user",   "content": POST_ANALYSIS_USER_TEMPLATE.format(title=title)},
                ],
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            result["_model"]  = response.model
            result["_tokens"] = {
                "prompt":     response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
            }
            return result
        except Exception as exc:
            last_exc = exc
            print(f"    attempt {attempt}/{POST_ANALYSIS_RETRY_LIMIT} failed: {exc}")
            if _is_credit_error(exc):
                raise CreditBalanceError(str(exc)) from exc
            if attempt < POST_ANALYSIS_RETRY_LIMIT:
                time.sleep(POST_ANALYSIS_RETRY_DELAY)
    return {"error": "max retries exceeded", "error_detail": str(last_exc), "topic": None, "confidence": None}


def _call_anthropic(client, title: str, model: str, max_tokens: int) -> dict:
    """Call the Anthropic API with retry logic. Returns a result dict."""
    last_exc: Exception = Exception("unknown")
    for attempt in range(1, POST_ANALYSIS_RETRY_LIMIT + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=PROMPT_POST_TITLE_ANALYSIS,
                messages=[
                    {"role": "user", "content": POST_ANALYSIS_USER_TEMPLATE.format(title=title)},
                ],
            )
            raw = response.content[0].text.strip()
            result = json.loads(raw)
            result["_model"]  = response.model
            result["_tokens"] = {
                "prompt":     response.usage.input_tokens,
                "completion": response.usage.output_tokens,
            }
            return result
        except Exception as exc:
            last_exc = exc
            print(f"    attempt {attempt}/{POST_ANALYSIS_RETRY_LIMIT} failed: {exc}")
            if _is_credit_error(exc):
                raise CreditBalanceError(str(exc)) from exc
            if attempt < POST_ANALYSIS_RETRY_LIMIT:
                time.sleep(POST_ANALYSIS_RETRY_DELAY)
    return {"error": "max retries exceeded", "error_detail": str(last_exc), "topic": None, "confidence": None}


def analyze_posts(
    scrape_dir: Optional[Union[str, Path]] = None,
    config_path: str = "config/config.yaml",
) -> str:
    """Analyze post titles using the configured LLM provider (anthropic or openai).

    Auto-increments the output run number so previous runs are never overwritten.
    Results are written to disk after every post so a mid-run failure loses nothing.

    Args:
        scrape_dir:  Path to the scrape folder (e.g. ``"data/raw/scrape_004"``).
                     If omitted, latest ``data/raw/scrape_*`` is used.
        config_path: Path to the unified YAML config file.

    Returns:
        Path to the written ``analysis_NNN.json`` file.
    """
    cfg         = Config(config_path)
    provider    = cfg.get("post_analysis.provider", "anthropic").lower()
    model       = cfg.get("post_analysis.model", "claude-3-5-haiku-20241022")
    temperature = cfg.get("post_analysis.temperature", 0.1)
    max_tokens  = cfg.get("post_analysis.max_tokens", 500)

    if provider == "anthropic":
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set – add it to your .env file")
        client = Anthropic(api_key=api_key)
    else:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set – add it to your .env file")
        client = OpenAI(api_key=api_key)

    scrape_dir = _resolve_scrape_dir(scrape_dir)
    print(f"Using scrape directory: {scrape_dir}")
    posts_file = scrape_dir / "posts.json"
    out_dir    = scrape_dir / "posts_analysis"
    out_dir.mkdir(exist_ok=True)

    # Resume incomplete run if latest analysis file has no companion metadata file;
    # otherwise auto-increment to a new run. Excludes *_metadata.json from glob.
    existing_analyses = sorted([f for f in out_dir.glob("analysis_*.json") if "_metadata" not in f.name])
    if existing_analyses:
        last      = existing_analyses[-1]
        num       = int(last.stem.split("_")[-1])
        last_meta = out_dir / f"{last.stem}_metadata.json"
        if not last_meta.exists():
            out_path  = last
            meta_path = last_meta
            next_num  = num
            all_saved = json.loads(last.read_text(encoding="utf-8"))
            # Retry entries that failed due to credit balance; keep all other results (success or different error).
            # If error_detail is absent (legacy entry), assume credit-balance and retry.
            def _is_saved_credit_error(r: dict) -> bool:
                if "error" not in r:
                    return False
                detail = r.get("error_detail", "")
                return not detail or "credit balance" in detail.lower()
            results   = [r for r in all_saved if not _is_saved_credit_error(r)]
            done_ids  = {r.get("_post_id") for r in results}
            retrying  = len(all_saved) - len(results)
            print(f"Resuming run {next_num:03d} — {len(results)} posts already done, skipping them"
                  + (f" ({retrying} credit-balance failures will be retried)" if retrying else ""))
        else:
            next_num  = len(existing_analyses) + 1
            out_path  = out_dir / f"analysis_{next_num:03d}.json"
            meta_path = out_dir / f"analysis_{next_num:03d}_metadata.json"
            results   = []
            done_ids  = set()
    else:
        next_num  = 1
        out_path  = out_dir / f"analysis_{next_num:03d}.json"
        meta_path = out_dir / f"analysis_{next_num:03d}_metadata.json"
        results   = []
        done_ids  = set()

    posts  = json.loads(posts_file.read_text(encoding="utf-8"))["posts"]
    print(f"Using provider: {provider} / model: {model}")
    print(f"Loaded {len(posts)} posts from {posts_file}")

    # Seed counters from already-completed results when resuming
    total_tokens = sum(
        r.get("_tokens", {}).get("prompt", 0) + r.get("_tokens", {}).get("completion", 0)
        for r in results
    )
    errors    = sum(1 for r in results if "error" in r)
    run_start = time.time()

    for idx, post in enumerate(posts, start=1):
        if post.get("post_id") in done_ids:
            continue
        title = post.get("title", "")
        print(f"  [{idx:03d}/{len(posts)}] {title[:70]}")

        try:
            result = (
                _call_anthropic(client, title, model, max_tokens)
                if provider == "anthropic"
                else _call_openai(client, title, model, temperature, max_tokens)
            )
        except CreditBalanceError:
            print("\n⚠  Credit balance error — aborting run. Re-run once credits are available.")
            tmp_path = out_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(out_path)
            return str(out_path)
        result["_post_id"]   = post.get("post_id")
        result["_community"] = post.get("community")
        result["_title"]     = title

        tokens = result.get("_tokens", {})
        total_tokens += tokens.get("prompt", 0) + tokens.get("completion", 0)
        if "error" in result:
            errors += 1
        results.append(result)
        # Write to a temp file then rename — avoids Windows file-lock errors
        # when the output file is open in an editor.
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(out_path)

    run_end = time.time()

    metadata = {
        "run":              next_num,
        "generated_at":     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start)),
        "completed_at":     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_end)),
        "duration_seconds": round(run_end - run_start, 1),
        "source_file":      str(posts_file),
        "output_file":      str(out_path),
        "provider":         provider,
        "model":            model,
        "temperature":      temperature,
        "max_tokens":       max_tokens,
        "posts_total":      len(posts),
        "posts_analysed":   len(results),
        "errors":           errors,
        "tokens": {
            "total":        total_tokens,
            "avg_per_post": round(total_tokens / len(results), 1) if results else 0,
        },
        "subreddits": sorted({p.get("community") for p in posts if p.get("community")}),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Results  → {out_path}")
    print(f"  Metadata → {meta_path}")
    print(f"  Tokens   : {total_tokens:,}  |  errors: {errors}  |  duration: {metadata['duration_seconds']}s")
    return str(out_path)
