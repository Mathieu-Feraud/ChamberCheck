"""
A/B/n Test Builder for comment analysis comparison.

Builds representative comment chains for A/B/n testing different LLM models/configurations.
Selects comments randomly, follows reply chains, and generates prompts for comparison.
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...analysis.prompt_builder import build_comment_prompt
from ...analysis.llm_provider import LLMProvider
from ...analysis.batch_analyzer import load_comments_and_posts, build_parent_map
from ...constants import (
    OUTPUT_DIR,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
)
from ...config import Config


def _get_next_number_for_pattern(
    directory: Path,
    filename_prefix: str,
    extension: str = "json",
) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    run_numbers = []
    for file in directory.glob(f"{filename_prefix}_*.{extension}"):
        stem = file.stem
        if not stem.startswith(f"{filename_prefix}_"):
            continue
        suffix = stem.split(f"{filename_prefix}_")[-1]
        try:
            run_numbers.append(int(suffix))
        except ValueError:
            continue
    return max(run_numbers, default=0) + 1


def _get_abn_test_output_dir(base_output_dir: str) -> Path:
    return Path(base_output_dir) / "abn_test"


def _split_prompt_sections(prompts_text: str) -> List[tuple[int, str]]:
    matches = list(re.finditer(r"(?m)^Prompt\s*#\s*(\d+)\s*$", prompts_text))
    sections: List[tuple[int, str]] = []

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(prompts_text)
        prompt_number = int(match.group(1))
        body = prompts_text[start:end].strip()
        sections.append((prompt_number, body))

    return sections


def _extract_balanced_json_object(text: str, start_index: int) -> Optional[str]:
    depth = 0
    in_string = False
    escape = False

    for idx in range(start_index, len(text)):
        char = text[idx]

        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index:idx + 1]

    return None


def _extract_balanced_json_object_loose(text: str, start_index: int) -> Optional[str]:
    """Fallback JSON boundary extractor that ignores string state.

    Useful when malformed quotes exist inside manually edited JSON.
    """
    depth = 0

    for idx in range(start_index, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index:idx + 1]

    return None


def _extract_prompt_output_json(prompt_body: str) -> Optional[str]:
    output_idx = prompt_body.find("Output:")
    if output_idx == -1:
        return None

    section = prompt_body[output_idx + len("Output:"):]
    json_start = section.find("{")
    if json_start == -1:
        return None

    json_text = _extract_balanced_json_object(section, json_start)
    if json_text:
        return json_text

    # Fallback for malformed quote cases in user-edited outputs.
    return _extract_balanced_json_object_loose(section, json_start)


def _repair_comment_type_array_quotes(json_text: str) -> str:
    """Repair missing quotes in comment_type array items.

    Example: ["meme/joke", low effort] -> ["meme/joke", "low effort"]
    """
    pattern = re.compile(r'("comment_type"\s*:\s*\[)([^\]]*)(\])', re.DOTALL)

    def _repair_match(match: re.Match) -> str:
        prefix = match.group(1)
        body = match.group(2)
        suffix = match.group(3)

        raw_items = [item.strip() for item in body.split(",") if item.strip()]
        repaired_items = []

        for item in raw_items:
            token = item.strip()
            if token.startswith('"') and token.endswith('"') and len(token) >= 2:
                repaired_items.append(token)
                continue

            if token.startswith('"') and not token.endswith('"'):
                repaired_items.append(f'{token}"')
                continue

            if token.endswith('"') and not token.startswith('"'):
                repaired_items.append(f'"{token}')
                continue

            escaped = token.replace('"', '\\"')
            repaired_items.append(f'"{escaped}"')

        return f"{prefix}{', '.join(repaired_items)}{suffix}"

    return pattern.sub(_repair_match, json_text)


def _repair_json_string(json_text: str) -> str:
    repaired = json_text

    # Remove accidental double commas and trailing commas.
    repaired = re.sub(r",\s*,", ",", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    repaired = _repair_comment_type_array_quotes(repaired)

    # Quote likely unquoted scalar values (e.g., label: Christianity, rationale: good).
    # Keeps numbers, booleans, null, arrays/objects untouched.
    value_pattern = re.compile(r'(:\s*)([^\s\{\["\-\d][^,\}\n\r]*)')

    def _quote_if_needed(match: re.Match) -> str:
        prefix = match.group(1)
        raw_value = match.group(2).strip()
        lowered = raw_value.lower()

        if lowered in {"true", "false", "null"}:
            return f"{prefix}{raw_value}"

        if re.fullmatch(r"-?\d+(\.\d+)?", raw_value):
            return f"{prefix}{raw_value}"

        escaped = raw_value.replace('"', '\\"')
        return f'{prefix}"{escaped}"'

    repaired = value_pattern.sub(_quote_if_needed, repaired)
    return repaired


def _normalize_numeric_or_na(value: Any) -> tuple[Any, bool]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value, True

    if isinstance(value, str):
        text = value.strip()
        if text.upper() == "N/A":
            return "N/A", True
        if re.fullmatch(r"-?\d+", text):
            return int(text), True
        if re.fullmatch(r"-?\d+\.\d+", text):
            return float(text), True

    return value, False


def _validate_and_normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    numeric_paths = [
        ("topic", "stance", "value"),
        ("epistemic_risk", "claim_strength"),
        ("epistemic_risk", "evidence_quality"),
        ("epistemic_risk", "reasoning_depth"),
        ("toxicity",),
        ("discrediting",),
        ("defensive",),
        ("civility",),
        ("emotion", "anger"),
        ("emotion", "anxiety"),
        ("emotion", "disgust"),
    ]

    normalized = entry
    issues: List[str] = []

    for path in numeric_paths:
        cursor: Any = normalized
        for key in path[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]

        if not isinstance(cursor, dict):
            continue

        leaf = path[-1]
        if leaf not in cursor:
            continue

        converted, is_valid = _normalize_numeric_or_na(cursor[leaf])
        cursor[leaf] = converted
        if not is_valid:
            issues.append(f"{'.'.join(path)} must be numeric or N/A")

    return {
        "entry": normalized,
        "issues": issues,
    }


def extract_abn_user_entries(
    prompts_txt_path: str,
    top: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract manually-completed JSON entries from an ABN prompts text file.

    Args:
        prompts_txt_path: Path to abn_test_prompts_XXX.txt
        top: Optional limit for number of prompt outputs to extract in appearance order
        output_dir: Base output directory (e.g. data/output/scrape_003)

    Returns:
        Dictionary with extraction summary and output file paths
    """
    prompts_path = Path(prompts_txt_path)
    if not prompts_path.exists():
        raise ValueError(f"Prompts file not found: {prompts_txt_path}")

    prompts_text = prompts_path.read_text(encoding="utf-8")
    all_sections = _split_prompt_sections(prompts_text)
    prompt_sections = all_sections[:top] if top is not None else all_sections

    extracted_entries = []
    parse_errors = []
    validation_warnings = []

    for prompt_number, prompt_body in prompt_sections:
        json_text = _extract_prompt_output_json(prompt_body)
        if not json_text:
            parse_errors.append({
                "prompt_number": prompt_number,
                "error": "No JSON object found after Output:",
            })
            continue

        parsed = None
        parse_mode = "strict"
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            repaired_text = _repair_json_string(json_text)
            try:
                parsed = json.loads(repaired_text)
                parse_mode = "repaired"
            except json.JSONDecodeError as exc:
                parse_errors.append({
                    "prompt_number": prompt_number,
                    "error": str(exc),
                })
                continue

        if not isinstance(parsed, dict):
            parse_errors.append({
                "prompt_number": prompt_number,
                "error": "Parsed output is not a JSON object",
            })
            continue

        normalized = _validate_and_normalize_entry(parsed)
        if normalized["issues"]:
            validation_warnings.append({
                "prompt_number": prompt_number,
                "issues": normalized["issues"],
            })

        extracted_entries.append({
            "prompt_number": prompt_number,
            "parse_mode": parse_mode,
            "entry": normalized["entry"],
        })

    if output_dir:
        base_output_path = Path(output_dir)
    else:
        if prompts_path.parent.name == "abn_test":
            base_output_path = prompts_path.parent.parent
        else:
            base_output_path = Path(OUTPUT_DIR)

    abn_test_dir = _get_abn_test_output_dir(str(base_output_path))
    run_number = _get_next_number_for_pattern(abn_test_dir, "abn_test_user_entries", extension="json")

    entries_path = abn_test_dir / f"abn_test_user_entries_{run_number:03d}.json"
    metadata_path = abn_test_dir / f"abn_test_user_entries_metadata_{run_number:03d}.json"

    with open(entries_path, "w", encoding="utf-8") as f:
        json.dump(extracted_entries, f, indent=2)

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "function": "extract_abn_user_entries",
        "prompts_txt_path": str(prompts_path),
        "top": top,
        "total_prompts_in_file": len(all_sections),
        "prompts_considered": len(prompt_sections),
        "entries_extracted": len(extracted_entries),
        "entries_parse_errors": len(parse_errors),
        "validation_warnings": validation_warnings,
        "parse_errors": parse_errors,
        "output_file": str(entries_path),
        "metadata_file": str(metadata_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "success",
        "output_file": str(entries_path),
        "metadata_file": str(metadata_path),
        "entries_extracted": len(extracted_entries),
        "entries_parse_errors": len(parse_errors),
        "validation_warnings": len(validation_warnings),
    }


def _strip_existing_output(prompt_body: str) -> str:
    output_idx = prompt_body.find("Output:")
    if output_idx == -1:
        return prompt_body.strip()
    return (prompt_body[:output_idx] + "Output:\n").strip()


def _parse_source_files_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    source_field = metadata.get("source_file", "")
    if isinstance(source_field, list):
        return [str(path).strip() for path in source_field if str(path).strip()]

    if isinstance(source_field, str):
        return [segment.strip() for segment in source_field.split(",") if segment.strip()]

    return []


def _build_llm_prompts_from_metadata(
    metadata_json_path: str,
    top: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata_path = Path(metadata_json_path)
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found: {metadata_json_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    selected_comment_ids = metadata.get("selected_comment_ids", [])
    if not isinstance(selected_comment_ids, list) or not selected_comment_ids:
        raise ValueError("selected_comment_ids missing or empty in metadata")

    ordered_rows = []
    for row in selected_comment_ids:
        if not isinstance(row, list) or len(row) < 2:
            continue
        ordered_rows.append(row)

    ordered_rows = sorted(ordered_rows, key=lambda item: item[0])
    if top is not None:
        ordered_rows = ordered_rows[:top]

    source_files = _parse_source_files_from_metadata(metadata)
    if not source_files:
        raise ValueError("No source_file entries found in metadata")

    comments_index: Dict[str, Dict[str, Any]] = {}
    posts_map_by_file: Dict[str, Dict[str, Dict[str, Any]]] = {}
    parent_map_by_file: Dict[str, Dict[str, Dict[str, Any]]] = {}
    source_by_comment_id: Dict[str, str] = {}

    for source_file in source_files:
        comments, posts = load_comments_and_posts(source_file)
        posts_map = {p.get("post_id"): p for p in posts if p.get("post_id")}
        parent_map = build_parent_map(comments)

        posts_map_by_file[source_file] = posts_map
        parent_map_by_file[source_file] = parent_map

        for comment in comments:
            comment_id = comment.get("comment_id")
            if not comment_id:
                continue
            comments_index[comment_id] = comment
            source_by_comment_id[comment_id] = source_file

    prompts_payload = []
    for row in ordered_rows:
        prompt_number = row[0]
        comment_id = row[1]

        comment = comments_index.get(comment_id)
        if not comment:
            prompts_payload.append({
                "prompt_number": prompt_number,
                "comment_id": comment_id,
                "error": "Comment ID not found in source files",
            })
            continue

        source_file = source_by_comment_id.get(comment_id)
        posts_map = posts_map_by_file.get(source_file, {})
        parent_map = parent_map_by_file.get(source_file, {})

        parent = None
        parent_is_post = False
        parent_id = comment.get("parent_id")
        if parent_id and parent_id in parent_map:
            parent = parent_map[parent_id]
            parent_is_post = False
        elif comment.get("post_id") in posts_map:
            parent = posts_map[comment.get("post_id")]
            parent_is_post = True

        prompt = build_comment_prompt(
            comment=comment,
            parent=parent,
            parent_is_post=parent_is_post,
            source_file=source_file,
            posts_map=posts_map,
            comments_map=parent_map,
            simplify_output_template=False,
        )

        prompts_payload.append({
            "prompt_number": prompt_number,
            "comment_id": comment_id,
            "prompt": prompt,
        })

    return prompts_payload, metadata


def _parse_llm_response_to_entry(response_text: str) -> tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    candidate = response_text.strip()

    # Remove markdown code fences if present.
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()

    json_start = candidate.find("{")
    if json_start == -1:
        return None, "none", "No JSON object found in model response"

    json_text = _extract_balanced_json_object(candidate, json_start)
    if not json_text:
        json_text = _extract_balanced_json_object_loose(candidate, json_start)
    if not json_text:
        return None, "none", "Could not determine JSON boundaries in model response"

    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            return None, "strict", "Parsed JSON is not an object"
        return parsed, "strict", None
    except json.JSONDecodeError:
        repaired = _repair_json_string(json_text)
        try:
            parsed = json.loads(repaired)
            if not isinstance(parsed, dict):
                return None, "repaired", "Parsed JSON is not an object"
            return parsed, "repaired", None
        except json.JSONDecodeError as exc:
            return None, "repaired", str(exc)


def _build_strict_json_retry_prompt(original_prompt: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "IMPORTANT: Return exactly one valid JSON object and nothing else. "
        "Do not include markdown, explanations, notes, or prose before/after JSON. "
        "The response must begin with '{' and end with '}'."
    )


def _analyze_with_retries(
    provider_client: Any,
    prompt: str,
    max_attempts: int = 4,
    base_sleep_seconds: float = 4.0,
) -> str:
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return provider_client.analyze_with_text(prompt)
        except Exception as exc:
            last_exc = exc
            message = str(exc).lower()
            transient = (
                "rate limit" in message
                or "429" in message
                or "timeout" in message
                or "temporarily" in message
            )
            if not transient or attempt == max_attempts:
                raise

            sleep_seconds = base_sleep_seconds * attempt
            time.sleep(sleep_seconds)

    # Defensive fallback.
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown LLM error")


def _infer_provider(model: Optional[str], provider: Optional[str]) -> str:
    if provider:
        return provider.strip().lower()

    if model:
        model_lower = model.strip().lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"

    return DEFAULT_LLM_PROVIDER


def _default_model_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    return DEFAULT_OPENAI_MODEL


def _resolve_provider_api_key(provider: str, explicit_api_key: Optional[str], config: Config) -> Optional[str]:
    if explicit_api_key:
        return explicit_api_key

    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY") or config.get("llm.api_key")

    if provider == "openai":
        return os.getenv("OPENAI_API_KEY") or config.get("llm.api_key")

    return config.get("llm.api_key")


def run_abn_llm_analysis(
    prompts_txt_path: Optional[str] = None,
    metadata_json_path: Optional[str] = None,
    top: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run LLM analysis for ABN prompts and save extracted JSON entries.

    Args:
        prompts_txt_path: Path to abn_test_prompts_XXX.txt (legacy fallback)
        metadata_json_path: Path to abn_test_prompts_metadata_XXX.json (preferred)
        top: Optional number of prompts to process by order of appearance
        provider: LLM provider override (openai/anthropic)
        model: Model override
        api_key: API key override
        output_dir: Base output directory (e.g. data/output/scrape_003)

    Returns:
        Summary dictionary with saved file paths and counts
    """
    prompts_path: Optional[Path] = None
    prompts_payload: List[Dict[str, Any]] = []
    source_metadata: Dict[str, Any] = {}

    if metadata_json_path:
        prompts_payload, source_metadata = _build_llm_prompts_from_metadata(
            metadata_json_path=metadata_json_path,
            top=top,
        )
    else:
        if not prompts_txt_path:
            raise ValueError("Either metadata_json_path or prompts_txt_path must be provided")
        prompts_path = Path(prompts_txt_path)
        if not prompts_path.exists():
            raise ValueError(f"Prompts file not found: {prompts_txt_path}")

        prompts_text = prompts_path.read_text(encoding="utf-8")
        all_sections = _split_prompt_sections(prompts_text)
        prompt_sections = all_sections[:top] if top is not None else all_sections
        for prompt_number, prompt_body in prompt_sections:
            prompts_payload.append({
                "prompt_number": prompt_number,
                "comment_id": None,
                "prompt": _strip_existing_output(prompt_body),
            })

    config = Config()
    configured_provider = config.get("llm.provider")
    if provider:
        provider_name = _infer_provider(model=model, provider=provider)
    elif model:
        provider_name = _infer_provider(model=model, provider=None)
    else:
        provider_name = _infer_provider(model=None, provider=configured_provider)

    configured_model = config.get("llm.model")
    model_name = model or configured_model
    if not model_name:
        model_name = _default_model_for_provider(provider_name)

    provider_client = LLMProvider.from_config(
        provider_name,
        api_key=_resolve_provider_api_key(provider_name, api_key, config),
        model=model_name,
    )

    prompts_considered = len(prompts_payload)

    llm_entries = []
    parse_errors = []
    validation_warnings = []
    observed_models: List[str] = []
    token_usage_per_prompt: List[Dict[str, Any]] = []

    for item in prompts_payload:
        prompt_number = item.get("prompt_number")
        comment_id = item.get("comment_id")
        response_text: Optional[str] = None
        token_usage_entry: Dict[str, Any] = {
            "prompt_number": prompt_number,
            "comment_id": comment_id,
            "model_used": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "status": "pending",
        }

        if item.get("error"):
            parse_errors.append({
                "prompt_number": prompt_number,
                "comment_id": comment_id,
                "error": item.get("error"),
            })
            token_usage_entry["status"] = "prompt_error"
            token_usage_entry["error"] = item.get("error")
            token_usage_per_prompt.append(token_usage_entry)
            continue

        llm_prompt = item.get("prompt", "")
        if not llm_prompt:
            parse_errors.append({
                "prompt_number": prompt_number,
                "comment_id": comment_id,
                "error": "Empty prompt",
            })
            token_usage_entry["status"] = "empty_prompt"
            token_usage_entry["error"] = "Empty prompt"
            token_usage_per_prompt.append(token_usage_entry)
            continue

        parsed = None
        parse_mode = "none"
        error = None

        try:
            response_text = _analyze_with_retries(provider_client, llm_prompt)
        except Exception as exc:
            parse_errors.append({
                "prompt_number": prompt_number,
                "comment_id": comment_id,
                "error": f"LLM call failed: {str(exc)}",
            })
            token_usage_entry["status"] = "llm_call_failed"
            token_usage_entry["error"] = str(exc)
            token_usage_per_prompt.append(token_usage_entry)
            continue

        resolved_model = getattr(provider_client, "last_response_model", None)
        if isinstance(resolved_model, str) and resolved_model and resolved_model not in observed_models:
            observed_models.append(resolved_model)
        if isinstance(resolved_model, str) and resolved_model:
            token_usage_entry["model_used"] = resolved_model

        provider_usage = getattr(provider_client, "last_response_usage", None)
        if isinstance(provider_usage, dict):
            token_usage_entry["prompt_tokens"] = provider_usage.get("prompt_tokens")
            token_usage_entry["completion_tokens"] = provider_usage.get("completion_tokens")
            token_usage_entry["total_tokens"] = provider_usage.get("total_tokens")

        parsed, parse_mode, error = _parse_llm_response_to_entry(response_text)
        if error or parsed is None:
            parse_error_entry: Dict[str, Any] = {
                "prompt_number": prompt_number,
                "comment_id": comment_id,
                "error": error or "Unknown parse error",
            }

            error_text = parse_error_entry["error"]
            if isinstance(error_text, str) and (
                "No JSON object found" in error_text
                or "Could not determine JSON boundaries" in error_text
            ):
                if isinstance(response_text, str):
                    parse_error_entry["model_response_text"] = response_text

            parse_errors.append(parse_error_entry)
            token_usage_entry["status"] = "parse_error"
            token_usage_entry["error"] = parse_error_entry["error"]
            token_usage_per_prompt.append(token_usage_entry)
            continue

        normalized = _validate_and_normalize_entry(parsed)
        if normalized["issues"]:
            validation_warnings.append({
                "prompt_number": prompt_number,
                "issues": normalized["issues"],
            })

        llm_entries.append({
            "prompt_number": prompt_number,
            "comment_id": comment_id,
            "parse_mode": parse_mode,
            "entry": normalized["entry"],
        })
        token_usage_entry["status"] = "ok"
        token_usage_per_prompt.append(token_usage_entry)

    model_used_value: Any = None
    if len(observed_models) == 1:
        model_used_value = observed_models[0]
    elif len(observed_models) > 1:
        model_used_value = observed_models

    if output_dir:
        base_output_path = Path(output_dir)
    else:
        if prompts_path and prompts_path.parent.name == "abn_test":
            base_output_path = prompts_path.parent.parent
        elif metadata_json_path and Path(metadata_json_path).parent.name == "abn_test":
            base_output_path = Path(metadata_json_path).parent.parent
        else:
            base_output_path = Path(OUTPUT_DIR)

    abn_test_dir = _get_abn_test_output_dir(str(base_output_path))
    run_number = _get_next_number_for_pattern(abn_test_dir, "abn_test_llm_entries", extension="json")

    entries_path = abn_test_dir / f"abn_test_llm_entries_{run_number:03d}.json"
    metadata_path = abn_test_dir / f"abn_test_llm_entries_metadata_{run_number:03d}.json"

    with open(entries_path, "w", encoding="utf-8") as f:
        json.dump(llm_entries, f, indent=2)

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "function": "run_abn_llm_analysis",
        "prompts_txt_path": str(prompts_path) if prompts_path else None,
        "metadata_json_path": metadata_json_path,
        "top": top,
        "provider": provider_name,
        "model": model_name,
        "model_requested": model_name,
        "model_used": model_used_value,
        "total_prompts_in_file": source_metadata.get("selected_comment_ids") and len(source_metadata.get("selected_comment_ids", [])) or prompts_considered,
        "prompts_considered": prompts_considered,
        "entries_extracted": len(llm_entries),
        "entries_parse_errors": len(parse_errors),
        "token_usage_per_prompt": token_usage_per_prompt,
        "validation_warnings": validation_warnings,
        "parse_errors": parse_errors,
        "output_file": str(entries_path),
        "metadata_file": str(metadata_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "success",
        "output_file": str(entries_path),
        "metadata_file": str(metadata_path),
        "entries_extracted": len(llm_entries),
        "entries_parse_errors": len(parse_errors),
        "validation_warnings": len(validation_warnings),
        "provider": provider_name,
        "model": model_name,
    }


def load_comments_from_raw(file_path: str) -> Dict[str, Any]:
    """Load comments from raw scraped JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_reply_map(comments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a map of comment_id -> list of replies, sorted by score descending.
    """
    reply_map = {}
    for comment in comments:
        parent_id = comment.get("parent_id")
        if parent_id not in reply_map:
            reply_map[parent_id] = []
        reply_map[parent_id].append(comment)

    for parent_id in reply_map:
        reply_map[parent_id].sort(key=lambda c: c.get("upvotes", 0), reverse=True)

    return reply_map


def find_top_level_comments(
    comments: List[Dict[str, Any]],
    post_id: str
) -> List[Dict[str, Any]]:
    """Find all top-level comments (those whose parent_id is the post_id)."""
    return [c for c in comments if c.get("parent_id") == post_id]


def follow_reply_chain(
    start_comment: Dict[str, Any],
    reply_map: Dict[str, List[Dict[str, Any]]],
    max_depth: int = 100
) -> List[Dict[str, Any]]:
    """Follow a reply chain from a top-level comment, taking the best reply at each level."""
    chain = [start_comment]
    current_id = start_comment.get("comment_id")
    depth = 0

    while depth < max_depth:
        if current_id not in reply_map or not reply_map[current_id]:
            break

        next_comment = reply_map[current_id][0]
        chain.append(next_comment)
        current_id = next_comment.get("comment_id")
        depth += 1

    return chain


def select_comment_chains(
    raw_file_path: str,
    num_comments: int = 50,
    post_id: Optional[str] = None,
    random_seed: Optional[int] = None
) -> tuple[List[List[Dict[str, Any]]], int, Dict[str, Dict[str, Any]]]:
    """Select random top-level comments and build their reply chains."""
    if random_seed is None:
        random_seed = int(time.time() * 1000) % (2**31)

    random.seed(random_seed)

    raw_data = load_comments_from_raw(raw_file_path)
    comments = raw_data.get("comments", [])
    posts = raw_data.get("posts", [])

    if not comments:
        raise ValueError(f"No comments found in {raw_file_path}")

    reply_map = build_reply_map(comments)
    posts_map = {post.get("post_id"): post for post in posts if post.get("post_id")}

    if post_id is None and posts:
        post_id = posts[0].get("post_id")

    if not post_id:
        raise ValueError("Could not determine post_id")

    top_level = find_top_level_comments(comments, post_id)
    if not top_level:
        raise ValueError(f"No top-level comments found for post {post_id}")

    chains = []
    total_collected = 0
    selection_attempts = 0
    max_attempts = len(top_level) * 3
    selected_top_level_ids = set()

    while selection_attempts < max_attempts:
        if num_comments is not None and total_collected >= num_comments:
            break

        top_level_comment = random.choice(top_level)
        top_level_id = top_level_comment.get("comment_id")

        if top_level_id in selected_top_level_ids:
            selection_attempts += 1
            continue

        selected_top_level_ids.add(top_level_id)
        chain = follow_reply_chain(top_level_comment, reply_map)
        chains.append(chain)
        total_collected += len(chain)
        selection_attempts += 1

    return chains, random_seed, posts_map


def build_prompts_for_chains(
    chains: List[List[Dict[str, Any]]],
    subreddit: str = "unknown",
    posts_map: Optional[Dict[str, Dict[str, Any]]] = None,
    post_source_map: Optional[Dict[str, str]] = None,
    post_subreddit_map: Optional[Dict[str, str]] = None
) -> List[tuple[str, Dict[str, Any]]]:
    """Build analysis prompts for each comment in the chains."""
    prompts_data = []

    for chain in chains:
        chain_comments_map = {c.get("comment_id"): c for c in chain if c.get("comment_id")}
        chain_root_id = chain[0].get("comment_id", "") if chain else ""
        for i, comment in enumerate(chain):
            if i == 0:
                parent_post = posts_map.get(comment.get("post_id")) if posts_map else None
                parent_comment = parent_post
                parent_is_post = True
            else:
                parent_comment = chain[i - 1]
                parent_is_post = False

            source_file = post_source_map.get(comment.get("post_id", "")) if post_source_map else None
            prompt = build_comment_prompt(
                comment=comment,
                parent=parent_comment,
                parent_is_post=parent_is_post,
                source_file=source_file,
                posts_map=posts_map,
                comments_map=chain_comments_map,
                simplify_output_template=True,
            )

            comment_id = comment.get("comment_id", "")
            comment_subreddit = subreddit
            if post_subreddit_map:
                comment_subreddit = post_subreddit_map.get(comment.get("post_id", ""), subreddit)

            prompts_data.append((
                prompt,
                {
                    "comment_id": comment_id,
                    "subreddit": comment_subreddit,
                    "post_id": comment.get("post_id", ""),
                    "chain_root_id": chain_root_id,
                    "chain_position": i,
                    "is_top_level": (i == 0),
                    "parent_id": comment.get("parent_id", ""),
                    "score": comment.get("upvotes", 0)
                }
            ))

    return prompts_data


def save_abn_test_prompts(
    chains: List[List[Dict[str, Any]]],
    prompts_data: List[tuple[str, Dict[str, Any]]],
    output_dir: str = OUTPUT_DIR,
    run_number: Optional[int] = None,
    random_seed: int = None,
    source_file: str = ""
) -> tuple[str, str]:
    """Save A/B/n test prompts and metadata."""
    abn_test_dir = _get_abn_test_output_dir(output_dir)
    abn_test_dir.mkdir(parents=True, exist_ok=True)

    if run_number is None:
        run_number = _get_next_number_for_pattern(
            abn_test_dir,
            "abn_test_prompts",
            extension="txt",
        )

    prompts_txt_path = abn_test_dir / f"abn_test_prompts_{run_number:03d}.txt"
    separator = "\n\n" + ("=" * 80) + "\n"
    prompts_with_numbers = []
    for idx, (prompt, _) in enumerate(prompts_data, start=1):
        prompts_with_numbers.append(f"Prompt # {idx}\n{prompt}")
    prompts_text = separator.join(prompts_with_numbers)

    with open(prompts_txt_path, 'w', encoding='utf-8') as f:
        f.write(prompts_text)

    comment_ids = [
        [
            idx,
            metadata["comment_id"],
            metadata.get("subreddit", "unknown")
        ]
        for idx, (_, metadata) in enumerate(prompts_data, start=1)
    ]

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "source_file": source_file,
        "random_seed": random_seed,
        "selected_comment_ids": comment_ids,
    }

    metadata_json_path = abn_test_dir / f"abn_test_prompts_metadata_{run_number:03d}.json"
    with open(metadata_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return str(prompts_txt_path), str(metadata_json_path)


def generate_abn_test_set(
    raw_folder_path: str,
    num_comments: int = 50,
    random_seed: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
    output_dir: str = OUTPUT_DIR
) -> Dict[str, Any]:
    """Generate a combined A/B/n test set from all JSON files in a folder."""
    if not provider:
        config = Config()
        provider_name = config.get("llm.provider") or DEFAULT_LLM_PROVIDER
        api_key = config.get("llm.api_key")
        provider = LLMProvider.from_config(provider_name, api_key=api_key)

    raw_folder = Path(raw_folder_path)
    if not raw_folder.exists():
        raise ValueError(f"raw_folder_path not found: {raw_folder_path}")

    output_dir = str(Path(output_dir) / raw_folder.name)

    json_files = []
    for path in sorted(raw_folder.glob("*.json")):
        if not path.is_file():
            continue
        name = path.name
        if name == "subreddits_info.json":
            continue
        if name.endswith("_metadata.json"):
            continue
        if name.endswith("_scraper_metadata.json"):
            continue
        if "preprocess_metadata" in name:
            continue
        if name.startswith("comment_prompts_") or name.startswith("comment_analysis_"):
            continue
        if name.startswith("abn_test_prompts_") or name.startswith("abn_test_analysis_"):
            continue
        json_files.append(path)

    if not json_files:
        raise ValueError(f"No JSON files found in {raw_folder_path}")

    print(f"Found {len(json_files)} comment files: {', '.join(f.name for f in json_files)}")

    chains_by_file = []
    all_posts_map = {}
    post_source_map = {}
    post_subreddit_map = {}
    source_files = []

    for json_file in json_files:
        print(f"\nLoading {json_file.name}...")

        try:
            chains, seed_used, posts_map = select_comment_chains(
                str(json_file),
                num_comments=None,
                post_id=None,
                random_seed=random_seed
            )

            all_posts_map.update(posts_map)
            for post_id, post in posts_map.items():
                post_source_map[post_id] = str(json_file)
                if isinstance(post, dict) and post.get("community"):
                    post_subreddit_map[post_id] = post.get("community")
                else:
                    post_subreddit_map[post_id] = json_file.stem

            chains_by_file.append({
                "file": json_file,
                "chains": chains,
                "subreddit": json_file.stem
            })
            source_files.append(str(json_file))

            print(f"  Loaded {sum(len(c) for c in chains)} comments in {len(chains)} chains")

        except Exception as e:
            print(f"  [ERROR] Failed to load {json_file.name}: {str(e)}")

    total_chains = sum(len(entry["chains"]) for entry in chains_by_file)
    if total_chains == 0:
        raise ValueError("No chains could be loaded from any files")

    for entry in chains_by_file:
        random.shuffle(entry["chains"])

    total = 0
    limited_chains = []
    selected_any = True
    while total < num_comments and selected_any:
        selected_any = False
        for entry in chains_by_file:
            if not entry["chains"]:
                continue
            chain = entry["chains"].pop(0)
            limited_chains.append(chain)
            total += len(chain)
            selected_any = True
            if total >= num_comments:
                break

    def _chain_sort_key(chain: List[Dict[str, Any]]) -> tuple:
        if not chain:
            return ("", "", "")
        post_id = chain[0].get("post_id", "")
        subreddit_name = post_subreddit_map.get(post_id, "")
        root_id = chain[0].get("comment_id", "")
        return (subreddit_name, post_id, root_id)

    limited_chains = sorted(limited_chains, key=_chain_sort_key)

    print(f"\nSelected {sum(len(c) for c in limited_chains)} comments in {len(limited_chains)} chains")
    print(f"Random seed: {seed_used}")

    prompts_data = build_prompts_for_chains(
        limited_chains,
        subreddit="combined",
        posts_map=all_posts_map,
        post_source_map=post_source_map,
        post_subreddit_map=post_subreddit_map
    )

    prompts_path, metadata_path = save_abn_test_prompts(
        limited_chains,
        prompts_data,
        output_dir=output_dir,
        random_seed=seed_used,
        source_file=", ".join(source_files)
    )

    result = {
        "status": "success",
        "prompt_file": prompts_path,
        "metadata_file": metadata_path,
        "random_seed": seed_used,
        "source_files": source_files,
        "num_chains": len(limited_chains),
        "total_comments": sum(len(c) for c in limited_chains),
    }

    print(f"\n{'=' * 80}")
    print("A/B/n test set created successfully!")
    print(f"Prompts: {prompts_path}")
    print(f"Metadata: {metadata_path}")

    return result
